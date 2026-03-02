"""Domain agent for processing Airbnb-related owner emails.

Classifies incoming emails, applies policy-based handling, and generates
draft responses with owner approval workflows (HITL).  Uses a
LangChain-first architecture with composable RunnableLambda stages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableLambda

from app.agents.base import Agent, AgentResult
from app.schemas import StepLog
from app.services.chat_service import ChatService
from app.services.gmail_service import EmailMessage, GmailService

NO_MAIL_RESPONSE = "No actionable Airbnb emails found in your inbox."

# ---------------------------------------------------------------------------
# Classification constants
# ---------------------------------------------------------------------------

CATEGORY_GUEST_MESSAGE = "guest_message"
CATEGORY_LEAVE_REVIEW = "leave_review_request"
CATEGORY_NEW_PROPERTY_REVIEW = "new_property_review"
CATEGORY_UNSUPPORTED_AIRBNB = "unsupported_airbnb"
CATEGORY_NON_AIRBNB = "non_airbnb"

IMPORTANCE_HIGH = "high"
IMPORTANCE_LOW = "low"
IMPORTANCE_UNCERTAIN = "uncertain"


@dataclass
class MailAgentConfig:
    """Configuration knobs for mail agent policy and thresholds."""

    bad_review_threshold: int = 3
    max_inbox_fetch: int = 20
    auto_send_enabled: bool = False
    max_answer_words: int = 200


@dataclass
class ClassifiedEmail:
    """An email enriched with classification metadata."""

    message: EmailMessage
    category: str
    confidence: float
    extracted_rating: int | None = None
    extracted_guest_name: str | None = None


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

_LEAVE_REVIEW_SIGNALS = [
    "leave a review",
    "leave review",
    "review your guest",
    "share your experience hosting",
    "rate your experience",
    "write a review for",
]

_NEW_REVIEW_SIGNALS = [
    r"left a review",
    r"new review",
    r"reviewed your property",
    r"left a.*star review",
    r"you can respond to this review",
]

_GUEST_MESSAGE_SIGNALS = [
    r"message from",
    r"new message",
    r"guest.*sent",
    r"inquiry",
    r"booking request",
    r"question from",
]

_HIGH_IMPORTANCE_SIGNALS = [
    "urgent",
    "emergency",
    "locked out",
    "can't get in",
    "no hot water",
    "broken",
    "leak",
    "flood",
    "fire",
    "safety",
    "dangerous",
    "cancel",
    "refund",
    "complaint",
    "police",
    "ambulance",
]

_MEDIUM_IMPORTANCE_SIGNALS = [
    "check-in",
    "check-out",
    "early",
    "late",
    "key",
    "wifi",
    "password",
    "directions",
    "parking",
    "question",
    "help",
    "issue",
]

_LOW_IMPORTANCE_SIGNALS = [
    "thank",
    "thanks",
    "looking forward",
    "excited",
    "appreciate",
    "wonderful",
]


def _extract_rating(text: str) -> int | None:
    """Extract a 1-5 star rating from email body/subject."""
    for pattern in [r"(\d)\s*/\s*5\s*stars?", r"(?<!/)(\d)\s*stars?", r"[Rr]ating:\s*(\d)"]:
        match = re.search(pattern, text)
        if match:
            rating = int(match.group(1))
            if 1 <= rating <= 5:
                return rating
    return None


def _extract_guest_name(text: str) -> str | None:
    """Extract a guest name from common Airbnb notification patterns."""
    for pattern in [
        r"(?:from|for)\s+(?:guest\s+|your\s+guest\s+)?([A-Z]\w+)",
        r"(?:review\s+from\s+(?:guest\s+)?)([A-Z]\w+)",
        r"([A-Z]\w+)(?:'s\s+stay)",
        r"([A-Z]\w+)\s+left\s+a\s+(?:\d-star\s+)?review",
    ]:
        match = re.search(pattern, text)
        if match:
            name = match.group(1).strip()
            if name.lower() not in {"your", "the", "a", "an", "new", "leave"}:
                return name
    return None


def _classify_email(email: EmailMessage, gmail_service: GmailService) -> ClassifiedEmail:
    """Classify a single email into one of the Airbnb workflow categories."""

    if not gmail_service.is_airbnb_sender(email.sender):
        return ClassifiedEmail(message=email, category=CATEGORY_NON_AIRBNB, confidence=1.0)

    combined = f"{email.subject.lower()} {email.body.lower()}"
    guest_name = _extract_guest_name(f"{email.subject} {email.body}")
    rating = _extract_rating(f"{email.subject} {email.body}")

    if any(sig in combined for sig in _LEAVE_REVIEW_SIGNALS):
        return ClassifiedEmail(
            message=email,
            category=CATEGORY_LEAVE_REVIEW,
            confidence=0.9,
            extracted_guest_name=guest_name,
        )

    if any(re.search(sig, combined) for sig in _NEW_REVIEW_SIGNALS):
        return ClassifiedEmail(
            message=email,
            category=CATEGORY_NEW_PROPERTY_REVIEW,
            confidence=0.9,
            extracted_rating=rating,
            extracted_guest_name=guest_name,
        )

    if any(re.search(sig, combined) for sig in _GUEST_MESSAGE_SIGNALS):
        return ClassifiedEmail(
            message=email,
            category=CATEGORY_GUEST_MESSAGE,
            confidence=0.85,
            extracted_guest_name=guest_name,
        )

    return ClassifiedEmail(message=email, category=CATEGORY_UNSUPPORTED_AIRBNB, confidence=0.5)


def _score_importance(cls: ClassifiedEmail) -> str:
    """Score importance of a guest message for triage."""
    combined = f"{cls.message.subject.lower()} {cls.message.body.lower()}"

    if any(s in combined for s in _HIGH_IMPORTANCE_SIGNALS):
        return IMPORTANCE_HIGH

    medium_hits = sum(1 for s in _MEDIUM_IMPORTANCE_SIGNALS if s in combined)
    if medium_hits >= 2:
        return IMPORTANCE_HIGH

    if any(s in combined for s in _LOW_IMPORTANCE_SIGNALS) and medium_hits == 0:
        return IMPORTANCE_LOW

    return IMPORTANCE_UNCERTAIN


def _cap_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


# ---------------------------------------------------------------------------
# LLM prompt builders
# ---------------------------------------------------------------------------


def _guest_reply_prompts(cls: ClassifiedEmail) -> tuple[str, str]:
    system = (
        "You are a professional, friendly Airbnb host assistant. "
        "Draft a polite, helpful reply to the guest message below. "
        "Be concise and address their question directly. "
        "Do not promise anything you cannot deliver. "
        "Keep the tone warm but professional."
    )
    guest_name = cls.extracted_guest_name or "Guest"
    user = (
        f"Guest name: {guest_name}\n"
        f"Subject: {cls.message.subject}\n"
        f"Message: {cls.message.body}\n\n"
        "Draft a reply:"
    )
    return system, user


def _positive_review_prompts(guest_name: str) -> tuple[str, str]:
    system = (
        "You are an Airbnb host writing a review for a guest who had a good stay. "
        "Write a brief, genuine, positive review. Keep it under 50 words. "
        "Be specific enough to feel authentic but not overly effusive."
    )
    user = f"Write a positive review for guest: {guest_name}"
    return system, user


def _detailed_review_prompts(
    guest_name: str,
    rating: int,
    issues: list[str] | None = None,
    free_text: str | None = None,
) -> tuple[str, str]:
    system = (
        "You are an Airbnb host writing an honest review for a guest. "
        "Be fair and factual. Mention specific issues but keep a professional tone. "
        "Keep it under 80 words."
    )
    issue_section = ""
    if issues:
        issue_section = f"Issues noted: {', '.join(issues)}\n"
    if free_text:
        issue_section += f"Additional notes: {free_text}\n"
    user = (
        f"Guest: {guest_name}\n"
        f"Rating given: {rating}/5\n"
        f"{issue_section}"
        "Write an honest, professional review:"
    )
    return system, user


def _review_response_prompts(guest_name: str, rating: int, review_text: str) -> tuple[str, str]:
    system = (
        "You are an Airbnb host responding to a guest review. "
        "Be professional and gracious. Thank them for feedback. "
        "If the review is positive, keep it warm and brief. "
        "If negative, acknowledge concerns and mention improvement plans. "
        "Keep under 60 words."
    )
    user = (
        f"Guest: {guest_name}\n"
        f"Rating: {rating}/5\n"
        f"Review: {review_text}\n\n"
        "Draft your response:"
    )
    return system, user


# ---------------------------------------------------------------------------
# LangChain-first mail pipeline
# ---------------------------------------------------------------------------


class MailPipeline:
    """LangChain-first mail pipeline composed of RunnableLambda stages.

    Processes the owner's inbox: filters for Airbnb emails, classifies them,
    applies policy-based handling, and compiles the answer.
    """

    def __init__(
        self,
        *,
        gmail_service: GmailService,
        chat_service: ChatService,
        config: MailAgentConfig,
    ) -> None:
        self._gmail = gmail_service
        self._chat = chat_service
        self._config = config

        self.fetch_inbox = RunnableLambda(self._fetch_inbox_stage).with_config(
            run_name="FetchInbox",
        )
        self.filter_airbnb = RunnableLambda(self._filter_airbnb_stage).with_config(
            run_name="FilterAirbnb",
        )
        self.classify_emails = RunnableLambda(self._classify_emails_stage).with_config(
            run_name="ClassifyEmails",
        )
        self.apply_policies = RunnableLambda(self._apply_policies_stage).with_config(
            run_name="ApplyPolicies",
        )
        self.build_answer = RunnableLambda(self._build_answer_stage).with_config(
            run_name="BuildAnswer",
        )

    # -- state merge helper --------------------------------------------------------

    @staticmethod
    def _apply(state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        merged = dict(state)
        new_steps = updates.pop("steps", [])
        merged.update(updates)
        merged["steps"] = merged.get("steps", []) + new_steps
        return merged

    # -- main orchestration --------------------------------------------------------

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        state = self._apply(state, self.fetch_inbox.invoke(state))
        if not state.get("raw_messages"):
            return state

        state = self._apply(state, self.filter_airbnb.invoke(state))
        if not state.get("airbnb_messages"):
            return state

        state = self._apply(state, self.classify_emails.invoke(state))
        state = self._apply(state, self.apply_policies.invoke(state))
        state = self._apply(state, self.build_answer.invoke(state))
        return state

    def invoke_with_messages(
        self, state: dict[str, Any], messages: list[EmailMessage]
    ) -> dict[str, Any]:
        """Run pipeline on a pre-fetched list of messages (e.g. from push). Skips fetch_inbox."""
        state = dict(state)
        state["raw_messages"] = messages
        state = self._apply(
            state,
            {
                "steps": [
                    StepLog(
                        module="mail_agent.push_fetch",
                        prompt={"source": "push", "count": len(messages)},
                        response={"status": "ok", "source": "push", "count": len(messages)},
                    )
                ],
            },
        )
        if not messages:
            return state

        state = self._apply(state, self.filter_airbnb.invoke(state))
        if not state.get("airbnb_messages"):
            return state

        state = self._apply(state, self.classify_emails.invoke(state))
        state = self._apply(state, self.apply_policies.invoke(state))
        state = self._apply(state, self.build_answer.invoke(state))
        return state

    # -- stage implementations -----------------------------------------------------

    def _fetch_inbox_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        cfg = self._config
        if not self._gmail.is_available:
            return {
                "raw_messages": [],
                "answer": (
                    "Mail agent is not configured. "
                    "Set MAIL_ENABLED=true to enable email processing."
                ),
                "steps": [
                    StepLog(
                        module="mail_agent.fetch_inbox",
                        prompt={"max_results": cfg.max_inbox_fetch},
                        response={"status": "unavailable", "reason": "Gmail service not configured"},
                    )
                ],
            }

        try:
            messages = self._gmail.list_unread_messages(max_results=cfg.max_inbox_fetch)
            return {
                "raw_messages": messages,
                "steps": [
                    StepLog(
                        module="mail_agent.fetch_inbox",
                        prompt={"max_results": cfg.max_inbox_fetch},
                        response={
                            "status": "ok",
                            "fetched_count": len(messages),
                            "demo_mode": self._gmail.is_demo_mode,
                        },
                    )
                ],
            }
        except Exception as exc:
            return {
                "raw_messages": [],
                "answer": f"Failed to fetch inbox: {type(exc).__name__}: {exc}",
                "steps": [
                    StepLog(
                        module="mail_agent.fetch_inbox",
                        prompt={"max_results": cfg.max_inbox_fetch},
                        response={"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                    )
                ],
            }

    def _filter_airbnb_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        raw_messages: list[EmailMessage] = state.get("raw_messages") or []
        airbnb: list[EmailMessage] = []
        non_airbnb_count = 0
        self_sent_count = 0

        for msg in raw_messages:
            # Skip messages sent by the owner (SENT label) to prevent reply loops.
            if "SENT" in msg.labels:
                self_sent_count += 1
                continue
            if self._gmail.is_airbnb_sender(msg.sender):
                airbnb.append(msg)
            else:
                non_airbnb_count += 1

        result: dict[str, Any] = {
            "airbnb_messages": airbnb,
            "steps": [
                StepLog(
                    module="mail_agent.airbnb_filter",
                    prompt={"total_messages": len(raw_messages)},
                    response={
                        "airbnb_count": len(airbnb),
                        "non_airbnb_ignored": non_airbnb_count,
                        "self_sent_skipped": self_sent_count,
                    },
                )
            ],
        }
        if not airbnb:
            result["answer"] = NO_MAIL_RESPONSE
        return result

    def _classify_emails_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        airbnb_messages: list[EmailMessage] = state.get("airbnb_messages") or []
        classified: list[ClassifiedEmail] = []
        categories: dict[str, int] = {}

        for msg in airbnb_messages:
            cls = _classify_email(msg, self._gmail)
            classified.append(cls)
            categories[cls.category] = categories.get(cls.category, 0) + 1

        return {
            "classified_emails": classified,
            "steps": [
                StepLog(
                    module="mail_agent.classify",
                    prompt={"email_count": len(airbnb_messages)},
                    response={
                        "classified_count": len(classified),
                        "category_breakdown": categories,
                    },
                )
            ],
        }

    def _apply_policies_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        classified: list[ClassifiedEmail] = state.get("classified_emails") or []
        owner_action = state.get("owner_action")
        cfg = self._config
        actions: list[dict[str, Any]] = []
        policy_steps: list[StepLog] = []

        for cls_email in classified:
            if cls_email.category == CATEGORY_GUEST_MESSAGE:
                action, steps = self._handle_guest_message(cls_email)
            elif cls_email.category == CATEGORY_LEAVE_REVIEW:
                action, steps = self._handle_leave_review(cls_email, owner_action)
            elif cls_email.category == CATEGORY_NEW_PROPERTY_REVIEW:
                action, steps = self._handle_property_review(cls_email, owner_action, cfg)
            elif cls_email.category == CATEGORY_UNSUPPORTED_AIRBNB:
                action = {
                    "email_id": cls_email.message.id,
                    "category": cls_email.category,
                    "action": "no_action",
                    "reason": "Unsupported Airbnb email subtype",
                    "requires_owner": False,
                }
                steps = [
                    StepLog(
                        module="mail_agent.policy",
                        prompt={"email_id": cls_email.message.id, "category": cls_email.category},
                        response={"action": "no_action", "reason": "unsupported_subtype"},
                    )
                ]
            else:
                continue

            actions.append(action)
            policy_steps.extend(steps)

        return {"mail_actions": actions, "steps": policy_steps}

    # -- Guest message flow --------------------------------------------------------

    def _handle_guest_message(
        self, cls_email: ClassifiedEmail
    ) -> tuple[dict[str, Any], list[StepLog]]:
        importance = _score_importance(cls_email)
        msg = cls_email.message
        llm_steps: list[StepLog] = []

        action: dict[str, Any] = {
            "email_id": msg.id,
            "thread_id": msg.thread_id,
            "reply_to": msg.sender,
            "subject": msg.subject,
            "in_reply_to": getattr(msg, "message_id_header", None) or None,
            "references": getattr(msg, "references", None) or None,
            "category": CATEGORY_GUEST_MESSAGE,
            "importance": importance,
            "guest_name": cls_email.extracted_guest_name,
        }

        if importance == IMPORTANCE_HIGH:
            action.update(
                action="owner_heads_up",
                reason="High-importance guest message requires owner attention",
                requires_owner=True,
                draft=None,
            )
        elif importance == IMPORTANCE_LOW:
            draft_text = self._try_generate(
                *_guest_reply_prompts(cls_email),
                fallback="Thank you for your message. I'll get back to you shortly.",
                module="mail_agent.guest_reply_generation",
                llm_steps=llm_steps,
            )
            action.update(
                action="draft_reply",
                reason="Low-importance message; draft reply generated for approval",
                requires_owner=True,
                draft=draft_text,
            )
        else:
            action.update(
                action="owner_consult",
                reason="Uncertain importance; owner consultation needed",
                requires_owner=True,
                draft=None,
            )

        step = StepLog(
            module="mail_agent.guest_message_policy",
            prompt={"email_id": msg.id, "importance": importance},
            response={"action": action["action"], "has_draft": action.get("draft") is not None},
        )
        return action, llm_steps + [step]

    # -- Leave-review flow ---------------------------------------------------------

    def _handle_leave_review(
        self,
        cls_email: ClassifiedEmail,
        owner_action: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], list[StepLog]]:
        guest_name = cls_email.extracted_guest_name or "Guest"
        msg = cls_email.message
        llm_steps: list[StepLog] = []
        action: dict[str, Any] = {
            "email_id": msg.id,
            "thread_id": msg.thread_id,
            "reply_to": msg.sender,
            "subject": msg.subject,
            "in_reply_to": getattr(msg, "message_id_header", None) or None,
            "references": getattr(msg, "references", None) or None,
            "category": CATEGORY_LEAVE_REVIEW,
            "guest_name": guest_name,
        }

        if owner_action and owner_action.get("email_id") == msg.id:
            rating = owner_action.get("rating")
            if isinstance(rating, int) and 1 <= rating <= 5:
                if rating >= 3:
                    sys_p, usr_p = _positive_review_prompts(guest_name)
                    draft = self._try_generate(
                        sys_p, usr_p,
                        fallback=f"Great guest! {guest_name} was wonderful to host.",
                        module="mail_agent.leave_review_generation",
                        llm_steps=llm_steps,
                    )
                else:
                    issues = owner_action.get("issues", [])
                    free_text = owner_action.get("free_text")
                    sys_p, usr_p = _detailed_review_prompts(guest_name, rating, issues, free_text)
                    draft = self._try_generate(
                        sys_p, usr_p,
                        fallback=f"Had some challenges hosting {guest_name}.",
                        module="mail_agent.leave_review_generation",
                        llm_steps=llm_steps,
                    )
                action.update(
                    action="review_draft_ready",
                    rating=rating,
                    draft=draft,
                    requires_owner=True,
                )
                step = StepLog(
                    module="mail_agent.leave_review_policy",
                    prompt={"email_id": msg.id, "guest_name": guest_name, "owner_rating": rating},
                    response={
                        "action": "review_draft_ready",
                        "rating_tier": "positive" if rating >= 3 else "negative",
                        "has_draft": True,
                    },
                )
                return action, llm_steps + [step]

        action.update(
            action="awaiting_owner_rating",
            reason="Owner must provide a 1-5 rating for this guest",
            requires_owner=True,
            draft=None,
        )
        step = StepLog(
            module="mail_agent.leave_review_policy",
            prompt={"email_id": msg.id, "guest_name": guest_name},
            response={"action": "awaiting_owner_rating"},
        )
        return action, [step]

    # -- Property-review flow ------------------------------------------------------

    def _handle_property_review(
        self,
        cls_email: ClassifiedEmail,
        owner_action: dict[str, Any] | None,
        cfg: MailAgentConfig,
    ) -> tuple[dict[str, Any], list[StepLog]]:
        guest_name = cls_email.extracted_guest_name or "Guest"
        rating = cls_email.extracted_rating
        msg = cls_email.message
        llm_steps: list[StepLog] = []
        action: dict[str, Any] = {
            "email_id": msg.id,
            "thread_id": msg.thread_id,
            "reply_to": msg.sender,
            "subject": msg.subject,
            "in_reply_to": getattr(msg, "message_id_header", None) or None,
            "references": getattr(msg, "references", None) or None,
            "category": CATEGORY_NEW_PROPERTY_REVIEW,
            "guest_name": guest_name,
            "rating": rating,
        }

        is_bad = rating is not None and rating <= cfg.bad_review_threshold

        if owner_action and owner_action.get("email_id") == msg.id and owner_action.get("don_t_reply"):
            action.update(
                action="owner_chose_dont_reply",
                requires_owner=False,
                draft=None,
                reason="Owner chose not to reply",
            )
            step = StepLog(
                module="mail_agent.property_review_policy",
                prompt={"email_id": msg.id, "don_t_reply": True},
                response={"action": "owner_chose_dont_reply"},
            )
            return action, [step]

        if is_bad:
            action.update(
                action="owner_consult_required",
                reason=(
                    f"Review rating ({rating}/5) is at or below threshold "
                    f"({cfg.bad_review_threshold}). Owner must review before any response."
                ),
                requires_owner=True,
            )
            if owner_action and owner_action.get("email_id") == msg.id:
                if owner_action.get("approved") or owner_action.get("reply_style"):
                    owner_instructions = owner_action.get("owner_instructions") or ""
                    sys_p, usr_p = _review_response_prompts(guest_name, rating, msg.body)
                    if owner_instructions:
                        usr_p += f"\n\nOwner instructions: {owner_instructions}"
                    draft = self._try_generate(
                        sys_p, usr_p,
                        fallback="Thank you for your feedback.",
                        module="mail_agent.property_review_response_generation",
                        llm_steps=llm_steps,
                    )
                    reply_style = owner_action.get("reply_style")
                    if reply_style:
                        opts = self._generate_review_reply_options(
                            guest_name, rating, msg.body, llm_steps=llm_steps,
                        )
                        for opt in opts:
                            if opt.get("style") == reply_style:
                                draft = opt.get("draft", draft)
                                break
                    action.update(action="response_draft_ready", draft=draft)
                else:
                    reply_options = self._generate_review_reply_options(
                        guest_name, rating, msg.body, llm_steps=llm_steps,
                    )
                    action.update(reply_options=reply_options, draft=reply_options[0]["draft"] if reply_options else None)
            else:
                reply_options = self._generate_review_reply_options(
                    guest_name, rating, msg.body, llm_steps=llm_steps,
                )
                action.update(reply_options=reply_options, draft=reply_options[0]["draft"] if reply_options else None)
        else:
            effective_rating = rating or 5
            sys_p, usr_p = _review_response_prompts(guest_name, effective_rating, msg.body)
            draft = self._try_generate(
                sys_p, usr_p,
                fallback="Thank you for your wonderful review!",
                module="mail_agent.property_review_response_generation",
                llm_steps=llm_steps,
            )
            action.update(
                action="response_draft_ready",
                draft=draft,
                requires_owner=True,
                reason="Good review; suggested response draft generated",
            )

        step = StepLog(
            module="mail_agent.property_review_policy",
            prompt={
                "email_id": msg.id,
                "rating": rating,
                "threshold": cfg.bad_review_threshold,
            },
            response={
                "is_bad_review": is_bad,
                "action": action["action"],
                "requires_owner": action.get("requires_owner", True),
                "has_draft": action.get("draft") is not None,
                "reply_options_count": len(action.get("reply_options", [])),
            },
        )
        return action, llm_steps + [step]

    def _generate_review_reply_options(
        self,
        guest_name: str,
        rating: int,
        review_text: str,
        *,
        llm_steps: list[StepLog] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate 2-3 preset reply styles for a bad review."""
        options: list[dict[str, Any]] = []
        styles = [
            ("apologetic", "Apologize sincerely and commit to improvement. Be empathetic and brief."),
            ("neutral", "Acknowledge the feedback in a neutral, professional tone. Thank them and note you take feedback seriously."),
            ("brief_thanks", "Very short: thank them for the feedback and that you value their input."),
        ]
        for style_key, instruction in styles:
            sys_p = (
                "You are an Airbnb host responding to a guest review. "
                "Be professional and gracious. Keep under 60 words."
            )
            usr_p = (
                f"Guest: {guest_name}, Rating: {rating}/5. Review: {review_text[:500]}...\n\n"
                f"Reply style: {instruction}"
            )
            draft = self._try_generate(
                sys_p, usr_p,
                fallback="Thank you for your feedback.",
                module="mail_agent.review_reply_option_generation",
                llm_steps=llm_steps,
            )
            options.append({"style": style_key, "draft": draft})
        return options

    # -- LLM helper ----------------------------------------------------------------

    def _try_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback: str,
        *,
        module: str = "",
        llm_steps: list[StepLog] | None = None,
    ) -> str:
        if not self._chat.is_available:
            return fallback
        try:
            result = self._chat.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            text = result or fallback
            if module and llm_steps is not None:
                llm_steps.append(StepLog(
                    module=module,
                    prompt={"system_prompt": system_prompt, "user_prompt": user_prompt},
                    response={"text": text},
                ))
            return text
        except Exception as exc:
            if module and llm_steps is not None:
                llm_steps.append(StepLog(
                    module=module,
                    prompt={"system_prompt": system_prompt, "user_prompt": user_prompt},
                    response={
                        "error": f"{type(exc).__name__}: {exc}",
                        "fallback_used": True,
                        "text": fallback,
                    },
                ))
            return fallback

    # -- Answer builder ------------------------------------------------------------

    def _build_answer_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        actions: list[dict[str, Any]] = state.get("mail_actions") or []
        cfg = self._config

        if not actions:
            return {
                "answer": NO_MAIL_RESPONSE,
                "steps": [
                    StepLog(
                        module="mail_agent.answer_generation",
                        prompt={"action_count": 0},
                        response={"text": NO_MAIL_RESPONSE},
                    )
                ],
            }

        lines: list[str] = [f"Processed {len(actions)} Airbnb email(s):\n"]
        for idx, act in enumerate(actions, start=1):
            category = act.get("category", "unknown")
            email_action = act.get("action", "unknown")
            subject = act.get("subject", "")
            guest = act.get("guest_name") or "Unknown"

            line = f"{idx}. [{category}] {subject}"
            if guest != "Unknown":
                line += f" (Guest: {guest})"
            line += f"\n   Action: {email_action}"
            if act.get("importance"):
                line += f" | Importance: {act['importance']}"
            if act.get("rating") is not None:
                line += f" | Rating: {act['rating']}/5"
            if act.get("requires_owner"):
                line += " | Requires owner action"
            if act.get("draft"):
                preview = act["draft"][:100] + ("..." if len(act["draft"]) > 100 else "")
                line += f"\n   Draft: {preview}"
            if act.get("reason"):
                line += f"\n   Reason: {act['reason']}"

            lines.append(line)

        answer = "\n\n".join(lines)
        answer = _cap_words(answer, cfg.max_answer_words)

        return {
            "answer": answer,
            "mail_actions": actions,
            "steps": [
                StepLog(
                    module="mail_agent.answer_generation",
                    prompt={"action_count": len(actions)},
                    response={
                        "text": answer,
                        "actions_summary": [
                            {
                                "email_id": a.get("email_id"),
                                "category": a.get("category"),
                                "action": a.get("action"),
                                "requires_owner": a.get("requires_owner", False),
                            }
                            for a in actions
                        ],
                    },
                )
            ],
        }


# ---------------------------------------------------------------------------
# MailAgent -- thin wrapper delegating to MailPipeline
# ---------------------------------------------------------------------------


class MailAgent(Agent):
    """Agent for processing Airbnb-related owner emails.

    Thin wrapper that delegates execution to a LangChain-first MailPipeline.
    """

    name = "mail_agent"

    def __init__(
        self,
        *,
        gmail_service: GmailService,
        chat_service: ChatService,
        config: MailAgentConfig | None = None,
    ) -> None:
        self.gmail_service = gmail_service
        self.chat_service = chat_service
        self.config = config or MailAgentConfig()
        self._pipeline = MailPipeline(
            gmail_service=gmail_service,
            chat_service=chat_service,
            config=self.config,
        )

    def run(self, prompt: str, context: dict[str, object] | None = None) -> AgentResult:
        """Process inbox and return mail analysis with action recommendations."""
        result = self._pipeline.invoke({
            "prompt": prompt,
            "context": context or {},
            "steps": [],
        })
        return AgentResult(
            response=result.get("answer", NO_MAIL_RESPONSE),
            steps=result.get("steps", []),
        )

    def run_with_action(
        self,
        prompt: str,
        owner_action: dict[str, Any],
        context: dict[str, object] | None = None,
    ) -> AgentResult:
        """Process inbox with an owner action (rating, approval, etc.)."""
        result = self._pipeline.invoke({
            "prompt": prompt,
            "context": context or {},
            "owner_action": owner_action,
            "steps": [],
        })
        return AgentResult(
            response=result.get("answer", NO_MAIL_RESPONSE),
            steps=result.get("steps", []),
            mail_actions=result.get("mail_actions"),
        )

    def run_on_messages(
        self,
        messages: list[EmailMessage],
        prompt: str = "Process new mail.",
        context: dict[str, object] | None = None,
    ) -> AgentResult:
        """Run pipeline on a pre-fetched list of messages (e.g. from push). No fetch step."""
        state: dict[str, Any] = {
            "prompt": prompt,
            "context": context or {},
            "steps": [],
        }
        result = self._pipeline.invoke_with_messages(state, messages)
        return AgentResult(
            response=result.get("answer", NO_MAIL_RESPONSE),
            steps=result.get("steps", []),
            mail_actions=result.get("mail_actions"),
        )

    def get_inbox_summary(self) -> list[dict[str, Any]]:
        """Return classified inbox items without applying full policies."""
        if not self.gmail_service.is_available:
            return []
        messages = self.gmail_service.list_unread_messages(
            max_results=self.config.max_inbox_fetch,
        )
        classified: list[dict[str, Any]] = []
        for msg in messages:
            cls = _classify_email(msg, self.gmail_service)
            classified.append({
                "email_id": cls.message.id,
                "subject": cls.message.subject,
                "sender": cls.message.sender,
                "date": cls.message.date,
                "category": cls.category,
                "confidence": cls.confidence,
                "guest_name": cls.extracted_guest_name,
                "rating": cls.extracted_rating,
                "snippet": cls.message.snippet,
            })
        return classified
