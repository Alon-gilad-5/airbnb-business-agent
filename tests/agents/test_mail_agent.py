"""Tests for the mail agent: classification, policy rules, and HITL flows.

Covers Airbnb-only filtering, email classification accuracy, threshold-based
policy decisions, and owner action handling -- all with dummy services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.agents.mail_agent import (
    CATEGORY_GUEST_MESSAGE,
    CATEGORY_LEAVE_REVIEW,
    CATEGORY_NEW_PROPERTY_REVIEW,
    CATEGORY_NON_AIRBNB,
    CATEGORY_UNSUPPORTED_AIRBNB,
    NO_MAIL_RESPONSE,
    MailAgent,
    MailAgentConfig,
    _classify_email,
    _extract_guest_name,
    _extract_rating,
    _score_importance,
)
from app.services.gmail_service import EmailMessage, GmailService
from app.services.gmail_service import _DirectGmailClient
from app.services.gmail_service import _decode_gmail_body
from app.services.gmail_service import _gmail_header
from app.services.mail_mock_emails import all_demo_mocks, get_mock_for_category

# -- Dummy services -----------------------------------------------------------


@dataclass
class DummyChatService:
    is_available: bool = True
    model: str = "test-model"
    _response: str = "Test draft reply."

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        if not self.is_available:
            raise RuntimeError("Chat unavailable")
        return self._response


def _make_gmail(enabled: bool = True) -> GmailService:
    """GmailService in demo mode: no MCP files, no Gmail API env vars."""
    return GmailService(
        enabled=enabled,
        gauth_path="__nonexistent_test_gauth__.json",
        accounts_path="__nonexistent_test_accounts__.json",
        gmail_client_id=None,
        gmail_client_secret=None,
        gmail_refresh_token=None,
    )


def _make_agent(
    *,
    gmail_enabled: bool = True,
    chat_available: bool = True,
    chat_response: str = "Test draft reply.",
    bad_review_threshold: int = 3,
) -> MailAgent:
    return MailAgent(
        gmail_service=_make_gmail(enabled=gmail_enabled),
        chat_service=DummyChatService(is_available=chat_available, _response=chat_response),
        config=MailAgentConfig(bad_review_threshold=bad_review_threshold),
    )


def _make_email(
    *,
    id: str = "test-001",
    sender: str = "no-reply@airbnb.com",
    subject: str = "Test email",
    body: str = "Test body",
) -> EmailMessage:
    return EmailMessage(
        id=id,
        thread_id="thread-001",
        sender=sender,
        recipient="owner@example.com",
        subject=subject,
        snippet=body[:80],
        body=body,
        date="2026-02-20T10:00:00Z",
    )


# -- Module-level helper tests ------------------------------------------------


MAIL_MODULE_NAMES = {
    "mail_agent.fetch_inbox",
    "mail_agent.airbnb_filter",
    "mail_agent.classify",
    "mail_agent.policy",
    "mail_agent.guest_message_policy",
    "mail_agent.leave_review_policy",
    "mail_agent.property_review_policy",
    "mail_agent.answer_generation",
    # LLM call steps (new — one per actual chat.generate() call)
    "mail_agent.guest_reply_generation",
    "mail_agent.leave_review_generation",
    "mail_agent.property_review_response_generation",
    "mail_agent.review_reply_option_generation",
    "mail_agent.push_fetch",
}


class TestRatingExtraction:
    def test_slash_notation(self) -> None:
        assert _extract_rating("Rating: 3/5 stars") == 3

    def test_star_notation(self) -> None:
        assert _extract_rating("Guest left a 2 star review") == 2

    def test_colon_notation(self) -> None:
        assert _extract_rating("Rating: 4") == 4

    def test_no_rating(self) -> None:
        assert _extract_rating("No rating information here") is None

    def test_out_of_range(self) -> None:
        assert _extract_rating("Rating: 7/5 stars") is None


class TestGuestNameExtraction:
    def test_from_pattern(self) -> None:
        assert _extract_guest_name("New message from guest John") == "John"

    def test_possessive_pattern(self) -> None:
        assert _extract_guest_name("Sarah's stay has ended") == "Sarah"

    def test_left_review_pattern(self) -> None:
        assert _extract_guest_name("Mike left a review") == "Mike"

    def test_no_name(self) -> None:
        assert _extract_guest_name("General notification") is None


class TestClassification:
    def test_non_airbnb_filtered(self) -> None:
        gmail = _make_gmail()
        email = _make_email(sender="promo@newsletter.com", subject="Buy stuff")
        cls = _classify_email(email, gmail)
        assert cls.category == CATEGORY_NON_AIRBNB
        assert cls.confidence == 1.0

    def test_guest_message(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="New message from guest Alex",
            body="Hi, I have a question about parking.",
        )
        cls = _classify_email(email, gmail)
        assert cls.category == CATEGORY_GUEST_MESSAGE
        assert cls.extracted_guest_name == "Alex"

    def test_leave_review_request(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="Leave a review for your guest Sarah",
            body="Sarah's stay has ended. Share your experience hosting them.",
        )
        cls = _classify_email(email, gmail)
        assert cls.category == CATEGORY_LEAVE_REVIEW

    def test_new_property_review_good(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="New review from guest Lisa: 5 stars",
            body="Lisa left a review of your property. Rating: 5/5 stars. Amazing!",
        )
        cls = _classify_email(email, gmail)
        assert cls.category == CATEGORY_NEW_PROPERTY_REVIEW
        assert cls.extracted_rating == 5
        assert cls.extracted_guest_name == "Lisa"

    def test_new_property_review_bad(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="New review from guest Mike: 2 stars",
            body="Mike left a review. Rating: 2/5 stars. Not clean.",
        )
        cls = _classify_email(email, gmail)
        assert cls.category == CATEGORY_NEW_PROPERTY_REVIEW
        assert cls.extracted_rating == 2

    def test_unsupported_airbnb_email(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="Your monthly earnings summary",
            body="Here is your payment report for February.",
        )
        cls = _classify_email(email, gmail)
        assert cls.category == CATEGORY_UNSUPPORTED_AIRBNB


class TestImportanceScoring:
    def test_high_importance_urgent(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="Urgent message from guest",
            body="I'm locked out and can't get in!",
        )
        cls = _classify_email(email, gmail)
        assert _score_importance(cls) == "high"

    def test_low_importance_thanks(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="Message from guest Bob",
            body="Just wanted to say thanks for the wonderful stay!",
        )
        cls = _classify_email(email, gmail)
        assert _score_importance(cls) == "low"


# -- Agent-level integration tests ---------------------------------------------


class TestMailAgentDisabled:
    def test_disabled_returns_config_message(self) -> None:
        agent = _make_agent(gmail_enabled=False)
        result = agent.run("check my inbox")
        assert "not configured" in result.response.lower() or "not configured" in result.response
        assert len(result.steps) >= 1
        assert result.steps[0].module == "mail_agent.fetch_inbox"

    def test_inbox_summary_empty_when_disabled(self) -> None:
        agent = _make_agent(gmail_enabled=False)
        items = agent.get_inbox_summary()
        assert items == []


class TestMailAgentDemoMode:
    def test_processes_demo_inbox(self) -> None:
        agent = _make_agent()
        result = agent.run("check my inbox")
        assert result.response is not None
        assert len(result.response) > 0
        assert len(result.steps) >= 1

        module_names = {s.module for s in result.steps}
        assert "mail_agent.fetch_inbox" in module_names
        assert "mail_agent.airbnb_filter" in module_names
        assert "mail_agent.classify" in module_names

    def test_step_shapes_are_valid(self) -> None:
        agent = _make_agent()
        result = agent.run("check my inbox")
        for step in result.steps:
            assert step.module in MAIL_MODULE_NAMES, f"Unexpected module: {step.module}"
            assert isinstance(step.prompt, dict)
            assert isinstance(step.response, dict)

    def test_non_airbnb_filtered_out(self) -> None:
        agent = _make_agent()
        items = agent.get_inbox_summary()
        categories = [item["category"] for item in items]
        assert CATEGORY_NON_AIRBNB in categories
        airbnb_categories = [c for c in categories if c != CATEGORY_NON_AIRBNB]
        assert len(airbnb_categories) > 0

    def test_inbox_summary_returns_classified_items(self) -> None:
        agent = _make_agent()
        items = agent.get_inbox_summary()
        assert len(items) > 0
        for item in items:
            assert "email_id" in item
            assert "category" in item
            assert "confidence" in item
            assert item["confidence"] > 0


class TestMailAgentPolicies:
    def test_leave_review_awaits_rating_without_action(self) -> None:
        agent = _make_agent()
        result = agent.run("check inbox")
        found_leave_review = False
        for step in result.steps:
            if step.module == "mail_agent.leave_review_policy":
                found_leave_review = True
                assert step.response.get("action") == "awaiting_owner_rating"
        assert found_leave_review

    def test_leave_review_with_positive_rating(self) -> None:
        agent = _make_agent()
        owner_action = {"email_id": "demo-002", "rating": 4}
        result = agent.run_with_action("check inbox", owner_action=owner_action)
        found_review_draft = False
        for step in result.steps:
            if (
                step.module == "mail_agent.leave_review_policy"
                and step.response.get("action") == "review_draft_ready"
            ):
                found_review_draft = True
                assert step.response.get("rating_tier") == "positive"
        assert found_review_draft

    def test_leave_review_with_negative_rating(self) -> None:
        agent = _make_agent()
        owner_action = {"email_id": "demo-002", "rating": 2, "issues": ["Noise complaints"]}
        result = agent.run_with_action("check inbox", owner_action=owner_action)
        found_review_draft = False
        for step in result.steps:
            if (
                step.module == "mail_agent.leave_review_policy"
                and step.response.get("action") == "review_draft_ready"
            ):
                found_review_draft = True
                assert step.response.get("rating_tier") == "negative"
        assert found_review_draft

    def test_bad_property_review_requires_owner_consult(self) -> None:
        agent = _make_agent(bad_review_threshold=3)
        result = agent.run("check inbox")
        found_bad_review = False
        for step in result.steps:
            if (
                step.module == "mail_agent.property_review_policy"
                and step.response.get("is_bad_review") is True
            ):
                found_bad_review = True
                assert step.response.get("requires_owner") is True
        assert found_bad_review

    def test_good_property_review_gets_draft(self) -> None:
        agent = _make_agent()
        result = agent.run("check inbox")
        found_good_review = False
        for step in result.steps:
            if (
                step.module == "mail_agent.property_review_policy"
                and step.response.get("is_bad_review") is False
            ):
                found_good_review = True
                assert step.response.get("has_draft") is True
        assert found_good_review

    def test_chat_unavailable_uses_fallbacks(self) -> None:
        agent = _make_agent(chat_available=False)
        result = agent.run("check inbox")
        assert result.response is not None
        assert len(result.response) > 0

    def test_good_review_emits_llm_step(self) -> None:
        """property_review_response_generation step appears for good reviews."""
        agent = _make_agent()
        result = agent.run("check inbox")
        llm_modules = [s.module for s in result.steps]
        assert "mail_agent.property_review_response_generation" in llm_modules
        for step in result.steps:
            if step.module == "mail_agent.property_review_response_generation":
                assert "system_prompt" in step.prompt
                assert "user_prompt" in step.prompt
                assert "text" in step.response

    def test_bad_review_emits_review_reply_option_steps(self) -> None:
        """review_reply_option_generation steps appear (one per style) for bad reviews."""
        agent = _make_agent(bad_review_threshold=3)
        result = agent.run("check inbox")
        option_steps = [s for s in result.steps if s.module == "mail_agent.review_reply_option_generation"]
        # 3 styles generated for each bad review — demo has one bad review
        assert len(option_steps) >= 3

    def test_positive_leave_review_emits_llm_step(self) -> None:
        """leave_review_generation step appears when owner provides a positive rating."""
        agent = _make_agent()
        owner_action = {"email_id": "demo-002", "rating": 4}
        result = agent.run_with_action("check inbox", owner_action=owner_action)
        llm_modules = [s.module for s in result.steps]
        assert "mail_agent.leave_review_generation" in llm_modules

    def test_chat_unavailable_no_llm_steps(self) -> None:
        """No LLM step modules appear when chat service is unavailable."""
        agent = _make_agent(chat_available=False)
        result = agent.run("check inbox")
        llm_only_modules = {
            "mail_agent.guest_reply_generation",
            "mail_agent.leave_review_generation",
            "mail_agent.property_review_response_generation",
            "mail_agent.review_reply_option_generation",
        }
        for step in result.steps:
            assert step.module not in llm_only_modules, (
                f"LLM step {step.module} appeared despite chat being unavailable"
            )


# -- run_on_messages (new-email / push path) with mock emails -----------------


def _mock_raw_to_message(raw: dict[str, Any]) -> EmailMessage:
    """Convert a mock email dict (from mail_mock_emails) to EmailMessage."""
    return GmailService._raw_to_message(raw)


class TestMailAgentRunOnMessages:
    """Agent reaction when receiving new email (run_on_messages with mock emails)."""

    def test_run_on_messages_uses_push_fetch_step(self) -> None:
        agent = _make_agent()
        raw = get_mock_for_category("guest_message")
        messages = [_mock_raw_to_message(raw)]
        result = agent.run_on_messages(messages)
        module_names = {s.module for s in result.steps}
        assert "mail_agent.push_fetch" in module_names
        assert "mail_agent.fetch_inbox" not in module_names

    def test_run_on_messages_guest_message_high(self) -> None:
        agent = _make_agent()
        raw = get_mock_for_category("guest_message", "high")
        messages = [_mock_raw_to_message(raw)]
        result = agent.run_on_messages(messages)
        assert result.mail_actions is not None
        assert len(result.mail_actions) >= 1
        action = next(a for a in result.mail_actions if a.get("email_id") == raw["id"])
        assert action["category"] == CATEGORY_GUEST_MESSAGE
        assert action.get("action") == "owner_heads_up"
        assert action.get("requires_owner") is True

    def test_run_on_messages_guest_message_low(self) -> None:
        agent = _make_agent()
        raw = get_mock_for_category("guest_message", "low")
        messages = [_mock_raw_to_message(raw)]
        result = agent.run_on_messages(messages)
        assert result.mail_actions is not None
        action = next(a for a in result.mail_actions if a.get("email_id") == raw["id"])
        assert action["category"] == CATEGORY_GUEST_MESSAGE
        assert action.get("action") in ("draft_reply", "owner_consult")
        assert action.get("requires_owner") is True
        if action.get("action") == "draft_reply":
            assert action.get("draft")

    def test_run_on_messages_leave_review_request(self) -> None:
        agent = _make_agent()
        raw = get_mock_for_category("leave_review_request")
        messages = [_mock_raw_to_message(raw)]
        result = agent.run_on_messages(messages)
        assert result.mail_actions is not None
        action = next(a for a in result.mail_actions if a.get("email_id") == raw["id"])
        assert action["category"] == CATEGORY_LEAVE_REVIEW
        assert action.get("action") == "awaiting_owner_rating"
        assert action.get("requires_owner") is True

    def test_run_on_messages_new_property_review_bad(self) -> None:
        agent = _make_agent(bad_review_threshold=3)
        raw = get_mock_for_category("new_property_review", "bad")
        messages = [_mock_raw_to_message(raw)]
        result = agent.run_on_messages(messages)
        assert result.mail_actions is not None
        action = next(a for a in result.mail_actions if a.get("email_id") == raw["id"])
        assert action["category"] == CATEGORY_NEW_PROPERTY_REVIEW
        assert action.get("requires_owner") is True
        assert "reply_options" in action
        assert isinstance(action["reply_options"], list)
        assert len(action["reply_options"]) >= 1
        assert action.get("draft")

    def test_run_on_messages_new_property_review_good(self) -> None:
        agent = _make_agent()
        raw = get_mock_for_category("new_property_review", "good")
        messages = [_mock_raw_to_message(raw)]
        result = agent.run_on_messages(messages)
        assert result.mail_actions is not None
        action = next(a for a in result.mail_actions if a.get("email_id") == raw["id"])
        assert action["category"] == CATEGORY_NEW_PROPERTY_REVIEW
        assert action.get("draft")
        assert action.get("action") != "owner_chose_dont_reply"

    def test_run_on_messages_unsupported_airbnb(self) -> None:
        agent = _make_agent()
        raw = get_mock_for_category("unsupported_airbnb")
        messages = [_mock_raw_to_message(raw)]
        result = agent.run_on_messages(messages)
        assert result.mail_actions is not None
        action = next(a for a in result.mail_actions if a.get("email_id") == raw["id"])
        assert action["category"] == CATEGORY_UNSUPPORTED_AIRBNB
        assert action.get("action") == "no_action"
        assert action.get("requires_owner") is False

    def test_run_on_messages_non_airbnb_excluded(self) -> None:
        agent = _make_agent()
        raw = get_mock_for_category("non_airbnb")
        messages = [_mock_raw_to_message(raw)]
        result = agent.run_on_messages(messages)
        # When only non-Airbnb is passed, filter excludes it so mail_actions may be None or empty
        if result.mail_actions is None:
            assert result.response
            return
        assert not any(a.get("email_id") == raw["id"] for a in result.mail_actions)

    def test_run_on_messages_all_demo_mocks(self) -> None:
        agent = _make_agent()
        raws = all_demo_mocks()
        messages = [_mock_raw_to_message(r) for r in raws]
        result = agent.run_on_messages(messages)
        assert result.response
        assert result.mail_actions is not None
        categories = {a.get("category") for a in result.mail_actions}
        assert CATEGORY_GUEST_MESSAGE in categories
        assert CATEGORY_LEAVE_REVIEW in categories
        assert CATEGORY_NEW_PROPERTY_REVIEW in categories
        assert CATEGORY_NON_AIRBNB not in categories
        assert len(result.mail_actions) >= 4


# -- Router integration --------------------------------------------------------


class TestDirectGmailClientParsing:
    """Unit tests for Gmail API response parsing (no live API calls)."""

    def test_gmail_header_extracts_from_payload(self) -> None:
        payload = {
            "headers": [
                {"name": "From", "value": "no-reply@airbnb.com"},
                {"name": "To", "value": "owner@example.com"},
                {"name": "Subject", "value": "New message"},
            ]
        }
        assert _gmail_header(payload, "From") == "no-reply@airbnb.com"
        assert _gmail_header(payload, "To") == "owner@example.com"
        assert _gmail_header(payload, "Subject") == "New message"
        assert _gmail_header(payload, "Date") == ""

    def test_decode_gmail_body_simple(self) -> None:
        import base64

        text = "Hello, this is the body."
        payload = {"body": {"data": base64.urlsafe_b64encode(text.encode()).decode()}}
        assert _decode_gmail_body(payload) == text

    def test_decode_gmail_body_empty(self) -> None:
        assert _decode_gmail_body({}) == ""
        assert _decode_gmail_body({"body": {}}) == ""

    def test_direct_client_message_to_email(self) -> None:
        import base64

        body_b64 = base64.urlsafe_b64encode(b"Email body here.").decode()
        msg = {
            "id": "msg-123",
            "threadId": "thread-456",
            "snippet": "Short snippet...",
            "internalDate": "1609459200000",
            "labelIds": ["INBOX", "UNREAD"],
            "payload": {
                "headers": [
                    {"name": "From", "value": "sender@airbnb.com"},
                    {"name": "To", "value": "me@example.com"},
                    {"name": "Subject", "value": "Test Subject"},
                ],
                "body": {"data": body_b64},
            },
        }
        client = _DirectGmailClient(
            client_id="fake-id",
            client_secret="fake-secret",
            refresh_token="fake-token",
        )
        email = client._message_to_email(msg)
        assert email.id == "msg-123"
        assert email.thread_id == "thread-456"
        assert email.sender == "sender@airbnb.com"
        assert email.recipient == "me@example.com"
        assert email.subject == "Test Subject"
        assert email.body == "Email body here."
        assert email.snippet == "Short snippet..."
        assert "INBOX" in email.labels


class TestRouterMailKeywords:
    def test_inbox_routes_to_mail(self) -> None:
        from app.agents.router_agent import RouterAgent

        router = RouterAgent()
        decision, step = router.route("check my email inbox")
        assert decision.agent_name == "mail_agent"

    def test_mail_keyword(self) -> None:
        from app.agents.router_agent import RouterAgent

        router = RouterAgent()
        decision, _ = router.route("any new mail?")
        assert decision.agent_name == "mail_agent"

    def test_gmail_keyword(self) -> None:
        from app.agents.router_agent import RouterAgent

        router = RouterAgent()
        decision, _ = router.route("check gmail for messages")
        assert decision.agent_name == "mail_agent"
