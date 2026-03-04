"""FastAPI entrypoint implementing required course endpoints and minimal GUI."""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import Body, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.agents.base import Agent
from app.agents.analyst_agent import AnalystAgent
from app.agents.mail_agent import MailAgent, MailAgentConfig
from app.agents.market_watch_agent import MarketWatchAgent, MarketWatchAgentConfig
from app.agents.pricing_agent import PricingAgent, PricingAgentConfig, outcome_to_response
from app.agents.reviews_agent import ReviewsAgent, ReviewsAgentConfig
from app.architecture import ensure_architecture_svg
from app.config import ActiveOwnerContext, load_settings
from app.schemas import (
    ActiveOwnerContextResponse,
    AnalysisExplainSelectionRequest,
    AnalysisExplainSelectionResponse,
    AnalysisRequest,
    AnalysisResponse,
    AgentInfoResponse,
    AgentPromptExample,
    AgentPromptTemplate,
    AnalysisCategoricalItem,
    AnalysisNumericItem,
    EvidenceFlagRequest,
    ExecuteRequest,
    ExecuteResponse,
    MailActionRequest,
    MailActionResponse,
    MailInboxItemResponse,
    MailInboxResponse,
    MailSettingsResponse,
    MailSettingsUpdateRequest,
    MarketAlertResponse,
    MarketWatchAlertsResponse,
    MarketWatchRunRequest,
    MarketWatchRunResponse,
    NotificationItem,
    NotificationsResponse,
    PricingRequest,
    PricingResponse,
    PropertyProfileResponse,
    PropertyProfilesResponse,
    StepLog,
    ThresholdLabelCandidate,
    ThresholdLabelCase,
    ThresholdLabelingDataResponse,
    ThresholdLabelSaveRequest,
    ThresholdLabelSaveResponse,
    TeamInfoResponse,
    TeamStudentResponse,
)
from app.services.market_alert_store import MarketAlertRecord, create_market_alert_store
from app.services.listing_store import create_listing_store
from app.services.neighbor_store import create_neighbor_store
from app.services.market_data_providers import MarketDataProviders
from app.services.market_watch_scheduler import MarketWatchScheduler
from app.services.chat_service import ChatService
from app.services.embeddings import EmbeddingService
from app.services.pinecone_retriever import PineconeRetriever
from app.services.gmail_service import GmailService
from app.services.mail_push_state import (
    get_mail_preferences,
    get_owner_choice,
    get_push_state,
    set_mail_preferences,
    set_owner_choice,
    set_push_state,
)
from app.services.evidence_flag_store import EvidenceFlagStore
from app.services.notification_store import NotificationStore
from app.services.region_utils import canonicalize_region
from app.services.web_review_ingest import WebReviewIngestService
from app.services.web_review_scraper import PlaywrightReviewScraper


load_dotenv()
settings = load_settings()
app = FastAPI(title="Airbnb Business Agent", version="0.1.0")
logger = logging.getLogger(__name__)

# Paths are relative to this file, making local + Render deployment consistent.
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
ARCH_PATH = STATIC_DIR / "model_architecture.svg"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
THRESHOLD_LABEL_POOL_PATH = BASE_DIR.parent / "outputs" / "reviews_threshold_label_pool.jsonl"
THRESHOLD_GOLD_PATH = BASE_DIR.parent / "outputs" / "reviews_threshold_gold.csv"
VALID_CHAT_PROVIDERS = {"llmod", "openrouter"}
PROPERTY_REVIEW_VOLUME_LABELS = {
    "42409434": "Many reviews",
    "10046908": "few reviews",
}


def _build_openrouter_headers() -> dict[str, str] | None:
    """Build optional OpenRouter headers for attribution."""

    headers: dict[str, str] = {}
    if settings.openrouter_http_referer:
        headers["HTTP-Referer"] = settings.openrouter_http_referer
    if settings.openrouter_app_title:
        headers["X-Title"] = settings.openrouter_app_title
    return headers or None


def _owner_source_urls(owner: ActiveOwnerContext) -> dict[str, str] | None:
    """Collect optional source URLs from owner/property context."""

    source_urls: dict[str, str] = {}
    if owner.google_maps_url:
        source_urls["google_maps"] = owner.google_maps_url
    if owner.tripadvisor_url:
        source_urls["tripadvisor"] = owner.tripadvisor_url
    return source_urls or None


def _build_property_profiles() -> dict[str, ActiveOwnerContext]:
    """Build selectable property profiles for UI and market-watch override support."""

    profiles: dict[str, ActiveOwnerContext] = {"primary": settings.active_owner}
    if settings.secondary_owner and settings.secondary_owner.property_id:
        profiles["secondary"] = settings.secondary_owner
    return profiles


def _owner_to_profile_response(
    *,
    profile_id: Literal["primary", "secondary"],
    owner: ActiveOwnerContext,
) -> PropertyProfileResponse:
    """Convert owner context into API profile response shape."""

    return PropertyProfileResponse(
        profile_id=profile_id,
        owner_id=owner.owner_id,
        owner_name=owner.owner_name,
        property_id=owner.property_id,
        property_name=owner.property_name,
        city=owner.city,
        region=canonicalize_region(owner.region),
        latitude=owner.latitude,
        longitude=owner.longitude,
        source_urls=_owner_source_urls(owner),
        max_scrape_reviews=owner.default_max_scrape_reviews,
        review_volume_label=PROPERTY_REVIEW_VOLUME_LABELS.get((owner.property_id or "").strip()),
    )


property_profiles = _build_property_profiles()


# Shared services are initialized once and reused by all agents.
embedding_service = EmbeddingService(
    api_key=settings.llmod_api_key,
    azure_endpoint=settings.base_url,
    model=settings.embedding_model,
    deployment=settings.embedding_deployment,
)
try:
    retriever = PineconeRetriever(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
    )
except Exception as exc:
    logger.warning("PineconeRetriever init failed, reviews will be unavailable: %s", exc)
    retriever = PineconeRetriever(api_key=None, index_name="", namespace="")
chat_services_by_provider: dict[str, ChatService] = {
    "llmod": ChatService(
        provider_name="llmod",
        api_key=settings.llmod_api_key,
        base_url=settings.base_url,
        model=settings.chat_model,
        max_output_tokens=settings.chat_max_output_tokens,
    ),
    "openrouter": ChatService(
        provider_name="openrouter",
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.openrouter_chat_model,
        default_headers=_build_openrouter_headers(),
        max_output_tokens=settings.chat_max_output_tokens,
    ),
}
default_chat_provider = (
    settings.llm_chat_provider
    if settings.llm_chat_provider in VALID_CHAT_PROVIDERS
    else "llmod"
)
chat_service = chat_services_by_provider[default_chat_provider]
web_scraper = PlaywrightReviewScraper(
    enabled=settings.scraping_enabled,
    allowlist=settings.scraping_allowlist,
    default_max_reviews=settings.scraping_default_max_reviews,
    timeout_seconds=settings.scraping_timeout_seconds,
    require_source_selectors=settings.scraping_require_source_selectors,
    min_review_chars=settings.scraping_min_review_chars,
    min_token_count=settings.scraping_min_token_count,
    reject_private_use_ratio=settings.scraping_reject_private_use_ratio,
    navigation_click_timeout_ms=settings.scraping_navigation_click_timeout_ms,
    gmaps_locale=settings.scraping_gmaps_locale,
    gmaps_viewport_width=settings.scraping_gmaps_viewport_width,
    gmaps_viewport_height=settings.scraping_gmaps_viewport_height,
    gmaps_user_agents=settings.scraping_gmaps_user_agents,
    gmaps_scroll_passes=settings.scraping_gmaps_scroll_passes,
    gmaps_scroll_pause_ms=settings.scraping_gmaps_scroll_pause_ms,
    gmaps_nav_timeout_ms=settings.scraping_gmaps_nav_timeout_ms,
)
try:
    web_ingest_service = WebReviewIngestService(
        enabled=settings.scraping_quarantine_upsert_enabled,
        pinecone_api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        namespace=settings.scraping_quarantine_namespace,
        embedding_service=embedding_service,
    )
except Exception as exc:
    logger.warning("WebReviewIngestService init failed: %s", exc)
    web_ingest_service = WebReviewIngestService(
        enabled=False, pinecone_api_key=None, index_name="", namespace="", embedding_service=embedding_service,
    )
market_data_providers = MarketDataProviders(
    ticketmaster_api_key=settings.ticketmaster_api_key,
    timeout_seconds=20,
)
try:
    market_alert_store = create_market_alert_store(
        database_url=settings.database_url,
        sqlite_path=settings.market_watch_alerts_db_path,
    )
except Exception as exc:
    # Keep the API bootable even if DB wiring is temporarily broken in deployment env.
    logger.warning(
        "market_alert_store init failed, falling back to /tmp sqlite: %s: %s",
        type(exc).__name__,
        exc,
    )
    market_alert_store = create_market_alert_store(
        database_url=None,
        sqlite_path="/tmp/market_watch_alerts.db",
    )
market_watch_agent = MarketWatchAgent(
    providers=market_data_providers,
    alert_store=market_alert_store,
    config=MarketWatchAgentConfig(
        lookahead_days=settings.market_watch_lookahead_days,
        event_radius_km=settings.market_watch_event_radius_km,
        max_alerts_per_run=settings.market_watch_max_alerts_per_run,
        storm_wind_kph_threshold=settings.market_watch_storm_wind_kph_threshold,
        heavy_rain_mm_threshold=settings.market_watch_heavy_rain_mm_threshold,
        snow_cm_threshold=settings.market_watch_snow_cm_threshold,
    ),
)

neighbor_store = create_neighbor_store(database_url=settings.database_url)
listing_store = create_listing_store(database_url=settings.database_url)

gmail_service = GmailService(
    enabled=settings.mail_enabled,
    gauth_path=settings.gmail_gauth_path,
    accounts_path=settings.gmail_accounts_path,
    credentials_dir=settings.gmail_credentials_dir,
    airbnb_sender_domains=settings.mail_airbnb_sender_domains,
    gmail_client_id=settings.gmail_client_id,
    gmail_client_secret=settings.gmail_client_secret,
    gmail_refresh_token=settings.gmail_refresh_token,
)


def _build_reviews_agent(provider_chat_service: ChatService) -> ReviewsAgent:
    """Construct a reviews agent bound to one configured chat provider."""

    return ReviewsAgent(
        embedding_service=embedding_service,
        retriever=retriever,
        chat_service=provider_chat_service,
        web_scraper=web_scraper,
        web_ingest_service=web_ingest_service,
        config=ReviewsAgentConfig(
            top_k=settings.pinecone_top_k,
            relevance_score_threshold=settings.reviews_relevance_score_threshold,
            min_lexical_relevance_for_upsert=settings.scraping_min_lexical_relevance_for_upsert,
        ),
        neighbor_store=neighbor_store,
    )


def _build_mail_agent(provider_chat_service: ChatService) -> MailAgent:
    """Construct a mail agent bound to one configured chat provider."""

    return MailAgent(
        gmail_service=gmail_service,
        chat_service=provider_chat_service,
        config=MailAgentConfig(
            bad_review_threshold=settings.mail_bad_review_threshold,
            max_inbox_fetch=settings.mail_max_inbox_fetch,
            auto_send_enabled=settings.mail_auto_send_enabled,
        ),
    )


def _build_analyst_agent(provider_chat_service: ChatService) -> AnalystAgent:
    """Construct an analyst agent bound to one configured chat provider."""

    return AnalystAgent(
        listing_store=listing_store,
        neighbor_store=neighbor_store,
        chat_service=provider_chat_service,
    )


def _build_pricing_agent(provider_chat_service: ChatService) -> PricingAgent:
    """Construct a pricing agent bound to one configured chat provider."""

    return PricingAgent(
        listing_store=listing_store,
        neighbor_store=neighbor_store,
        market_data_providers=market_data_providers,
        chat_service=provider_chat_service,
        config=PricingAgentConfig(
            default_horizon_days=settings.pricing_default_horizon_days,
            max_horizon_days=settings.pricing_max_horizon_days,
            low_conf_cap_pct=settings.pricing_raise_cap_low_conf_pct,
            medium_conf_cap_pct=settings.pricing_raise_cap_med_conf_pct,
            high_conf_cap_pct=settings.pricing_raise_cap_high_conf_pct,
            event_radius_km=settings.pricing_event_radius_km,
            strong_event_threshold=settings.pricing_strong_event_threshold,
            weather_soft_threshold_days=settings.pricing_weather_soft_threshold_days,
            storm_wind_kph_threshold=settings.market_watch_storm_wind_kph_threshold,
            heavy_rain_mm_threshold=settings.market_watch_heavy_rain_mm_threshold,
            snow_cm_threshold=settings.market_watch_snow_cm_threshold,
            review_volume_adjustment_pct=settings.pricing_review_volume_adjustment_pct,
        ),
    )


reviews_agents_by_provider: dict[str, ReviewsAgent] = {
    provider_name: _build_reviews_agent(provider_chat_service)
    for provider_name, provider_chat_service in chat_services_by_provider.items()
}
mail_agents_by_provider: dict[str, MailAgent] = {
    provider_name: _build_mail_agent(provider_chat_service)
    for provider_name, provider_chat_service in chat_services_by_provider.items()
}
analysis_agents_by_provider: dict[str, AnalystAgent] = {
    provider_name: _build_analyst_agent(provider_chat_service)
    for provider_name, provider_chat_service in chat_services_by_provider.items()
}
pricing_agents_by_provider: dict[str, PricingAgent] = {
    provider_name: _build_pricing_agent(provider_chat_service)
    for provider_name, provider_chat_service in chat_services_by_provider.items()
}
mail_agent = mail_agents_by_provider[default_chat_provider]
analyst_agent = analysis_agents_by_provider[default_chat_provider]
pricing_agent = pricing_agents_by_provider[default_chat_provider]
notification_store = NotificationStore(database_url=settings.database_url)
evidence_flag_store = EvidenceFlagStore(database_url=settings.database_url)

# Serialize push webhook processing — the Gmail API client (httplib2) is not
# thread-safe, so concurrent threadpool workers would corrupt SSL state.
_push_lock = threading.Lock()
# Coalesce push bursts to one background worker so request threads stay free.
_push_worker_state_lock = threading.Lock()
_push_worker_running = False
_pending_push_history_id: str | None = None

agent_registry: dict[str, Agent] = {
    "reviews_agent": reviews_agents_by_provider[default_chat_provider],
    "market_watch_agent": market_watch_agent,
    "mail_agent": mail_agent,
    "analyst_agent": analyst_agent,
    "pricing_agent": pricing_agent,
}


def _load_threshold_label_pool() -> list[dict[str, object]]:
    """Load JSONL label-pool cases used by threshold calibration UI."""

    if not THRESHOLD_LABEL_POOL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Label pool file not found: {THRESHOLD_LABEL_POOL_PATH}. "
                "Run scripts/export_threshold_labeling_pool.py first."
            ),
        )

    cases: list[dict[str, object]] = []
    with THRESHOLD_LABEL_POOL_PATH.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid JSONL in {THRESHOLD_LABEL_POOL_PATH} at line {line_no}: {exc}",
                ) from exc
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                continue
            cases.append(row)
    return cases


def _load_threshold_gold_rows() -> tuple[list[dict[str, str]], dict[str, dict[str, object]]]:
    """Load existing manual labels from CSV and index by case_id."""

    rows: list[dict[str, str]] = []
    labels_by_case: dict[str, dict[str, object]] = {}
    if not THRESHOLD_GOLD_PATH.exists():
        return rows, labels_by_case

    with THRESHOLD_GOLD_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                continue
            should_answer_raw = str(row.get("should_answer", "")).strip()
            relevant_vector_ids = [
                vector_id.strip()
                for vector_id in str(row.get("relevant_vector_ids", "")).split("|")
                if vector_id.strip()
            ]
            if should_answer_raw == "1":
                should_answer: bool | None = True
            elif should_answer_raw == "0":
                should_answer = False
            else:
                should_answer = None
                should_answer_raw = ""

            clean_row = {
                "case_id": case_id,
                "should_answer": should_answer_raw,
                "relevant_vector_ids": "|".join(relevant_vector_ids),
            }
            rows.append(clean_row)
            labels_by_case[case_id] = {
                "should_answer": should_answer,
                "relevant_vector_ids": relevant_vector_ids,
            }
    return rows, labels_by_case


def _build_threshold_label_cases() -> tuple[list[ThresholdLabelCase], int]:
    """Merge label-pool candidates with current gold labels for UI rendering."""

    pool_cases_raw = _load_threshold_label_pool()
    pool_cases: list[dict[str, object]] = []
    seen_case_keys: set[tuple[str, str, str]] = set()
    for pool_case in pool_cases_raw:
        case_key = (
            str(pool_case.get("property_id", "")).strip(),
            str(pool_case.get("topic", "")).strip(),
            str(pool_case.get("prompt", "")).strip(),
        )
        if case_key in seen_case_keys:
            continue
        seen_case_keys.add(case_key)
        pool_cases.append(pool_case)
    _, labels_by_case = _load_threshold_gold_rows()

    merged_cases: list[ThresholdLabelCase] = []
    labeled_cases = 0
    for pool_case in pool_cases:
        case_id = str(pool_case.get("case_id", "")).strip()
        candidates_raw = pool_case.get("candidates") or []
        if not isinstance(candidates_raw, list):
            candidates_raw = []

        candidates: list[ThresholdLabelCandidate] = []
        candidate_ids: set[str] = set()
        for candidate in candidates_raw:
            if not isinstance(candidate, dict):
                continue
            vector_id = str(candidate.get("vector_id", "")).strip()
            if not vector_id:
                continue
            score_raw = candidate.get("score", 0.0)
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 0.0
            candidate_ids.add(vector_id)
            candidates.append(
                ThresholdLabelCandidate(
                    vector_id=vector_id,
                    score=score,
                    review_text=str(candidate.get("review_text", "")),
                    review_id=(
                        str(candidate.get("review_id"))
                        if candidate.get("review_id") is not None
                        else None
                    ),
                    property_id=(
                        str(candidate.get("property_id"))
                        if candidate.get("property_id") is not None
                        else None
                    ),
                    region=(str(candidate.get("region")) if candidate.get("region") is not None else None),
                    review_date=(
                        str(candidate.get("review_date"))
                        if candidate.get("review_date") is not None
                        else None
                    ),
                )
            )

        label = labels_by_case.get(case_id, {})
        should_answer = label.get("should_answer")
        relevant_vector_ids = [
            vector_id
            for vector_id in label.get("relevant_vector_ids", [])
            if vector_id in candidate_ids
        ]
        labeled = should_answer is not None
        if labeled:
            labeled_cases += 1

        merged_cases.append(
            ThresholdLabelCase(
                case_id=case_id,
                property_id=str(pool_case.get("property_id", "")),
                region=str(pool_case.get("region", "")),
                tier=str(pool_case.get("tier", "")),
                topic=str(pool_case.get("topic", "")),
                prompt=str(pool_case.get("prompt", "")),
                candidate_count=len(candidates),
                candidates=candidates,
                should_answer=should_answer,
                relevant_vector_ids=relevant_vector_ids,
                labeled=labeled,
            )
        )

    return merged_cases, labeled_cases


def _write_threshold_gold_rows(rows: list[dict[str, str]]) -> None:
    """Write gold CSV atomically to avoid partial label-file corruption."""

    THRESHOLD_GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = THRESHOLD_GOLD_PATH.with_suffix(".csv.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "should_answer", "relevant_vector_ids"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": str(row.get("case_id", "")).strip(),
                    "should_answer": str(row.get("should_answer", "")).strip(),
                    "relevant_vector_ids": str(row.get("relevant_vector_ids", "")).strip(),
                }
            )
    temp_path.replace(THRESHOLD_GOLD_PATH)


def _build_effective_context(payload: ExecuteRequest) -> dict[str, object]:
    """Merge request context over active-owner defaults from environment settings."""

    owner = settings.active_owner
    source_urls: dict[str, str] = {}
    owner_source_urls = _owner_source_urls(owner) or {}
    source_urls.update(owner_source_urls)
    if payload.source_urls:
        source_urls.update(payload.source_urls)

    max_scrape_reviews = (
        payload.max_scrape_reviews
        if payload.max_scrape_reviews is not None
        else owner.default_max_scrape_reviews
    )

    return {
        "owner_id": owner.owner_id,
        "owner_name": owner.owner_name,
        "property_id": payload.property_id or owner.property_id,
        "property_name": payload.property_name or owner.property_name,
        "city": payload.city or owner.city,
        "region": canonicalize_region(payload.region) or canonicalize_region(owner.region),
        "latitude": payload.latitude if payload.latitude is not None else owner.latitude,
        "longitude": payload.longitude if payload.longitude is not None else owner.longitude,
        "source_urls": source_urls or None,
        "max_scrape_reviews": max_scrape_reviews,
    }


def _build_active_owner_context() -> ActiveOwnerContextResponse:
    """Expose default owner/property context without requiring an execute payload."""

    owner = settings.active_owner
    return ActiveOwnerContextResponse(
        owner_id=owner.owner_id,
        owner_name=owner.owner_name,
        property_id=owner.property_id,
        property_name=owner.property_name,
        city=owner.city,
        region=canonicalize_region(owner.region),
        latitude=owner.latitude,
        longitude=owner.longitude,
        source_urls=_owner_source_urls(owner),
        max_scrape_reviews=owner.default_max_scrape_reviews,
    )


def _resolve_analysis_agent(
    requested_provider_raw: str | None,
) -> tuple[AnalystAgent | None, str | None]:
    """Resolve analysis agent/provider pair and return an error string when invalid."""

    requested_provider = (requested_provider_raw or "default").strip().lower()
    provider_is_explicit = requested_provider in VALID_CHAT_PROVIDERS
    resolved_provider = (
        default_chat_provider
        if requested_provider in {"", "default"}
        else requested_provider
    )
    if resolved_provider not in VALID_CHAT_PROVIDERS:
        resolved_provider = default_chat_provider

    provider_chat_service = chat_services_by_provider.get(resolved_provider)
    if provider_is_explicit and (provider_chat_service is None or not provider_chat_service.is_available):
        return None, (
            f"Requested llm_provider '{resolved_provider}' is not configured. "
            "Set required provider env vars and retry."
        )

    target_agent = analysis_agents_by_provider.get(resolved_provider)
    if target_agent is None:
        target_agent = analysis_agents_by_provider.get(default_chat_provider)
    if target_agent is None:
        target_agent = analyst_agent
    return target_agent, None


def _resolve_pricing_agent(
    requested_provider_raw: str | None,
) -> tuple[PricingAgent | None, str | None]:
    """Resolve pricing agent/provider pair and return an error string when invalid."""

    requested_provider = (requested_provider_raw or "default").strip().lower()
    provider_is_explicit = requested_provider in VALID_CHAT_PROVIDERS
    resolved_provider = (
        default_chat_provider
        if requested_provider in {"", "default"}
        else requested_provider
    )
    if resolved_provider not in VALID_CHAT_PROVIDERS:
        resolved_provider = default_chat_provider

    provider_chat_service = chat_services_by_provider.get(resolved_provider)
    if provider_is_explicit and (provider_chat_service is None or not provider_chat_service.is_available):
        return None, (
            f"Requested llm_provider '{resolved_provider}' is not configured. "
            "Set required provider env vars and retry."
        )

    target_agent = pricing_agents_by_provider.get(resolved_provider)
    if target_agent is None:
        target_agent = pricing_agents_by_provider.get(default_chat_provider)
    if target_agent is None:
        target_agent = pricing_agent
    return target_agent, None


def _build_property_profiles_response() -> PropertyProfilesResponse:
    """Return profile list consumed by the reviews/market-watch UIs."""

    profiles: list[PropertyProfileResponse] = []
    for profile_id in ("primary", "secondary"):
        owner = property_profiles.get(profile_id)
        if owner is None:
            continue
        profiles.append(
            _owner_to_profile_response(
                profile_id=("secondary" if profile_id == "secondary" else "primary"),
                owner=owner,
            )
        )
    return PropertyProfilesResponse(default_profile_id="primary", profiles=profiles)


def _owner_for_property_id(property_id: str | None) -> ActiveOwnerContext | None:
    """Resolve one configured owner profile by property_id when possible."""

    clean_property_id = property_id.strip() if property_id else ""
    if clean_property_id:
        for owner in property_profiles.values():
            if (owner.property_id or "").strip() == clean_property_id:
                return owner
        return None
    return settings.active_owner


def _build_autonomous_context() -> dict[str, object]:
    """Build context for autonomous market-watch runs from active owner defaults."""

    owner = settings.active_owner
    return {
        "owner_id": owner.owner_id,
        "owner_name": owner.owner_name,
        "property_id": owner.property_id,
        "property_name": owner.property_name,
        "city": owner.city,
        "region": canonicalize_region(owner.region),
        "latitude": owner.latitude,
        "longitude": owner.longitude,
    }


def _merge_market_watch_context(payload: MarketWatchRunRequest | None) -> dict[str, object]:
    """Merge optional manual override context over active-owner defaults."""

    merged = _build_autonomous_context()
    if payload is None:
        return merged

    if payload.owner_id is not None and payload.owner_id.strip():
        merged["owner_id"] = payload.owner_id.strip()
    if payload.owner_name is not None and payload.owner_name.strip():
        merged["owner_name"] = payload.owner_name.strip()
    if payload.property_id is not None and payload.property_id.strip():
        merged["property_id"] = payload.property_id.strip()
    if payload.property_name is not None and payload.property_name.strip():
        merged["property_name"] = payload.property_name.strip()
    if payload.city is not None and payload.city.strip():
        merged["city"] = payload.city.strip()
    if payload.region is not None:
        merged["region"] = canonicalize_region(payload.region)
    if payload.latitude is not None:
        merged["latitude"] = payload.latitude
    if payload.longitude is not None:
        merged["longitude"] = payload.longitude
    return merged


def _serialize_alert(record: MarketAlertRecord) -> MarketAlertResponse:
    """Convert internal alert record to API response schema."""

    return MarketAlertResponse(
        id=record.id,
        created_at_utc=record.created_at_utc,
        owner_id=record.owner_id,
        property_id=record.property_id,
        property_name=record.property_name,
        city=record.city,
        region=record.region,
        alert_type=record.alert_type,
        severity=record.severity,
        title=record.title,
        summary=record.summary,
        start_at_utc=record.start_at_utc,
        end_at_utc=record.end_at_utc,
        source_name=record.source_name,
        source_url=record.source_url,
        evidence=record.evidence,
    )


def _extract_bearer_token(authorization: str | None) -> str | None:
    """Parse Authorization header as Bearer token, returning None if absent/invalid."""

    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _assert_market_watch_trigger_authorized(
    *,
    x_market_watch_secret: str | None,
    authorization: str | None,
) -> None:
    """Enforce secret check in external cron mode while allowing local internal-mode runs."""

    if settings.market_watch_autonomous_mode != "external_cron":
        return
    expected = settings.market_watch_cron_secret
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="MARKET_WATCH_CRON_SECRET is required when MARKET_WATCH_AUTONOMOUS_MODE=external_cron.",
        )
    bearer = _extract_bearer_token(authorization)
    provided = x_market_watch_secret or bearer
    if provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized market_watch run trigger.")


def _run_market_watch_cycle() -> None:
    """Background scheduler callback that executes one autonomous market-watch cycle."""

    if not settings.market_watch_enabled:
        return
    outcome = market_watch_agent.run_autonomous(context=_build_autonomous_context())
    logger.info(
        "market_watch autonomous run finished: inserted_alerts=%s response_preview=%s",
        outcome.inserted_count,
        outcome.response[:120],
    )


market_watch_scheduler = MarketWatchScheduler(
    enabled=settings.market_watch_enabled and settings.market_watch_autonomous_enabled,
    mode=settings.market_watch_autonomous_mode,
    interval_hours=settings.market_watch_interval_hours,
    run_job=_run_market_watch_cycle,
    logger=logger,
)


def _refresh_architecture_svg() -> None:
    """Regenerate the architecture diagram, falling back to an existing file on write failure."""

    try:
        ensure_architecture_svg(ARCH_PATH)
    except Exception:
        if not ARCH_PATH.exists():
            raise
        logger.warning("model_architecture regeneration failed, serving existing SVG", exc_info=True)


@app.on_event("startup")
def startup() -> None:
    """Ensure static assets exist and start autonomous scheduler when configured."""

    try:
        _refresh_architecture_svg()
    except Exception as exc:
        logger.warning(
            "model_architecture generation skipped: %s: %s",
            type(exc).__name__,
            exc,
        )
    market_watch_scheduler.start()

    # Gmail push: renew watch in a background thread so slow DB / API calls
    # never block uvicorn from binding the port on Render.
    if settings.mail_push_enabled and settings.mail_push_topic and settings.mail_enabled:
        def _renew_watch() -> None:
            try:
                state = get_push_state(settings.database_url)
                now_ms = int(time.time() * 1000)
                expiration_ts = (state or {}).get("expiration_ts") if state else None
                if expiration_ts is None or expiration_ts <= now_ms:
                    watch_result = gmail_service.setup_watch(settings.mail_push_topic)
                    if watch_result:
                        set_push_state(
                            settings.database_url,
                            str(watch_result.get("historyId", "")),
                            expiration_ts=watch_result.get("expiration"),
                        )
                        logger.info("Gmail push watch renewed on startup")
                    else:
                        logger.warning("Gmail push watch renewal failed on startup")
                else:
                    logger.info("Gmail push watch still valid, skipping renewal")
            except Exception as exc:
                logger.warning("Gmail push watch renewal error: %s: %s", type(exc).__name__, exc)

        threading.Thread(target=_renew_watch, daemon=True, name="gmail-watch-renew").start()


@app.on_event("shutdown")
def shutdown() -> None:
    """Stop background scheduler and MCP connections cleanly on process shutdown."""

    market_watch_scheduler.stop()
    gmail_service.close()


@app.get("/", response_class=HTMLResponse)
def web_ui(request: Request) -> HTMLResponse:
    """Minimal UI for running the agent and inspecting `steps` trace."""

    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/analysis", response_class=HTMLResponse)
def analysis_ui(request: Request) -> HTMLResponse:
    """Analysis console for structured benchmarking against neighbors."""

    return templates.TemplateResponse("analysis.html", {"request": request})


@app.get("/pricing", response_class=HTMLResponse)
def pricing_ui(request: Request) -> HTMLResponse:
    """Pricing console for deterministic nightly-rate recommendations."""

    return templates.TemplateResponse("pricing.html", {"request": request})


@app.get("/labeling", response_class=HTMLResponse)
def labeling_ui(request: Request) -> HTMLResponse:
    """UI for manual relevance-label selection during threshold calibration."""

    return templates.TemplateResponse("threshold_labeling.html", {"request": request})


@app.get("/api/threshold_labeling/data", response_model=ThresholdLabelingDataResponse)
def threshold_labeling_data() -> ThresholdLabelingDataResponse:
    """Return all labeling cases merged with existing gold labels."""

    cases, labeled_cases = _build_threshold_label_cases()
    return ThresholdLabelingDataResponse(
        status="ok",
        error=None,
        total_cases=len(cases),
        labeled_cases=labeled_cases,
        cases=cases,
    )


@app.post("/api/threshold_labeling/save", response_model=ThresholdLabelSaveResponse)
def threshold_labeling_save(payload: ThresholdLabelSaveRequest) -> ThresholdLabelSaveResponse:
    """Persist one case's should-answer and relevant-vector selection to gold CSV."""

    pool_cases = _load_threshold_label_pool()
    pool_case = next((case for case in pool_cases if case.get("case_id") == payload.case_id), None)
    if pool_case is None:
        raise HTTPException(status_code=404, detail=f"Unknown case_id: {payload.case_id}")

    candidates = pool_case.get("candidates") or []
    if not isinstance(candidates, list):
        candidates = []
    valid_vector_ids = {
        str(candidate.get("vector_id", "")).strip()
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("vector_id", "")).strip()
    }

    selected_vector_ids: list[str] = []
    seen_vector_ids: set[str] = set()
    for vector_id in payload.relevant_vector_ids:
        clean_id = vector_id.strip()
        if not clean_id or clean_id in seen_vector_ids:
            continue
        seen_vector_ids.add(clean_id)
        selected_vector_ids.append(clean_id)

    invalid_vector_ids = [vector_id for vector_id in selected_vector_ids if vector_id not in valid_vector_ids]
    if invalid_vector_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid vector IDs for case {payload.case_id}: {invalid_vector_ids}",
        )
    if payload.should_answer and not selected_vector_ids:
        raise HTTPException(
            status_code=400,
            detail="should_answer=1 requires at least one selected relevant_vector_id.",
        )
    if (not payload.should_answer) and selected_vector_ids:
        raise HTTPException(
            status_code=400,
            detail="should_answer=0 requires relevant_vector_ids to be empty.",
        )

    gold_rows, _ = _load_threshold_gold_rows()
    rows_by_case: dict[str, dict[str, str]] = {
        str(row.get("case_id", "")).strip(): row for row in gold_rows if str(row.get("case_id", "")).strip()
    }
    ordered_case_ids: list[str] = []
    seen_case_ids: set[str] = set()
    for case in pool_cases:
        case_id = str(case.get("case_id", "")).strip()
        if case_id and case_id not in seen_case_ids:
            ordered_case_ids.append(case_id)
            seen_case_ids.add(case_id)
        if case_id and case_id not in rows_by_case:
            rows_by_case[case_id] = {
                "case_id": case_id,
                "should_answer": "",
                "relevant_vector_ids": "",
            }

    rows_by_case[payload.case_id] = {
        "case_id": payload.case_id,
        "should_answer": "1" if payload.should_answer else "0",
        "relevant_vector_ids": "|".join(selected_vector_ids),
    }
    _write_threshold_gold_rows([rows_by_case[case_id] for case_id in ordered_case_ids])

    return ThresholdLabelSaveResponse(status="ok", error=None, case_id=payload.case_id)


@app.get("/api/team_info", response_model=TeamInfoResponse)
def team_info() -> TeamInfoResponse:
    """Required endpoint: returns team metadata."""

    return TeamInfoResponse(
        group_batch_order_number=settings.group_batch_order_number,
        team_name=settings.team_name,
        students=[
            TeamStudentResponse(name=s.name, email=s.email) for s in settings.students
        ],
    )


@app.get("/api/active_owner_context", response_model=ActiveOwnerContextResponse)
def active_owner_context() -> ActiveOwnerContextResponse:
    """Return currently configured default owner/property context."""

    return _build_active_owner_context()


@app.get("/api/property_profiles", response_model=PropertyProfilesResponse)
def property_profiles_endpoint() -> PropertyProfilesResponse:
    """Return selectable property profiles for reviews/market-watch UI flows."""

    return _build_property_profiles_response()


@app.get("/api/agent_info", response_model=AgentInfoResponse)
def agent_info() -> AgentInfoResponse:
    """Required endpoint: returns purpose, template, and full prompt examples."""

    example_steps = [
        StepLog(
            module="dispatch",
            prompt={"user_prompt": "What do guests think about wifi in Santa Clara?"},
            response={
                "selected_agent": "reviews_agent",
                "reason": "Direct dispatch to reviews_agent.",
            },
        ),
        StepLog(
            module="reviews_agent.retrieval",
            prompt={"top_k": 8, "metadata_filter": {"region": {"$eq": "santa clara"}}},
            response={"match_count": 8, "top_match_ids": ["santa-clara:123", "santa-clara:456"]},
        ),
        StepLog(
            module="reviews_agent.evidence_guard",
            prompt={"relevance_score_threshold": 0.4, "thin_evidence_min": 1, "thin_evidence_max": 2},
            response={"decision": "answer_normal", "relevant_evidence_count": 6},
        ),
        StepLog(
            module="reviews_agent.web_quarantine_upsert",
            prompt={"attempted": 3},
            response={"status": "ok", "upserted": 3, "namespace": "airbnb-reviews-web-quarantine"},
        ),
        StepLog(
            module="reviews_agent.answer_generation",
            prompt={"model": settings.chat_model, "system_prompt": "...", "user_prompt": "..."},
            response={"text": "Guests mostly describe wifi as stable and fast, with a few weak-signal cases."},
        ),
        StepLog(
            module="reviews_agent.hallucination_guard",
            prompt={"checked_phrases": ["many guests", "a lot of reviews", "most guests", "guests generally", "everyone"]},
            response={"risk_flag": False, "matched_phrases": [], "evidence_count": 6, "action": "flag_only"},
        ),
    ]

    return AgentInfoResponse(
        description=(
            "Multi-agent-ready hospitality insights API. "
            "Enabled domain agents: reviews_agent, market_watch_agent, mail_agent, analyst_agent, and pricing_agent."
        ),
        purpose=(
            "Answer business questions from guest reviews, provide proactive market intelligence "
            "from weather/events/holiday signals, manage Airbnb email workflows, "
            "benchmark one property against its neighbors using structured listing data, "
            "and recommend a nightly rate using comp, quality, market, and review-volume signals."
        ),
        prompt_template=AgentPromptTemplate(
            template=(
                "Question: {business_question}\n"
                "Optional filters: {region/property/date}\n"
                "Goal: actionable summary with evidence and confidence"
            )
        ),
        prompt_examples=[
            AgentPromptExample(
                prompt="What do guests say about wifi quality in Santa Clara?",
                full_response=(
                    "Guests generally report good wifi stability and speed, with occasional weaker signal "
                    "in specific rooms. Confidence: medium."
                ),
                steps=example_steps,
            ),
            AgentPromptExample(
                prompt="Check my inbox and draft a reply for any guest messages.",
                full_response=(
                    "Processed 1 Airbnb email(s):\n\n"
                    "1. [guest_message] Question about early check-in (Guest: Alice)\n"
                    "   Action: draft_reply | Requires owner action\n"
                    "   Draft: Hi Alice, thank you for reaching out! Early check-in may be ..."
                ),
                steps=[
                    StepLog(
                        module="mail_agent.guest_reply_generation",
                        prompt={
                            "system_prompt": (
                                "You are a professional, friendly Airbnb host assistant. "
                                "Draft a polite, helpful reply to the guest message below."
                            ),
                            "user_prompt": (
                                "Guest name: Alice\n"
                                "Subject: Question about early check-in\n"
                                "Message: Hi, can we check in at noon?\n\nDraft a reply:"
                            ),
                        },
                        response={"text": "Hi Alice, thank you for reaching out! Early check-in may be possible depending on availability. I'll confirm 24 hours before your arrival. Looking forward to hosting you!"},
                    ),
                ],
            ),
        ],
    )


@app.get("/api/model_architecture")
def model_architecture() -> FileResponse:
    """Required endpoint: returns architecture diagram SVG."""

    try:
        _refresh_architecture_svg()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"model_architecture unavailable: {type(exc).__name__}: {exc}",
        ) from exc
    return FileResponse(path=str(ARCH_PATH), media_type="image/svg+xml", filename="model_architecture.svg")


@app.get("/api/market_watch/alerts", response_model=MarketWatchAlertsResponse)
def market_watch_alerts(
    limit: int = Query(default=20, ge=1, le=100),
    owner_id: str | None = Query(default=None),
    property_id: str | None = Query(default=None),
) -> MarketWatchAlertsResponse:
    """Return latest stored market-watch alerts scoped by optional owner/property filters."""

    owner_filter = owner_id.strip() if owner_id and owner_id.strip() else settings.active_owner.owner_id
    property_filter = (
        property_id.strip()
        if property_id and property_id.strip()
        else settings.active_owner.property_id
    )
    try:
        records = market_alert_store.list_latest_alerts(
            owner_id=owner_filter,
            property_id=property_filter,
            limit=limit,
        )
        alerts = [_serialize_alert(record) for record in records]
        return MarketWatchAlertsResponse(status="ok", error=None, alerts=alerts)
    except Exception as exc:
        return MarketWatchAlertsResponse(status="error", error=f"{type(exc).__name__}: {exc}", alerts=[])


@app.post("/api/market_watch/run", response_model=MarketWatchRunResponse)
def market_watch_run(
    payload: MarketWatchRunRequest | None = Body(default=None),
    x_market_watch_secret: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> MarketWatchRunResponse:
    """Trigger one market-watch cycle (manual local use or secured external cron use)."""

    _assert_market_watch_trigger_authorized(
        x_market_watch_secret=x_market_watch_secret,
        authorization=authorization,
    )
    if not settings.market_watch_enabled:
        return MarketWatchRunResponse(
            status="error",
            error="market_watch is disabled (MARKET_WATCH_ENABLED=false).",
            response=None,
            inserted_alerts=0,
            steps=[],
        )

    try:
        context = _merge_market_watch_context(payload)
        outcome = market_watch_agent.run_autonomous(context=context)
        return MarketWatchRunResponse(
            status="ok",
            error=None,
            response=outcome.response,
            inserted_alerts=outcome.inserted_count,
            steps=outcome.steps,
        )
    except Exception as exc:
        return MarketWatchRunResponse(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            response=None,
            inserted_alerts=0,
            steps=[],
        )


@app.post("/api/analysis", response_model=AnalysisResponse)
def run_analysis(payload: AnalysisRequest) -> AnalysisResponse:
    """Run prompt-driven neighbor benchmarking analysis for one property."""

    target_agent, provider_error = _resolve_analysis_agent(payload.llm_provider)
    if provider_error:
        return AnalysisResponse(
            status="error",
            error=provider_error,
            response=None,
            analysis_category=None,
            numeric_comparison=[],
            categorical_comparison=[],
            steps=[],
        )

    property_id = (
        payload.property_id.strip()
        if payload.property_id and payload.property_id.strip()
        else settings.active_owner.property_id
    )
    context: dict[str, object] = {"property_id": property_id}
    if payload.category in {"review_scores", "property_specs"}:
        context["analysis_category"] = payload.category

    prompt = (
        payload.prompt.strip()
        if payload.prompt and payload.prompt.strip()
        else f"Analyze my {payload.category or 'review_scores'} against neighbors."
    )

    try:
        assert target_agent is not None
        outcome = target_agent.analyze(prompt, context=context)
        return AnalysisResponse(
            status=("error" if outcome.error else "ok"),
            error=outcome.error,
            response=(None if outcome.error else outcome.narrative),
            analysis_category=outcome.analysis_category,
            numeric_comparison=outcome.numeric_comparison,
            categorical_comparison=outcome.categorical_comparison,
            steps=outcome.steps,
        )
    except Exception as exc:
        return AnalysisResponse(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            response=None,
            analysis_category=None,
            numeric_comparison=[],
            categorical_comparison=[],
            steps=[],
        )


@app.post("/api/pricing", response_model=PricingResponse)
def run_pricing(payload: PricingRequest) -> PricingResponse:
    """Run prompt-driven pricing recommendation for one property."""

    if not settings.pricing_enabled:
        return PricingResponse(
            status="error",
            error="pricing_agent is disabled (PRICING_ENABLED=false).",
            response=None,
            recommendation=None,
            signals=None,
            steps=[],
        )
    if payload.horizon_days > settings.pricing_max_horizon_days:
        return PricingResponse(
            status="error",
            error=(
                f"horizon_days must be <= configured PRICING_MAX_HORIZON_DAYS "
                f"({settings.pricing_max_horizon_days})."
            ),
            response=None,
            recommendation=None,
            signals=None,
            steps=[],
        )

    target_agent, provider_error = _resolve_pricing_agent(payload.llm_provider)
    if provider_error:
        return PricingResponse(
            status="error",
            error=provider_error,
            response=None,
            recommendation=None,
            signals=None,
            steps=[],
        )

    owner = _owner_for_property_id(payload.property_id)
    property_id = (
        payload.property_id.strip()
        if payload.property_id and payload.property_id.strip()
        else owner.property_id
    )
    context: dict[str, object] = {
        "property_id": property_id,
        "horizon_days": payload.horizon_days,
        "price_mode": payload.price_mode,
    }
    if owner is not None:
        context.update(
            {
                "property_name": owner.property_name,
                "latitude": owner.latitude,
                "longitude": owner.longitude,
            }
        )
    prompt = (
        payload.prompt.strip()
        if payload.prompt and payload.prompt.strip()
        else "What should I charge for the selected horizon?"
    )
    try:
        assert target_agent is not None
        outcome = target_agent.recommend(prompt, context=context)
        return outcome_to_response(outcome)
    except Exception as exc:
        return PricingResponse(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            response=None,
            recommendation=None,
            signals=None,
            steps=[],
        )


@app.post("/api/analysis/explain-selection", response_model=AnalysisExplainSelectionResponse)
def explain_analysis_selection(payload: AnalysisExplainSelectionRequest) -> AnalysisExplainSelectionResponse:
    """Explain one selected visualization element from the analysis console."""

    target_agent, provider_error = _resolve_analysis_agent(payload.llm_provider)
    if provider_error:
        return AnalysisExplainSelectionResponse(
            status="error",
            error=provider_error,
            response=None,
            steps=[],
        )

    try:
        assert target_agent is not None
        result = target_agent.explain_selection(
            prompt=payload.prompt,
            property_id=payload.property_id,
            category=payload.category,
            selection_type=payload.selection_type,
            metric_column=payload.metric_column,
            selection_payload=payload.selection_payload,
        )
        return AnalysisExplainSelectionResponse(
            status="ok",
            error=None,
            response=result.response,
            steps=result.steps,
        )
    except Exception as exc:
        return AnalysisExplainSelectionResponse(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            response=None,
            steps=[],
        )


@app.post("/api/execute", response_model=ExecuteResponse)
def execute(payload: ExecuteRequest) -> ExecuteResponse:
    """Required endpoint: run reviews agent and return full trace."""

    steps: list[StepLog] = []
    try:
        requested_provider = (payload.llm_provider or "default").strip().lower()
        provider_is_explicit = requested_provider in VALID_CHAT_PROVIDERS
        resolved_provider = (
            default_chat_provider
            if requested_provider in {"", "default"}
            else requested_provider
        )
        if resolved_provider not in VALID_CHAT_PROVIDERS:
            resolved_provider = default_chat_provider

        steps.append(StepLog(
            module="dispatch",
            prompt={"user_prompt": payload.prompt},
            response={
                "selected_agent": "reviews_agent",
                "reason": "Direct dispatch to reviews_agent.",
                "llm_provider_requested": requested_provider,
                "llm_provider_resolved": resolved_provider,
            },
        ))

        provider_chat_service = chat_services_by_provider.get(resolved_provider)
        if provider_is_explicit and (provider_chat_service is None or not provider_chat_service.is_available):
            return ExecuteResponse(
                status="error",
                error=(
                    f"Requested llm_provider '{resolved_provider}' is not configured. "
                    f"Set required provider env vars and retry."
                ),
                response=None,
                steps=steps,
            )
        target_agent = reviews_agents_by_provider.get(resolved_provider)
        if target_agent is None:
            target_agent = reviews_agents_by_provider.get(default_chat_provider)
        if target_agent is None:
            raise HTTPException(status_code=500, detail="No reviews_agent available")

        context = _build_effective_context(payload)
        result = target_agent.run(payload.prompt, context=context)
        steps.extend(result.steps)
        for step in result.steps:
            if step.module == "reviews_agent.hallucination_guard" and step.response.get("risk_flag") is True:
                logger.warning(
                    "Hallucination-risk flag raised by reviews_agent. matched_phrases=%s evidence_count=%s",
                    step.response.get("matched_phrases"),
                    step.response.get("evidence_count"),
                )
        return ExecuteResponse(status="ok", error=None, response=result.response, steps=steps)
    except Exception as exc:
        return ExecuteResponse(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            response=None,
            steps=steps,
        )


# ---------------------------------------------------------------------------
# Mail agent endpoints
# ---------------------------------------------------------------------------


def _try_auto_send_good_review(action: dict, gmail_svc: GmailService) -> bool:
    """If action is a good review (rating > 3) with draft and reply metadata, send reply. Returns True if sent, False otherwise."""
    if action.get("category") != "new_property_review":
        return False
    rating = action.get("rating")
    if rating is None or rating <= 3:
        return False
    draft = action.get("draft")
    thread_id = action.get("thread_id")
    reply_to = action.get("reply_to")
    if not draft or not thread_id or not reply_to:
        return False
    subject = action.get("subject") or "Re: (no subject)"
    ok = gmail_svc.send_reply(
        thread_id=thread_id,
        to=reply_to,
        subject=subject,
        body=draft,
        in_reply_to=action.get("in_reply_to"),
        references=action.get("references"),
    )
    if ok:
        logger.info("Auto-sent good review reply for email_id=%s", action.get("email_id"))
    else:
        logger.warning("Auto-send good review failed for email_id=%s", action.get("email_id"))
    return ok


def _notify_owner_for_mail_actions(
    mail_actions: list[dict],
    gmail_svc: GmailService,
    notify_email: str | None,
    app_base_url: str | None,
) -> None:
    """If any action requires_owner and notify_email is set, send a short summary email with link."""
    if not notify_email or not mail_actions:
        return
    needing = [a for a in mail_actions if a.get("requires_owner")]
    if not needing:
        return
    lines: list[str] = []
    for a in needing:
        cat = a.get("category", "unknown")
        subject = a.get("subject", "")
        guest = a.get("guest_name") or "Guest"
        action = a.get("action", "")
        if cat == "new_property_review" and a.get("rating") is not None and a.get("rating") <= 3:
            snippet = (a.get("reason") or a.get("snippet", ""))[:200]
            lines.append(f"Bad review: {guest}, {a.get('rating')}/5 — {snippet}")
        else:
            lines.append(f"[{cat}] {subject} (Guest: {guest}) — {action}")
    body = "Mail agent: action(s) need your attention.\n\n" + "\n\n".join(lines)
    if app_base_url:
        body += f"\n\nOpen: {app_base_url.rstrip('/')}/mail"
    gmail_svc.send_message(notify_email, "Mail agent: action needed", body)


@app.get("/mail", response_class=HTMLResponse)
def mail_ui(request: Request) -> HTMLResponse:
    """Mail agent UI for inbox triage and owner HITL decisions."""

    return templates.TemplateResponse("mail.html", {"request": request})


@app.get("/market-watch", response_class=HTMLResponse)
def market_watch_ui(request: Request) -> HTMLResponse:
    """Market watch alerts page — shows weather, events, and demand signals."""

    return templates.TemplateResponse("market_watch.html", {"request": request})


@app.get("/api/mail/inbox", response_model=MailInboxResponse)
def mail_inbox() -> MailInboxResponse:
    """Return classified inbox items AND agent-processed actions for the mail agent UI."""

    if not settings.mail_enabled:
        return MailInboxResponse(
            status="error",
            error="mail_agent is disabled (MAIL_ENABLED=false).",
            items=[],
            demo_mode=False,
        )
    try:
        items_raw = mail_agent.get_inbox_summary()
        items = [
            MailInboxItemResponse(
                email_id=item["email_id"],
                subject=item["subject"],
                sender=item["sender"],
                date=item["date"],
                category=item["category"],
                confidence=item["confidence"],
                guest_name=item.get("guest_name"),
                rating=item.get("rating"),
                snippet=item["snippet"],
            )
            for item in items_raw
        ]

        mail_actions: list[dict] | None = None
        try:
            messages = mail_agent.gmail_service.list_unread_messages(
                max_results=mail_agent.config.max_inbox_fetch,
            )
            if messages:
                context = {
                    "owner_id": settings.active_owner.owner_id,
                    "owner_name": settings.active_owner.owner_name,
                    "property_id": settings.active_owner.property_id,
                    "property_name": settings.active_owner.property_name,
                }
                result = mail_agent.run_on_messages(messages, context=context)
                mail_actions = result.mail_actions
                if mail_actions:
                    prefs = get_mail_preferences(settings.database_url)
                    auto_send = prefs.get("auto_send_good_reviews", False)
                    auto_send_failed = False
                    for action in mail_actions:
                        if not action.get("requires_owner"):
                            continue
                        # Stop auto-sending after first failure (likely 429 rate limit).
                        if auto_send and not auto_send_failed:
                            if _try_auto_send_good_review(action, gmail_service):
                                continue
                            elif action.get("category") == "new_property_review" and (action.get("rating") or 0) > 3 and action.get("draft"):
                                auto_send_failed = True
                        notification_store.add_notification(action)
        except Exception as proc_exc:
            logger.warning("mail inbox auto-process failed: %s", proc_exc)

        return MailInboxResponse(
            status="ok",
            error=None,
            items=items,
            demo_mode=gmail_service.is_demo_mode,
            mail_actions=mail_actions,
        )
    except Exception as exc:
        return MailInboxResponse(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            items=[],
            demo_mode=gmail_service.is_demo_mode,
        )


# ---------------------------------------------------------------------------
# Mail notification endpoints
# ---------------------------------------------------------------------------


@app.get("/api/mail/notifications", response_model=NotificationsResponse)
def mail_notifications() -> NotificationsResponse:
    """Return all pending notifications that need owner attention."""
    try:
        pending = notification_store.get_pending()
        items = [NotificationItem(**n) for n in pending]
        return NotificationsResponse(status="ok", notifications=items, count=len(items))
    except Exception as exc:
        return NotificationsResponse(
            status="error", error=f"{type(exc).__name__}: {exc}"
        )


@app.get("/api/mail/notifications/stream")
async def mail_notifications_stream() -> StreamingResponse:
    """SSE endpoint that pushes notification events to the browser in real-time."""

    async def event_generator():
        queue = notification_store.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            notification_store.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/mail/notifications/{notification_id}/dismiss")
def dismiss_notification(notification_id: str) -> dict[str, str | bool]:
    """Dismiss a notification without taking action."""
    ok = notification_store.dismiss(notification_id)
    if ok:
        return {"status": "ok", "dismissed": True}
    return {"status": "error", "error": "Notification not found or already handled"}


@app.post("/api/mail/notifications/dismiss-all")
def dismiss_all_notifications() -> dict[str, Any]:
    """Dismiss every pending notification at once."""
    ids = notification_store.dismiss_all()
    return {"status": "ok", "dismissed_count": len(ids)}


# ---------------------------------------------------------------------------
# Evidence flags
# ---------------------------------------------------------------------------


@app.post("/api/evidence/flag")
def flag_evidence(payload: EvidenceFlagRequest) -> dict[str, bool]:
    """Flag a piece of evidence as irrelevant (saved for future retrieval tuning)."""
    ok = evidence_flag_store.add_flag(
        vector_id=payload.vector_id,
        query_text=payload.query_text,
        flag=payload.flag,
    )
    return {"ok": ok}


# ---------------------------------------------------------------------------
# Nav badges
# ---------------------------------------------------------------------------


@app.get("/api/nav/badges")
def nav_badges(
    market_since: str | None = None,
    property_id: str | None = None,
) -> dict[str, Any]:
    """Lightweight counts for navigation badge bubbles."""
    mail_count = notification_store.count_pending()
    market_count = 0
    property_filter = property_id.strip() if property_id and property_id.strip() else None
    if market_since:
        try:
            market_count = market_alert_store.count_since(
                market_since,
                property_id=property_filter,
            )
        except Exception:
            pass
    return {"mail_count": mail_count, "market_count": market_count}


# ---------------------------------------------------------------------------
# Mail settings (persisted preferences)
# ---------------------------------------------------------------------------


@app.get("/api/mail/settings", response_model=MailSettingsResponse)
def get_mail_settings() -> MailSettingsResponse:
    """Return current mail preferences (e.g. auto_send_good_reviews)."""
    try:
        prefs = get_mail_preferences(settings.database_url)
        return MailSettingsResponse(
            status="ok",
            auto_send_good_reviews=prefs.get("auto_send_good_reviews", False),
        )
    except Exception as exc:
        return MailSettingsResponse(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            auto_send_good_reviews=False,
        )


@app.post("/api/mail/settings", response_model=MailSettingsResponse)
def update_mail_settings(payload: MailSettingsUpdateRequest) -> MailSettingsResponse:
    """Update mail preferences (e.g. auto_send_good_reviews)."""
    try:
        set_mail_preferences(settings.database_url, payload.auto_send_good_reviews)
        return MailSettingsResponse(
            status="ok",
            auto_send_good_reviews=payload.auto_send_good_reviews,
        )
    except Exception as exc:
        return MailSettingsResponse(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            auto_send_good_reviews=False,
        )


def _history_id_as_int(value: str) -> int | None:
    """Parse Gmail history IDs safely for max/coalescing comparisons."""

    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _enqueue_mail_push_history(history_id: str) -> bool:
    """Store latest pending history ID and return True when worker should start."""

    global _push_worker_running, _pending_push_history_id

    normalized = str(history_id).strip()
    if not normalized:
        return False

    with _push_worker_state_lock:
        if _pending_push_history_id is None:
            _pending_push_history_id = normalized
        else:
            current_int = _history_id_as_int(_pending_push_history_id)
            incoming_int = _history_id_as_int(normalized)
            if current_int is None or incoming_int is None or incoming_int >= current_int:
                _pending_push_history_id = normalized

        if _push_worker_running:
            return False
        _push_worker_running = True
        return True


def _take_pending_mail_push_history() -> str | None:
    """Pop one pending history ID for worker processing."""

    global _pending_push_history_id
    with _push_worker_state_lock:
        value = _pending_push_history_id
        _pending_push_history_id = None
        return value


def _mark_mail_push_worker_idle_if_drained() -> bool:
    """Set worker as idle only when queue is still empty."""

    global _push_worker_running
    with _push_worker_state_lock:
        if _pending_push_history_id is None:
            _push_worker_running = False
            return True
        return False


def _process_mail_push_history(history_id: str) -> None:
    """Run one serialized push-processing cycle for one target history ID."""

    # Serialize Gmail API access — httplib2 is not thread-safe.
    with _push_lock:
        db_url = settings.database_url
        state = get_push_state(db_url)
        last_history_id = state.get("history_id") if state else None
        if not last_history_id:
            set_push_state(db_url, str(history_id))
            return

        # Advance history cursor immediately so the worker can coalesce bursts
        # without re-fetching old batches.
        set_push_state(db_url, str(history_id))

        try:
            messages = gmail_service.list_messages_since_history(last_history_id)
        except Exception as exc:
            if "404" in str(exc) or "history" in str(exc).lower():
                return
            logger.warning("mail push: list_messages_since_history failed: %s", exc)
            return

        if not messages:
            return

        result = mail_agent.run_on_messages(messages)
        actions = result.mail_actions or []
        prefs = get_mail_preferences(db_url)
        auto_send = prefs.get("auto_send_good_reviews", False)
        auto_send_failed = False
        actions_needing_notify: list[dict] = []
        for action in actions:
            if not action.get("requires_owner"):
                continue
            # Stop auto-sending after first failure (likely 429 rate limit).
            if auto_send and not auto_send_failed:
                if _try_auto_send_good_review(action, gmail_service):
                    continue
                # Only flag as failed if it was a sendable good review.
                if (
                    action.get("category") == "new_property_review"
                    and (action.get("rating") or 0) > 3
                    and action.get("draft")
                ):
                    auto_send_failed = True
            notification_store.add_notification(action)
            actions_needing_notify.append(action)
        if actions_needing_notify:
            _notify_owner_for_mail_actions(
                actions_needing_notify,
                gmail_service,
                settings.mail_owner_notify_email,
                settings.app_base_url,
            )


def _mail_push_worker() -> None:
    """Process coalesced Gmail push notifications without blocking request threads."""

    while True:
        history_id = _take_pending_mail_push_history()
        if history_id is None:
            if _mark_mail_push_worker_idle_if_drained():
                return
            continue
        try:
            _process_mail_push_history(history_id)
        except Exception as exc:
            logger.warning("mail push worker cycle failed: %s: %s", type(exc).__name__, exc)


@app.post("/api/mail/push")
def mail_push(
    request: Request,
    x_gmail_push_secret: str | None = Header(default=None, alias="X-Gmail-Push-Secret"),
    payload: dict | None = Body(default=None),
) -> dict[str, str]:
    """Gmail push webhook.

    Request path stays fast: it validates and enqueues latest history ID, then
    a single background worker performs Gmail/LLM/DB processing.
    """
    try:
        body = payload or {}
        if not isinstance(body, dict):
            return {"status": "ok"}
        msg = body.get("message", {})
        if not isinstance(msg, dict):
            return {"status": "ok"}
        data_b64 = msg.get("data")
        if not data_b64:
            return {"status": "ok"}
        secret = settings.mail_push_webhook_secret
        if secret and secret.strip():
            provided = x_gmail_push_secret or (request.headers.get("Authorization") or "").replace("Bearer ", "").strip()
            if provided != secret:
                raise HTTPException(status_code=401, detail="Invalid or missing push secret")
        try:
            data_bytes = base64.b64decode(data_b64)
            data = json.loads(data_bytes.decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            return {"status": "ok"}
        history_id = data.get("historyId") if isinstance(data, dict) else None
        if not history_id:
            return {"status": "ok"}
        if not settings.mail_enabled or not settings.mail_push_enabled:
            return {"status": "ok"}
        start_worker = _enqueue_mail_push_history(str(history_id))
        if start_worker:
            threading.Thread(
                target=_mail_push_worker,
                daemon=True,
                name="gmail-push-worker",
            ).start()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("mail push handler error: %s", exc)
    return {"status": "ok"}


@app.post("/api/mail/watch")
def mail_watch() -> dict:
    """Register Gmail push watch when mail_push_enabled and mail_push_topic are set."""
    try:
        if not settings.mail_push_enabled or not settings.mail_push_topic:
            return {"status": "error", "error": "Mail push or topic not configured", "ok": False}
        if not settings.mail_enabled:
            return {"status": "error", "error": "Mail agent disabled", "ok": False}
        is_demo = gmail_service.is_demo_mode
        has_api = gmail_service._api_client is not None
        if is_demo and not has_api:
            return {
                "status": "error",
                "ok": False,
                "error": f"Gmail service is in demo mode (no API client). Check GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN env vars.",
                "debug": {"demo_mode": is_demo, "has_api_client": has_api},
            }
        watch_result = gmail_service.setup_watch(settings.mail_push_topic)
        if not watch_result:
            return {"status": "error", "error": "setup_watch returned None", "ok": False}
        set_push_state(
            settings.database_url,
            str(watch_result.get("historyId", "")),
            expiration_ts=watch_result.get("expiration"),
        )
        return {"status": "ok", "ok": True, "historyId": str(watch_result.get("historyId")), "expiration": str(watch_result.get("expiration"))}
    except Exception as exc:
        logger.exception("mail_watch failed")
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "ok": False}


@app.post("/api/mail/action", response_model=MailActionResponse)
def mail_action(payload: MailActionRequest) -> MailActionResponse:
    """Execute an owner HITL action (rate guest, approve draft, don't reply, approve and send, etc.)."""

    if not settings.mail_enabled:
        return MailActionResponse(
            status="error",
            error="mail_agent is disabled (MAIL_ENABLED=false).",
            response=None,
            steps=[],
        )
    try:
        if payload.don_t_reply:
            set_owner_choice(settings.database_url, payload.email_id, "don_t_reply")
            notification_store.mark_handled_by_email(payload.email_id)
            return MailActionResponse(
                status="ok",
                error=None,
                response="Owner chose not to reply.",
                steps=[],
            )
        if payload.approve_and_send and payload.edited_draft:
            thread_id = payload.thread_id
            reply_to_addr = payload.reply_to
            subject = payload.subject
            if not thread_id or not reply_to_addr:
                return MailActionResponse(
                    status="error",
                    error="approve_and_send requires thread_id and reply_to.",
                    response=None,
                    steps=[],
                )
            if not subject:
                subject = "Re: (no subject)"
            ok = gmail_service.send_reply(
                thread_id=thread_id,
                to=reply_to_addr,
                subject=subject,
                body=payload.edited_draft,
                in_reply_to=payload.in_reply_to,
                references=payload.references,
            )
            if not ok:
                return MailActionResponse(
                    status="error",
                    error="Send failed, please try again.",
                    response=None,
                    steps=[],
                )
            notification_store.mark_handled_by_email(payload.email_id)
            return MailActionResponse(
                status="ok",
                error=None,
                response="Reply sent.",
                steps=[],
            )
        owner_action: dict[str, object] = {
            "email_id": payload.email_id,
            "action_type": payload.action_type,
        }
        if payload.rating is not None:
            owner_action["rating"] = payload.rating
        if payload.issues is not None:
            owner_action["issues"] = payload.issues
        if payload.free_text is not None:
            owner_action["free_text"] = payload.free_text
        if payload.approved is not None:
            owner_action["approved"] = payload.approved
        if payload.edited_draft is not None:
            owner_action["edited_draft"] = payload.edited_draft
        if payload.owner_instructions is not None:
            owner_action["owner_instructions"] = payload.owner_instructions
        if payload.reply_style is not None:
            owner_action["reply_style"] = payload.reply_style
        if payload.don_t_reply is not None:
            owner_action["don_t_reply"] = payload.don_t_reply
        if payload.approve_and_send is not None:
            owner_action["approve_and_send"] = payload.approve_and_send

        context = {
            "owner_id": settings.active_owner.owner_id,
            "owner_name": settings.active_owner.owner_name,
            "property_id": settings.active_owner.property_id,
            "property_name": settings.active_owner.property_name,
        }
        result = mail_agent.run_with_action(
            prompt=f"mail action: {payload.action_type}",
            owner_action=owner_action,
            context=context,
        )
        return MailActionResponse(
            status="ok",
            error=None,
            response=result.response,
            steps=result.steps,
            mail_actions=result.mail_actions,
        )
    except Exception as exc:
        return MailActionResponse(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            response=None,
            steps=[],
        )
