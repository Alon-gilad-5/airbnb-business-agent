"""Phase 0 contract tests: verify agent response shapes, module names, and error paths.

These tests run with dummy services (no live API calls) and assert the exact
response contract that the FastAPI endpoints depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from app.agents.reviews_agent import ReviewsAgent, ReviewsAgentConfig
from app.agents.analyst_agent import AnalystAgent
from app.agents.market_watch_agent import MarketWatchAgent, MarketWatchAgentConfig
from app.schemas import AnalysisExplainSelectionRequest, AnalysisRequest, ExecuteRequest, PricingRequest
from app.services.pinecone_retriever import RetrievedReview
from app.services.web_review_ingest import WebIngestResult
from app.services.web_review_scraper import ScrapedReview


# -- Dummy services --


@dataclass
class DummyEmbeddingService:
    is_available: bool = True

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


@dataclass
class DummyRetriever:
    is_available: bool = True
    _matches: list[RetrievedReview] | None = None

    def query(
        self,
        *,
        embedding: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedReview]:
        if self._matches is not None:
            return self._matches
        return []


@dataclass
class DummyChatService:
    is_available: bool = True
    model: str = "test-model"
    _response: str = "Test LLM answer about wifi."

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        if not self.is_available:
            raise RuntimeError("Chat service unavailable")
        return self._response


class DummyWebScraper:
    def __init__(self, *, available: bool = False) -> None:
        self.is_available = available

    def scrape_reviews(self, **kwargs: Any) -> tuple[list[ScrapedReview], dict[str, Any]]:
        return [], {"status": "disabled", "raw_count": 0, "deduped_count": 0}


class DummyWebIngest:
    is_available = False

    def upsert_scraped_reviews(self, **kwargs: Any) -> WebIngestResult:
        return WebIngestResult(attempted=0, upserted=0, vector_ids=[], namespace="test")


# -- Market watch dummies --


class DummyProviders:
    def fetch_weather_forecast(self, **kwargs: Any) -> tuple[list, dict]:
        return [], {"source": "test", "status": "ok"}

    def fetch_ticketmaster_events(self, **kwargs: Any) -> tuple[list, dict]:
        return [], {"source": "test", "status": "ok"}

    def fetch_us_public_holidays(self, **kwargs: Any) -> tuple[list, dict]:
        return [], {"source": "test", "status": "ok"}


class DummyAlertStore:
    def insert_alerts(self, records: list) -> int:
        return len(records)


class DummyNeighborStore:
    def __init__(self, neighbors: list[str] | None = None) -> None:
        self._neighbors = neighbors or ["n1", "n2"]

    def get_neighbors(self, property_id: str) -> list[str] | None:
        return list(self._neighbors)


class DummyListingStore:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or [
            {
                "id": "42409434",
                "name": "The Burlington Hotel",
                "review_scores_rating": 4.55,
                "review_scores_accuracy": 4.40,
                "review_scores_cleanliness": 4.60,
                "review_scores_checkin": 4.70,
                "review_scores_communication": 4.75,
                "review_scores_location": 4.10,
                "review_scores_value": 4.00,
                "property_type": "Room in hotel",
                "room_type": "Private room",
                "accommodates": 2,
                "bathrooms": 1,
                "bedrooms": 1,
                "beds": 1,
                "price": "$120.00",
                "host_is_superhost": "t",
            },
            {
                "id": "n1",
                "name": "Neighbor One",
                "review_scores_rating": 4.30,
                "review_scores_accuracy": 4.10,
                "review_scores_cleanliness": 4.20,
                "review_scores_checkin": 4.40,
                "review_scores_communication": 4.35,
                "review_scores_location": 4.00,
                "review_scores_value": 4.10,
                "property_type": "Room in hotel",
                "room_type": "Private room",
                "accommodates": 2,
                "bathrooms": 1,
                "bedrooms": 1,
                "beds": 1,
                "price": "$115.00",
                "host_is_superhost": "f",
            },
            {
                "id": "n2",
                "name": "Neighbor Two",
                "review_scores_rating": 4.10,
                "review_scores_accuracy": 4.05,
                "review_scores_cleanliness": 4.15,
                "review_scores_checkin": 4.25,
                "review_scores_communication": 4.20,
                "review_scores_location": 4.05,
                "review_scores_value": 3.90,
                "property_type": "Apartment",
                "room_type": "Entire home/apt",
                "accommodates": 3,
                "bathrooms": 1,
                "bedrooms": 1,
                "beds": 2,
                "price": "$140.00",
                "host_is_superhost": "t",
            },
        ]

    def get_listings_by_ids(self, listing_ids: list[str], columns: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self._rows:
            if row["id"] not in listing_ids:
                continue
            out.append({column: row.get(column) for column in ["id", "name", *columns]})
        return out


# -- API schema contract tests --


def test_execute_request_without_provider_is_valid() -> None:
    payload = ExecuteRequest(prompt="What do guests say about wifi?")
    assert payload.llm_provider is None


def test_execute_request_with_openrouter_provider_is_valid() -> None:
    payload = ExecuteRequest(prompt="What do guests say about wifi?", llm_provider="openrouter")
    assert payload.llm_provider == "openrouter"


def test_analysis_request_with_openrouter_provider_is_valid() -> None:
    payload = AnalysisRequest(
        prompt="How do I compare to nearby competitors on cleanliness?",
        llm_provider="openrouter",
    )
    assert payload.prompt == "How do I compare to nearby competitors on cleanliness?"
    assert payload.llm_provider == "openrouter"


def test_analysis_explain_selection_request_is_valid() -> None:
    payload = AnalysisExplainSelectionRequest(
        property_id="42409434",
        prompt="How are my review scores compared to my neighbors?",
        category="review_scores",
        selection_type="numeric_point",
        metric_column="review_scores_rating",
        selection_payload={"listing_id": "n1", "selected_value": 4.7},
        llm_provider="openrouter",
    )
    assert payload.metric_column == "review_scores_rating"
    assert payload.llm_provider == "openrouter"


def test_pricing_request_with_openrouter_provider_is_valid() -> None:
    payload = PricingRequest(
        prompt="What should I charge next weekend?",
        property_id="42409434",
        llm_provider="openrouter",
        horizon_days=7,
        price_mode="conservative",
    )
    assert payload.llm_provider == "openrouter"
    assert payload.horizon_days == 7
    assert payload.price_mode == "conservative"


def test_execute_request_with_invalid_provider_fails_validation() -> None:
    with pytest.raises(ValidationError):
        ExecuteRequest(prompt="What do guests say about wifi?", llm_provider="invalid-provider")  # type: ignore[arg-type]


# -- Reviews Agent contract tests --


REVIEWS_MODULE_NAMES = {
    "reviews_agent.retrieval",
    "reviews_agent.web_scrape",
    "reviews_agent.web_quarantine_upsert",
    "reviews_agent.evidence_guard",
    "reviews_agent.answer_generation",
    "reviews_agent.hallucination_guard",
}


def _make_reviews_agent(
    *,
    embedding_available: bool = True,
    retriever_available: bool = True,
    retriever_matches: list[RetrievedReview] | None = None,
    chat_available: bool = True,
    chat_response: str = "Test wifi answer.",
) -> ReviewsAgent:
    return ReviewsAgent(
        embedding_service=DummyEmbeddingService(is_available=embedding_available),
        retriever=DummyRetriever(is_available=retriever_available, _matches=retriever_matches),
        chat_service=DummyChatService(is_available=chat_available, _response=chat_response),
        web_scraper=DummyWebScraper(available=False),
        web_ingest_service=DummyWebIngest(),
        config=ReviewsAgentConfig(relevance_score_threshold=0.40),
    )


def test_reviews_no_matches_returns_no_evidence() -> None:
    agent = _make_reviews_agent(retriever_matches=[])
    result = agent.run("What do guests say about wifi?", context={"property_id": "42409434", "region": "los angeles"})
    assert result.response == "I couldn't find enough data to answer your question."
    module_names = {s.module for s in result.steps}
    assert "reviews_agent.retrieval" in module_names
    assert "reviews_agent.evidence_guard" in module_names


def test_reviews_with_matches_produces_answer_and_all_modules() -> None:
    matches = [
        RetrievedReview(vector_id="v1", score=0.85, metadata={
            "review_text": "Wifi was excellent and fast.",
            "property_id": "42409434", "region": "los angeles", "review_date": "2024-01-01",
        }),
        RetrievedReview(vector_id="v2", score=0.72, metadata={
            "review_text": "Good internet connection.",
            "property_id": "42409434", "region": "los angeles", "review_date": "2024-02-15",
        }),
        RetrievedReview(vector_id="v3", score=0.68, metadata={
            "review_text": "Wifi worked well throughout stay.",
            "property_id": "42409434", "region": "los angeles", "review_date": "2024-03-10",
        }),
    ]
    agent = _make_reviews_agent(retriever_matches=matches)
    result = agent.run("What do guests say about wifi?", context={"property_id": "42409434", "region": "los angeles"})

    assert result.response is not None
    assert isinstance(result.response, str)
    assert len(result.response) > 0

    module_names = [s.module for s in result.steps]
    assert "reviews_agent.retrieval" in module_names
    assert "reviews_agent.evidence_guard" in module_names
    assert "reviews_agent.answer_generation" in module_names
    assert "reviews_agent.hallucination_guard" in module_names

    for step in result.steps:
        assert step.module in REVIEWS_MODULE_NAMES
        assert isinstance(step.prompt, dict)
        assert isinstance(step.response, dict)


def test_reviews_embedding_unavailable() -> None:
    agent = _make_reviews_agent(embedding_available=False)
    result = agent.run("What do guests say about wifi?")
    assert "Embedding service is not configured" in result.response
    assert len(result.steps) == 0


def test_reviews_retriever_unavailable() -> None:
    agent = _make_reviews_agent(retriever_available=False)
    result = agent.run("What do guests say about wifi?")
    assert "Pinecone is not configured" in result.response
    assert len(result.steps) == 0


def test_reviews_chat_unavailable_uses_deterministic_fallback() -> None:
    matches = [
        RetrievedReview(vector_id="v1", score=0.85, metadata={
            "review_text": "Wifi was excellent.", "property_id": "p1", "region": "la",
        }),
        RetrievedReview(vector_id="v2", score=0.72, metadata={
            "review_text": "Good internet.", "property_id": "p1", "region": "la",
        }),
        RetrievedReview(vector_id="v3", score=0.68, metadata={
            "review_text": "Wifi worked well.", "property_id": "p1", "region": "la",
        }),
    ]
    agent = _make_reviews_agent(retriever_matches=matches, chat_available=False)
    result = agent.run("What do guests say about wifi?")
    assert "LLM synthesis service is currently unavailable" in result.response
    module_names = {s.module for s in result.steps}
    assert "reviews_agent.retrieval" in module_names
    assert "reviews_agent.evidence_guard" in module_names
    assert "reviews_agent.hallucination_guard" in module_names


def test_reviews_thin_evidence_produces_disclaimer() -> None:
    matches = [
        RetrievedReview(vector_id="v1", score=0.85, metadata={
            "review_text": "Wifi was excellent.", "property_id": "p1", "region": "la",
        }),
    ]
    agent = _make_reviews_agent(retriever_matches=matches)
    result = agent.run("What do guests say about wifi?")
    assert "Evidence is limited" in result.response


# -- Market Watch Agent contract tests --


MARKET_WATCH_MODULE_NAMES = {
    "market_watch_agent.signal_collection",
    "market_watch_agent.weather_analysis",
    "market_watch_agent.event_analysis",
    "market_watch_agent.demand_analysis",
    "market_watch_agent.alert_decision",
    "market_watch_agent.inbox_write",
    "market_watch_agent.answer_generation",
}

ANALYST_MODULE_NAMES = {
    "analyst_agent.neighbor_lookup",
    "analyst_agent.data_fetch",
    "analyst_agent.comparison_compute",
    "analyst_agent.answer_generation",
}


def _make_market_agent() -> MarketWatchAgent:
    return MarketWatchAgent(
        providers=DummyProviders(),
        alert_store=DummyAlertStore(),
        config=MarketWatchAgentConfig(),
    )


def _make_analyst_agent(
    *,
    chat_available: bool = True,
    chat_response: str = "Structured benchmark narrative.",
) -> AnalystAgent:
    return AnalystAgent(
        listing_store=DummyListingStore(),
        neighbor_store=DummyNeighborStore(),
        chat_service=DummyChatService(is_available=chat_available, _response=chat_response),
    )


def test_market_watch_missing_coordinates() -> None:
    agent = _make_market_agent()
    result = agent.run("Any nearby events?", context={})
    assert result.response == "I couldn't find enough data to answer your question."
    module_names = {s.module for s in result.steps}
    assert "market_watch_agent.signal_collection" in module_names
    assert "market_watch_agent.answer_generation" in module_names


def test_market_watch_with_coordinates_returns_full_trace() -> None:
    agent = _make_market_agent()
    result = agent.run("Any nearby events?", context={"latitude": 34.0522, "longitude": -118.2437})
    assert isinstance(result.response, str)
    module_names = [s.module for s in result.steps]
    for step in result.steps:
        assert step.module in MARKET_WATCH_MODULE_NAMES
        assert isinstance(step.prompt, dict)
        assert isinstance(step.response, dict)


def test_market_watch_autonomous_returns_outcome() -> None:
    agent = _make_market_agent()
    outcome = agent.run_autonomous(context={"latitude": 34.0522, "longitude": -118.2437})
    assert isinstance(outcome.response, str)
    assert isinstance(outcome.steps, list)
    assert isinstance(outcome.alerts, list)
    assert isinstance(outcome.inserted_count, int)


def test_analyst_agent_review_scores_returns_full_trace() -> None:
    agent = _make_analyst_agent()
    result = agent.run(
        "Compare my review scores against nearby competitors.",
        context={"property_id": "42409434", "analysis_category": "review_scores"},
    )

    assert isinstance(result.response, str)
    assert len(result.response) > 0
    module_names = [s.module for s in result.steps]
    for step in result.steps:
        assert step.module in ANALYST_MODULE_NAMES
        assert isinstance(step.prompt, dict)
        assert isinstance(step.response, dict)
    assert "analyst_agent.neighbor_lookup" in module_names
    assert "analyst_agent.answer_generation" in module_names
    first_numeric = result.steps[2].response["numeric_items"]
    assert first_numeric == 7


def test_analyst_agent_infers_review_scores_from_reviews_score_wording() -> None:
    agent = _make_analyst_agent()
    outcome = agent.analyze(
        "How are my reviews score compared to my neighbors?",
        context={"property_id": "42409434"},
    )

    data_fetch_step = next(step for step in outcome.steps if step.module == "analyst_agent.data_fetch")
    assert data_fetch_step.prompt["category"] == "review_scores"
    assert len(outcome.numeric_comparison) == 7
    assert len(outcome.categorical_comparison) == 0
    assert outcome.analysis_category == "review_scores"


def test_analyst_agent_numeric_items_include_neighbor_points_and_ties() -> None:
    agent = AnalystAgent(
        listing_store=DummyListingStore(rows=[
            {
                "id": "42409434",
                "name": "Owner",
                "review_scores_rating": 4.6,
            },
            {
                "id": "n1",
                "name": "Neighbor One",
                "review_scores_rating": 5.0,
            },
            {
                "id": "n2",
                "name": "Neighbor Two",
                "review_scores_rating": 5.0,
            },
            {
                "id": "n3",
                "name": "Neighbor Three",
                "review_scores_rating": 4.1,
            },
        ]),
        neighbor_store=DummyNeighborStore(neighbors=["n1", "n2", "n3"]),
        chat_service=DummyChatService(),
    )

    outcome = agent.analyze(
        "How are my review scores compared to nearby competitors?",
        context={"property_id": "42409434", "analysis_category": "review_scores"},
    )

    rating_item = outcome.numeric_comparison[0]
    assert rating_item.label == "Overall Rating"
    assert len(rating_item.neighbor_points) == 3
    assert [point.listing_id for point in rating_item.neighbor_max_points] == ["n1", "n2"]
    assert [point.listing_id for point in rating_item.neighbor_min_points] == ["n3"]


def test_analyst_agent_categorical_buckets_include_membership_lists() -> None:
    agent = _make_analyst_agent()
    outcome = agent.analyze(
        "How do my property specs compare to nearby competitors?",
        context={"property_id": "42409434", "analysis_category": "property_specs"},
    )

    bucket = outcome.categorical_comparison[0].buckets[0]
    assert bucket.listing_ids
    assert bucket.listing_names


def test_analyst_agent_owner_selection_fallback_uses_owner_wording() -> None:
    agent = _make_analyst_agent(chat_available=False)
    result = agent.explain_selection(
        prompt="How are my review scores compared to my neighbors?",
        property_id="42409434",
        category="review_scores",
        selection_type="numeric_point",
        metric_column="review_scores_rating",
        selection_payload={
            "metric_label": "Overall Rating",
            "point_role": "owner",
            "listing_id": "42409434",
            "listing_name": "The Burlington Hotel",
            "selected_value": 4.55,
            "owner_value": 4.55,
            "neighbor_avg": 4.35,
            "neighbor_count": 3,
            "higher_count": 1,
            "tied_count": 1,
            "lower_count": 1,
        },
    )

    assert "Your property is at 4.55 for Overall Rating" in result.response
    assert "1 are above you, 1 are tied, and 1 are below you" in result.response
    assert "selected listing ranks" not in result.response


def test_analyst_agent_without_chat_uses_fallback_summary() -> None:
    agent = _make_analyst_agent(chat_available=False)
    result = agent.run(
        "Compare my property specs against neighbors.",
        context={"property_id": "42409434", "analysis_category": "property_specs"},
    )

    assert "Competitive analysis completed" in result.response
    module_names = {s.module for s in result.steps}
    assert "analyst_agent.comparison_compute" in module_names
    assert "analyst_agent.answer_generation" in module_names


