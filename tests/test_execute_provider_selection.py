from __future__ import annotations

from dataclasses import dataclass

from app.agents.base import AgentResult
from app.schemas import ExecuteRequest
import app.main as main_module


@dataclass
class _DummyChatService:
    is_available: bool


class _DummyAgent:
    def __init__(self, label: str) -> None:
        self._label = label

    def run(self, prompt: str, context: dict[str, object] | None = None) -> AgentResult:
        return AgentResult(response=self._label, steps=[])


def _patch_provider_state(monkeypatch, *, openrouter_available: bool = True) -> None:
    monkeypatch.setattr(main_module, "default_chat_provider", "llmod")
    monkeypatch.setattr(
        main_module,
        "chat_services_by_provider",
        {
            "llmod": _DummyChatService(is_available=True),
            "openrouter": _DummyChatService(is_available=openrouter_available),
        },
    )
    monkeypatch.setattr(
        main_module,
        "reviews_agents_by_provider",
        {
            "llmod": _DummyAgent("reviews:llmod"),
            "openrouter": _DummyAgent("reviews:openrouter"),
        },
    )


def test_execute_uses_default_provider_when_no_override(monkeypatch) -> None:
    _patch_provider_state(monkeypatch, openrouter_available=True)

    result = main_module.execute(ExecuteRequest(prompt="wifi"))

    assert result.status == "ok"
    assert result.response == "reviews:llmod"


def test_execute_uses_openrouter_when_explicit_override(monkeypatch) -> None:
    _patch_provider_state(monkeypatch, openrouter_available=True)

    result = main_module.execute(
        ExecuteRequest(prompt="wifi", llm_provider="openrouter")
    )

    assert result.status == "ok"
    assert result.response == "reviews:openrouter"


def test_execute_returns_error_when_explicit_provider_unavailable(monkeypatch) -> None:
    _patch_provider_state(monkeypatch, openrouter_available=False)

    result = main_module.execute(
        ExecuteRequest(prompt="wifi", llm_provider="openrouter")
    )

    assert result.status == "error"
    assert result.error is not None
    assert "openrouter" in result.error
