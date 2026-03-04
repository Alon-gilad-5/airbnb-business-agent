"""Utilities to build the architecture diagram as a crisp SVG asset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class BoxStyle:
    fill: str
    stroke: str
    accent: str
    title_fill: str = "#0f172a"
    body_fill: str = "#475569"


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    title: str
    lines: tuple[str, ...]
    style: BoxStyle

    @property
    def left(self) -> tuple[int, int]:
        return (self.x, self.y + self.h // 2)

    @property
    def right(self) -> tuple[int, int]:
        return (self.x + self.w, self.y + self.h // 2)

    @property
    def top(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y)

    @property
    def bottom(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h)

    @property
    def left_upper(self) -> tuple[int, int]:
        return (self.x, self.y + self.h // 3)

    @property
    def left_lower(self) -> tuple[int, int]:
        return (self.x, self.y + (2 * self.h) // 3)

    @property
    def right_upper(self) -> tuple[int, int]:
        return (self.x + self.w, self.y + self.h // 3)

    @property
    def right_lower(self) -> tuple[int, int]:
        return (self.x + self.w, self.y + (2 * self.h) // 3)


SVG_WIDTH = 1880
SVG_HEIGHT = 1160

API_STYLE = BoxStyle(fill="#fff7ed", stroke="#fb923c", accent="#f59e0b", title_fill="#7c2d12")
AGENT_STYLE = BoxStyle(fill="#eff6ff", stroke="#60a5fa", accent="#2563eb")
INFRA_STYLE = BoxStyle(fill="#ffffff", stroke="#cbd5e1", accent="#94a3b8")
SCHED_STYLE = BoxStyle(fill="#ecfdf5", stroke="#4ade80", accent="#16a34a", title_fill="#14532d")
INTEGRATION_STYLE = BoxStyle(fill="#f0fdfa", stroke="#5eead4", accent="#0f766e", title_fill="#134e4a")
DATA_STYLE = BoxStyle(fill="#f8fafc", stroke="#cbd5e1", accent="#475569")


def _polyline(points: list[tuple[int, int]]) -> str:
    return " ".join(f"{x},{y}" for x, y in points)


def _text_block(*, x: int, y: int, title: str, lines: tuple[str, ...], style: BoxStyle) -> str:
    title_svg = (
        f'<text x="{x}" y="{y}" font-size="22" font-weight="700" fill="{style.title_fill}">'
        f"{escape(title)}</text>"
    )
    body_lines = []
    for idx, line in enumerate(lines):
        body_lines.append(
            f'<tspan x="{x}" dy="{0 if idx == 0 else 23}">{escape(line)}</tspan>'
        )
    body_svg = (
        f'<text x="{x}" y="{y + 34}" font-size="17" font-weight="500" fill="{style.body_fill}">'
        + "".join(body_lines)
        + "</text>"
    )
    return title_svg + body_svg


def _box_svg(box: Box) -> str:
    return f"""
    <g>
      <rect x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}" rx="24"
            fill="{box.style.fill}" stroke="{box.style.stroke}" stroke-width="2"
            filter="url(#softShadow)"/>
      <rect x="{box.x}" y="{box.y}" width="{box.w}" height="12" rx="24"
            fill="{box.style.accent}" opacity="0.95"/>
      <rect x="{box.x}" y="{box.y + 12}" width="{box.w}" height="18"
            fill="{box.style.accent}" opacity="0.08"/>
      {_text_block(x=box.x + 22, y=box.y + 46, title=box.title, lines=box.lines, style=box.style)}
    </g>
    """


def _panel(*, x: int, y: int, w: int, h: int, label: str, subtitle: str, accent: str) -> str:
    return f"""
    <g>
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="32"
            fill="#ffffff" fill-opacity="0.62" stroke="#dbe4ee" stroke-width="1.5"/>
      <rect x="{x + 24}" y="{y + 24}" width="10" height="48" rx="5" fill="{accent}"/>
      <text x="{x + 50}" y="{y + 46}" font-size="15" font-weight="800" letter-spacing="2"
            fill="#334155">{escape(label)}</text>
      <text x="{x + 50}" y="{y + 72}" font-size="16" font-weight="500" fill="#64748b">
        {escape(subtitle)}
      </text>
    </g>
    """


def _route(points: list[tuple[int, int]], *, stroke: str, marker_id: str, width: int = 3) -> str:
    return (
        f'<polyline points="{_polyline(points)}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" '
        f'marker-end="url(#{marker_id})"/>'
    )


def render_architecture_svg() -> str:
    """Return the architecture diagram SVG markup."""

    left_panel = (40, 170, 420, 930)
    center_panel = (510, 170, 440, 930)
    right_panel = (1000, 170, 840, 930)

    inp = Box(
        x=90,
        y=250,
        w=320,
        h=110,
        title="Input API",
        lines=("dedicated endpoints per agent", "prompt + context"),
        style=API_STYLE,
    )
    out = Box(
        x=1270,
        y=250,
        w=430,
        h=110,
        title="Output API",
        lines=("status / response / steps", "JSON payload"),
        style=API_STYLE,
    )

    reviews = Box(
        x=560,
        y=410,
        w=340,
        h=110,
        title="reviews_agent",
        lines=("vector retrieval + web fallback", "answer synthesis + guardrails"),
        style=AGENT_STYLE,
    )
    market = Box(
        x=560,
        y=550,
        w=340,
        h=110,
        title="market_watch_agent",
        lines=("weather / events / holidays", "signal scoring + alert generation"),
        style=AGENT_STYLE,
    )
    analyst = Box(
        x=560,
        y=690,
        w=340,
        h=110,
        title="analyst_agent",
        lines=("neighbor benchmarking", "structured listing comparisons"),
        style=AGENT_STYLE,
    )
    pricing = Box(
        x=560,
        y=830,
        w=340,
        h=110,
        title="pricing_agent",
        lines=("comp-based pricing", "market signals + review volume"),
        style=AGENT_STYLE,
    )
    mail = Box(
        x=560,
        y=970,
        w=340,
        h=110,
        title="mail_agent",
        lines=("inbox classify + draft + send", "Gmail push webhooks + HITL"),
        style=AGENT_STYLE,
    )

    pinecone = Box(
        x=90,
        y=430,
        w=320,
        h=110,
        title="Pinecone",
        lines=("review embeddings", "+ web quarantine namespace"),
        style=DATA_STYLE,
    )
    llm = Box(
        x=90,
        y=650,
        w=320,
        h=110,
        title="LLM Gateway",
        lines=("Azure OpenAI compatible", "embeddings + chat completions"),
        style=INFRA_STYLE,
    )
    scheduler = Box(
        x=90,
        y=930,
        w=320,
        h=110,
        title="Scheduler",
        lines=("internal thread / Vercel cron", "autonomous market runs"),
        style=SCHED_STYLE,
    )

    market_apis = Box(
        x=1270,
        y=550,
        w=430,
        h=110,
        title="External Market APIs",
        lines=("Open-Meteo + Ticketmaster", "Nager.Date holidays"),
        style=INTEGRATION_STYLE,
    )
    alerts = Box(
        x=1270,
        y=690,
        w=430,
        h=110,
        title="Alerts Inbox",
        lines=("SQLite / Postgres", "GET /api/market_watch/alerts"),
        style=DATA_STYLE,
    )
    supabase = Box(
        x=1270,
        y=830,
        w=430,
        h=110,
        title="Supabase Listings",
        lines=("large_dataset_table", "benchmark + pricing data"),
        style=DATA_STYLE,
    )
    gmail = Box(
        x=1270,
        y=970,
        w=430,
        h=110,
        title="Gmail API",
        lines=("OAuth2 inbox fetch + send", "push notifications"),
        style=INTEGRATION_STYLE,
    )

    agents = (reviews, market, analyst, pricing, mail)
    control_routes = []
    for agent in agents:
        mid_y = agent.top[1] - 16
        control_routes.append(
            _route(
                [inp.right, (inp.right[0] + 30, inp.right[1]), (inp.right[0] + 30, mid_y), (agent.top[0], mid_y), agent.top],
                stroke="#2563eb",
                marker_id="arrowControl",
            )
        )
        control_routes.append(
            _route(
                [agent.right, (out.left[0] - 30, agent.right[1]), (out.left[0] - 30, out.left[1]), out.left],
                stroke="#2563eb",
                marker_id="arrowControl",
            )
        )

    data_routes = [
        _route([reviews.left_upper, pinecone.right_upper], stroke="#64748b", marker_id="arrowData"),
        _route(
            [reviews.left_lower, (460, reviews.left_lower[1]), (460, llm.right_upper[1]), llm.right_upper],
            stroke="#64748b",
            marker_id="arrowData",
        ),
        _route(
            [pricing.left, (455, pricing.left[1]), (455, llm.right_lower[1]), llm.right_lower],
            stroke="#64748b",
            marker_id="arrowData",
        ),
        _route(
            [mail.left, (440, mail.left[1]), (440, llm.right[1] + 18), (llm.right[0], llm.right[1] + 18)],
            stroke="#64748b",
            marker_id="arrowData",
        ),
        _route([market.right, market_apis.left], stroke="#64748b", marker_id="arrowData"),
        _route(
            [market.right_lower, (950, market.right_lower[1]), (950, alerts.left_upper[1]), alerts.left_upper],
            stroke="#64748b",
            marker_id="arrowData",
        ),
        _route(
            [analyst.right, (980, analyst.right[1]), (980, supabase.left_upper[1]), supabase.left_upper],
            stroke="#64748b",
            marker_id="arrowData",
        ),
        _route([pricing.right, supabase.left], stroke="#64748b", marker_id="arrowData"),
        _route(
            [pricing.right_upper, (1010, pricing.right_upper[1]), (1010, market_apis.left_lower[1]), market_apis.left_lower],
            stroke="#64748b",
            marker_id="arrowData",
        ),
        _route([mail.right, gmail.left], stroke="#64748b", marker_id="arrowData"),
    ]

    scheduler_route = _route(
        [scheduler.top, (scheduler.top[0], market.left_lower[1]), market.left_lower],
        stroke="#16a34a",
        marker_id="arrowSchedule",
    )

    boxes_svg = "".join(_box_svg(box) for box in (
        inp,
        out,
        reviews,
        market,
        analyst,
        pricing,
        mail,
        pinecone,
        llm,
        scheduler,
        market_apis,
        alerts,
        supabase,
        gmail,
    ))

    panels_svg = (
        _panel(
            x=left_panel[0],
            y=left_panel[1],
            w=left_panel[2],
            h=left_panel[3],
            label="INTERFACES & PLATFORM",
            subtitle="HTTP entrypoints, shared infrastructure, and automation",
            accent="#f59e0b",
        )
        + _panel(
            x=center_panel[0],
            y=center_panel[1],
            w=center_panel[2],
            h=center_panel[3],
            label="AGENT ORCHESTRATION",
            subtitle="Domain agents with dedicated endpoints",
            accent="#2563eb",
        )
        + _panel(
            x=right_panel[0],
            y=right_panel[1],
            w=right_panel[2],
            h=right_panel[3],
            label="DATA & INTEGRATIONS",
            subtitle="External providers, persistence, and outbound communication",
            accent="#0f766e",
        )
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}"
viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">Airbnb Business Agent system architecture</title>
  <desc id="desc">A multi-agent architecture with review, market watch, analyst, pricing, and mail agents connected to platform services and external integrations.</desc>
  <defs>
    <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#f8fbff"/>
      <stop offset="55%" stop-color="#f7f7fb"/>
      <stop offset="100%" stop-color="#eef6ff"/>
    </linearGradient>
    <linearGradient id="haloGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#dbeafe" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#dbeafe" stop-opacity="0"/>
    </linearGradient>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="14" stdDeviation="18" flood-color="#94a3b8" flood-opacity="0.18"/>
    </filter>
    <marker id="arrowControl" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
    <marker id="arrowData" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>
    </marker>
    <marker id="arrowSchedule" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#16a34a"/>
    </marker>
    <pattern id="dotGrid" width="28" height="28" patternUnits="userSpaceOnUse">
      <circle cx="2" cy="2" r="1.3" fill="#cbd5e1" opacity="0.28"/>
    </pattern>
  </defs>

  <rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="url(#bgGradient)"/>
  <rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="url(#dotGrid)"/>
  <ellipse cx="1280" cy="170" rx="480" ry="170" fill="url(#haloGradient)" opacity="0.8"/>

  <g>
    <text x="60" y="72" font-size="36" font-weight="800" fill="#0f172a">Airbnb Business Agent</text>
    <text x="60" y="110" font-size="20" font-weight="500" fill="#475569">
      Multi-agent platform for reviews, market intelligence, analysis, pricing, and inbox operations
    </text>
  </g>

  <g transform="translate(1315 46)">
    <rect x="0" y="0" width="500" height="92" rx="24" fill="#ffffff" fill-opacity="0.72" stroke="#dbe4ee"/>
    <text x="28" y="30" font-size="14" font-weight="800" letter-spacing="2" fill="#334155">FLOW LEGEND</text>
    <line x1="28" y1="56" x2="88" y2="56" stroke="#2563eb" stroke-width="4" stroke-linecap="round" marker-end="url(#arrowControl)"/>
    <text x="102" y="61" font-size="16" font-weight="500" fill="#475569">request routing</text>
    <line x1="218" y1="56" x2="278" y2="56" stroke="#64748b" stroke-width="4" stroke-linecap="round" marker-end="url(#arrowData)"/>
    <text x="292" y="61" font-size="16" font-weight="500" fill="#475569">data dependency</text>
    <line x1="28" y1="78" x2="88" y2="78" stroke="#16a34a" stroke-width="4" stroke-linecap="round" marker-end="url(#arrowSchedule)"/>
    <text x="102" y="83" font-size="16" font-weight="500" fill="#475569">autonomous trigger</text>
  </g>

  {panels_svg}

  <g opacity="0.98">
    {"".join(control_routes)}
    {"".join(data_routes)}
    {scheduler_route}
  </g>

  {boxes_svg}
</svg>
"""


def ensure_architecture_svg(output_path: Path) -> None:
    """Write the current architecture diagram to an SVG file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_architecture_svg(), encoding="utf-8")
