from pathlib import Path

from app.architecture import ensure_architecture_svg, render_architecture_svg


def test_render_architecture_svg_contains_key_modules() -> None:
    svg = render_architecture_svg()

    assert svg.lstrip().startswith("<svg")
    assert "Airbnb Business Agent" in svg
    assert "reviews_agent" in svg
    assert "market_watch_agent" in svg
    assert "Supabase Listings" in svg
    assert 'id="arrowControl"' in svg


def test_ensure_architecture_svg_writes_file() -> None:
    output_path = Path.cwd() / "test_model_architecture.svg"
    if output_path.exists():
        output_path.unlink()

    try:
        ensure_architecture_svg(output_path)

        assert output_path.exists()
        svg = output_path.read_text(encoding="utf-8")
        assert "<svg" in svg
        assert "Gmail API" in svg
    finally:
        if output_path.exists():
            output_path.unlink()
