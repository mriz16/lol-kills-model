"""Smoke test: the page renders and the slate box previews matchups (bundled data, no
GitHub secrets, nothing run or logged)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def test_app_renders_and_previews_slate():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    at.session_state["slate_text"] = "HLE vs GEN -150 +120 bo1\nZeka 3.5\nChovy 4.5"
    at.run()
    assert not at.exception
    btn = [b for b in at.button if b.key == "run_slate"][0]
    assert not btn.disabled and "1 matchup" in btn.label and "2 line" in btn.label
