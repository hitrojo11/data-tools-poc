# tests/test_session_adapter.py
from session_adapter import ensure_defaults, get_state, set_state, reset_state


def test_ensure_and_reset(tmp_path, monkeypatch):
    # streamlit's st.session_state is a proxy; emulate with dict for unit test
    ss = {}
    ensure_defaults(ss)
    assert "work_df" in ss and ss["history"] == []
    set_state(ss, "foo", 123)
    assert get_state(ss, "foo") == 123
    reset_state(ss)
    assert ss["history"] == []
