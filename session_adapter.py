# session_adapter.py
from typing import Any, MutableMapping

DEFAULT_STATE = {
    "work_df": None,
    "history": [],
    "changelog": [],
    "uploaded_df": None,
    "schema_input": "{}",
    "last_schema_report": None,
}


def ensure_defaults(session_state: MutableMapping[Any, Any]) -> None:
    """
    Ensure required keys exist in session_state with safe defaults.
    Call at app startup.
    """
    for k, v in DEFAULT_STATE.items():
        if k not in session_state:
            session_state[k] = v


def get_state(
    session_state: MutableMapping[Any, Any], key: str, default: Any = None
) -> Any:
    return session_state.get(key, default)


def set_state(session_state: MutableMapping[Any, Any], key: str, value: Any) -> None:
    session_state[key] = value


def reset_state(session_state: MutableMapping[Any, Any]) -> None:
    """
    Reset the session state to the DEFAULT_STATE minimal set (non-destructive for other keys).
    Useful as a 'Reset session' button action.
    """
    for k, v in DEFAULT_STATE.items():
        session_state[k] = v
