import uuid
from pydantic import BaseModel

active_sessions = {}

def create_new_session() -> str:
    """Generates a new session ID and initializes their storage locker."""
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {
        "en_retriever": None,
        "strict_retriever": None,
        "safe_ret": None,
        "query_chain": None,
        "history": [], # For Query Mode chat history
    }
    return session_id

def get_session(session_id: str) -> dict:
    """Retrieves the user's session data."""
    if session_id not in active_sessions:
        raise ValueError("Invalid or expired session_id. Please upload documents again.")
    return active_sessions[session_id]