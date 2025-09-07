import streamlit as st
import time
from backend.tutor import initial_state

def init_session_state():
    if "docs_uploaded" not in st.session_state:
        st.session_state.docs_uploaded = False
    if "history" not in st.session_state:
        st.session_state.history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"tutor-session-{int(time.time())}"
    if "current_state" not in st.session_state:
        st.session_state.current_state = initial_state.copy()

def reset_session():
    st.session_state.current_state = initial_state.copy()
    st.session_state.thread_id = f"tutor-session-{int(time.time())}"
