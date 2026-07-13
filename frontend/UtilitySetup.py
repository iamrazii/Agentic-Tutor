import streamlit as st
import time

def init_session_state():
    if "docs_uploaded" not in st.session_state:
        st.session_state.docs_uploaded = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"tutor-session-{int(time.time())}"
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # We will keep a simplified local state to manage UI visibility
    if "current_state" not in st.session_state:
        st.session_state.current_state = {
            "overview_text": None,
            "evaluation_style": None,
            "questions_ready": False,
            "quiz_started": False,
            "current_question": None,
            "question_no": 0,
            "total_questions": 0,
            "current_points": 0,
            "feedback_text": "",
            "answer_correct": False,
            "quiz_complete": False
        }

def reset_session():
    st.session_state.current_state = {
        "overview_text": None,
        "evaluation_style": None,
        "questions_ready": False,
        "quiz_started": False,
        "current_question": None,
        "question_no": 0,
        "total_questions": 0,
        "current_points": 0,
        "feedback_text": "",
        "answer_correct": False,
        "quiz_complete": False
    }
    st.session_state.thread_id = f"tutor-session-{int(time.time())}"