import streamlit as st
from frontend.UtilitySetup import init_session_state, reset_session
from frontend.UploadInterface import upload_section
from frontend.QuerymodeInterface import query_mode_tab
from frontend.TutormodeInterface import tutor_mode_tab

st.set_page_config(page_title="Tutor Agent", page_icon="📚", layout="wide")
st.title("📚 Interactive Tutor Agent")

# Init state
init_session_state()

# Upload flow
if not st.session_state.docs_uploaded:
    upload_section()
else:
    tab1, tab2 = st.tabs(["💬 Query Mode", "🎓 Tutor Mode"])
    with tab1:
        query_mode_tab()
    with tab2:
        tutor_mode_tab()
