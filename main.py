import streamlit as st
from frontend.UtilitySetup import init_session_state
from frontend.UploadInterface import upload_section
from frontend.QuerymodeInterface import query_mode_tab
from frontend.TutormodeInterface import tutor_mode_tab

st.set_page_config(page_title="AI Document Tutor", layout="wide")

# Initialize our state tracking
init_session_state()

st.title("🧠 Active AI Document Tutor")

# Flow Logic: If documents aren't uploaded, ONLY show the upload screen
if not st.session_state.docs_uploaded:
    upload_section()

# If uploaded, hide upload screen and show the main tabs
else:
    # Optional: A button to restart completely
    if st.sidebar.button("Upload Different Documents"):
        st.session_state.clear()
        st.rerun()

    tab1, tab2 = st.tabs(["💬 Query Mode", "🎓 Tutor Mode"])
    
    with tab1:
        query_mode_tab()
        
    with tab2:
        tutor_mode_tab()