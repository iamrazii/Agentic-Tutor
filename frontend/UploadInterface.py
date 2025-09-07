import streamlit as st
from backend.tutor import init_tutor_agent, initial_state
from backend.querymode import SettingUp, build_chain

def upload_section():
    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload up to 3 documents (.pdf, .txt, .docx). Note: more files will take more time to preprocess and embedd ",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if len(uploaded_files) > 3:
            st.error("⚠️ Maximum 3 files only.")
        else:
            if st.button("Submit Documents"):
                with st.spinner("🔄 Processing documents..."):
                    # Directly pass file objects instead of paths
                    file_objs = uploaded_files  

                    # Query Mode
                    chunks, en_retriever = SettingUp(file_objs)
                    print("Upload Interface: retriever created \n")
                    st.session_state.query_chain = build_chain(chunks, en_retriever)
                    print("Upload Interface: chain created \n")
                    # Tutor Mode
                    (
                        st.session_state.en_retriever,
                        st.session_state.strict_ret,
                        st.session_state.safe_ret,
                        st.session_state.llm,
                        st.session_state.embedding,
                    ) = init_tutor_agent(file_objs)

                    st.session_state.docs_uploaded = True
                    st.session_state.current_state = initial_state.copy()
                st.success("✅ Documents processed!")
                st.rerun()
