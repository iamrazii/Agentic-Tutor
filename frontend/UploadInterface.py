import streamlit as st
import requests

API_URL = "http://localhost:8000"

def upload_section():
    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload up to 3 documents (.pdf, .txt, .docx).",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if len(uploaded_files) > 3:
            st.error("⚠️ Maximum 3 files only.")
        else:
            if st.button("Submit Documents"):
                with st.spinner("🔄 Sending documents to backend for processing..."):
                    try:
                        # Prepare files for the requests library
                        files_data = [
                            ("files", (file.name, file.getvalue(), file.type)) 
                            for file in uploaded_files
                        ]
                        
                        response = requests.post(f"{API_URL}/upload", files=files_data)
                        
                        if response.status_code == 200:
                            data = response.json()
                            # Save the vital session ID
                            st.session_state.session_id = data["session_id"]
                            st.session_state.docs_uploaded = True
                            st.success("✅ Documents processed successfully!")
                            st.rerun()
                        else:
                            st.error(f"Server Error: {response.json().get('detail', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"Failed to connect to backend: {str(e)}")