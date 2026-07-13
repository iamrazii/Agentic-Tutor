import streamlit as st
import requests

API_URL = "http://localhost:8000"

def query_mode_tab():
    st.subheader("💬 Ask Questions from Documents")
    
    if not st.session_state.get("session_id"):
        st.warning("Please upload documents first.")
        return

    query = st.chat_input("Ask a question...")
    
    if query:
        # Append user message instantly for UI responsiveness
        st.session_state.history.append((query, "Thinking..."))
        
        try:
            payload = {
                "session_id": st.session_state.session_id,
                "question": query
            }
            response = requests.post(f"{API_URL}/query", json=payload)
            
            if response.status_code == 200:
                answer = response.json()["answer"]
                # Replace the "Thinking..." placeholder with the real answer
                st.session_state.history[-1] = (query, answer)
            else:
                st.session_state.history[-1] = (query, f"Error: {response.json().get('detail')}")
                
        except Exception as e:
            st.session_state.history[-1] = (query, f"Connection error: {str(e)}")

    for q, a in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)