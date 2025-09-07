import streamlit as st
from backend.querymode import format_chat_history


def query_mode_tab():
    st.subheader("💬 Ask Questions from Documents")
    query = st.chat_input("Ask a question...")
    if query:
        response = st.session_state.query_chain.invoke({
            "question": query,
            "chat_history": format_chat_history(st.session_state.history)
        })
        st.session_state.history.append((query, response))

    for q, a in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
