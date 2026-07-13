import streamlit as st
import requests
from frontend.UtilitySetup import reset_session

API_URL = "http://localhost:8000"

def tutor_mode_tab():
    st.subheader("🎓 Tutor Mode")
    
    if not st.session_state.get("session_id"):
        st.warning("Please upload documents first.")
        return

    state = st.session_state.current_state

    # ==========================================
    # SCREEN 1: Ask for Topic
    # ==========================================
    if not state.get("overview_text"):
        topic_input = st.text_input("Enter a topic you want to learn about:")
        if st.button("Submit Topic") and topic_input.strip():
            with st.spinner("🔄 Scanning documents and generating overview..."):
                try:
                    # 1. Initialize session on backend
                    requests.post(f"{API_URL}/tutor/start", json={
                        "session_id": st.session_state.session_id,
                        "thread_id": st.session_state.thread_id
                    })
                    
                    # 2. Set topic
                    res = requests.post(f"{API_URL}/tutor/topic", json={
                        "session_id": st.session_state.session_id,
                        "thread_id": st.session_state.thread_id,
                        "topic": topic_input
                    })
                    
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("status") == "error":
                            st.error("⚠️ Topic not found in documents. Try another.")
                        else:
                            state["overview_text"] = data["overview"]
                            st.rerun() # Rerunning hides this input and shows Screen 2
                    else:
                        st.error(f"Backend Error: {res.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to connect to server: {str(e)}")

    # ==========================================
    # SCREEN 2: Overview & Evaluation Style
    # ==========================================
    elif state.get("overview_text") and not state.get("evaluation_style"):
        st.write("### 📘 Topic Overview")
        st.markdown(state["overview_text"])

        st.write("### How would you like to be evaluated?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Multiple Choice (MCQ)", use_container_width=True):
                _generate_questions("MCQ", state)
        with col2:
            if st.button("✏️ Short Questions (SQ)", use_container_width=True):
                _generate_questions("SQ", state)

    # ==========================================
    # SCREEN 3: Start Quiz Gateway
    # ==========================================
    elif state.get("questions_ready") and not state.get("quiz_started"):
        st.success(f"✅ Generated {state['total_questions']} questions successfully!")
        st.write("### 🎯 Ready to Begin?")
        st.write(f"**Mode:** {state['evaluation_style']}")
        
        if st.button("🚀 Start Test Now", type="primary"):
            state["quiz_started"] = True
            st.rerun() # Hides gateway, shows Question 1

    # ==========================================
    # SCREEN 4: The Testing Page (Question by Question)
    # ==========================================
    elif state.get("quiz_started") and not state.get("quiz_complete"):
        q_no = state["question_no"]
        total_qs = state["total_questions"]
        question = state.get("current_question")

        if question:
            # Progress bar
            st.progress((q_no) / total_qs, text=f"Question {q_no + 1} of {total_qs}")
            st.write(f"### Question {q_no + 1}")
            st.markdown(question["question"])

            # --- FEEDBACK VIEW ---
            # If they just submitted, show feedback and PAUSE until they click Next
            if state.get("feedback_text"):
                if state.get("answer_correct"):
                    st.success(state["feedback_text"])
                else:
                    st.error(state["feedback_text"])
                
                st.info(f"**Current Score:** {state.get('current_points', 0)} points")
                
                # Force user to acknowledge feedback before moving on
                if st.button("Next Question", type="primary"):
                    # Apply the buffered next question
                    state["feedback_text"] = ""
                    state["answer_correct"] = False
                    state["question_no"] += 1
                    
                    if state.get("is_complete_buffer"):
                        state["quiz_complete"] = True
                    else:
                        state["current_question"] = state.get("next_question_buffer")
                    st.rerun()
            
            # --- ANSWER SUBMISSION VIEW ---
            # If no feedback, it means they need to answer the question
            else:
                if state["evaluation_style"] == "MCQ":
                    options = question["options"]
                    st.write("**Select your answer:**")
                    
                    for key, value in options.items():
                        # Using full width buttons for clean UI
                        if st.button(f"{key.upper()}) {value}", key=f"opt_{key}_{q_no}", use_container_width=True):
                            _submit_answer_to_api(key, state)
                else:
                    user_answer = st.text_area("Type your answer here:", key=f"sq_{q_no}", height=150)
                    if st.button("Submit Answer", type="primary"):
                        if user_answer.strip():
                            _submit_answer_to_api(user_answer, state)
                        else:
                            st.warning("Please enter an answer before submitting.")

    # ==========================================
    # SCREEN 5: Results Page
    # ==========================================
# ==========================================
    # SCREEN 5: Results Page
    # ==========================================
    elif state.get("quiz_complete"):
        st.write("### 🎉 Test Completed!")
        st.balloons()
        
        # Calculate the math
        curr_pts = state.get("current_points", 0)
        max_pts = state.get("max_points", 1) # Default to 1 to prevent division by zero errors
        percentage = (curr_pts / max_pts) * 100
        
        # Display side-by-side metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Final Score", value=f"{curr_pts} / {max_pts}")
        with col2:
            # Color code the percentage!
            if percentage >= 80:
                st.metric(label="Accuracy", value=f"{percentage:.1f}%", delta="Great Job!")
            elif percentage >= 50:
                st.metric(label="Accuracy", value=f"{percentage:.1f}%", delta="Good Effort", delta_color="off")
            else:
                st.metric(label="Accuracy", value=f"{percentage:.1f}%", delta="Needs Review", delta_color="inverse")
        
        st.info("Because this is a continuous test, you cannot go back to previous questions. To learn more, you can query your documents in the Query tab, or start a new test!")
        
        if st.button("Start New Topic"):
            reset_session() 
            st.rerun()
    else:
        st.write("wdgye")

# --- Helper Functions ---

def _generate_questions(style: str, state: dict):
    """Helper to call style API and update state."""
    with st.spinner(f"Generating {style} questions..."):
        try:
            res = requests.post(f"{API_URL}/tutor/set_evaluation_style", json={
                "session_id": st.session_state.session_id,
                "thread_id": st.session_state.thread_id,
                "evaluation_style": style
            })
            if res.status_code == 200:
                data = res.json()
                state["evaluation_style"] = style
                state["questions_ready"] = True
                state["total_questions"] = data["total_questions"]
                state["current_question"] = data["first_question"]
                st.rerun()
            else:
                st.error(f"Failed to generate questions: {res.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")

def _submit_answer_to_api(answer_text: str, state: dict):
    """Helper to grade answer and buffer the next question."""
    with st.spinner("Grading..."):
        try:
            res = requests.post(f"{API_URL}/tutor/submit_answer", json={
                "session_id": st.session_state.session_id,
                "thread_id": st.session_state.thread_id,
                "user_answer": answer_text
            })
            
            if res.status_code == 200:
                data = res.json()
                state["feedback_text"] = data["feedback"]
                state["current_points"] = data["current_points"]
                state["max_points"] = data.get("max_points", 1)
                state["answer_correct"] = data["is_correct"]
                
                # CRITICAL: Buffer the next question so the UI doesn't swap text behind the feedback
                state["next_question_buffer"] = data["next_question"]
                state["is_complete_buffer"] = data["is_complete"]
                st.rerun()
            else:
                st.error(f"Failed to submit answer: {res.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")