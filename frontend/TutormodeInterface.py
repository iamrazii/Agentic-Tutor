import streamlit as st
from backend.tutor import workflow, initial_state, calculate_similarity
import time
from frontend.UtilitySetup import reset_session

def tutor_mode_tab():
    st.subheader("🎓 Tutor Mode")
    # paste your Step 1–5 logic here as-is (with minor refactoring)

        # Step 1: Enter Topic
    if not st.session_state.current_state.get("overview_text"):
        topic_input = st.text_input("Enter a topic:")
        if st.button("Submit Topic") and topic_input.strip():
            with st.spinner("🔄 Validating topic and generating overview..."):
                try:
                    in_state = st.session_state.current_state.copy()
                    in_state["topic"] = topic_input

                    result = workflow.invoke(
                        in_state,
                        retrievers={"strict_retriever": st.session_state.strict_ret},
                        config={
                            "configurable": {
                                "thread_id": st.session_state.thread_id,
                                "until": ["evaluation_mode"]
                            }
                        },
                    )
                    st.session_state.current_state = result
                    if result.get("branch") == "invalid":
                        st.error("⚠️ Topic not found in documents. Try another.")
                        st.session_state.current_state = initial_state.copy()
                    else:
                        st.success("✅ Topic validated! Overview generated.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error validating topic: {str(e)}")
                    st.session_state.current_state = initial_state.copy()

    # Step 2: Show Overview and Choose Mode
    if st.session_state.current_state.get("overview_text") and not st.session_state.current_state.get("evaluation_style"):
        st.write("### 📘 Overview")
        st.markdown(st.session_state.current_state["overview_text"])

        st.write("### Choose Evaluation Mode")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 MCQ"):
                st.session_state.current_state["evaluation_style"] = "MCQ"
                st.rerun()
        with col2:
            if st.button("✏️ Short Questions"):
                st.session_state.current_state["evaluation_style"] = "SQ"
                st.rerun()

    # Step 3: Generate Questions
    if st.session_state.current_state.get("evaluation_style") and not st.session_state.current_state.get("questions"):
        st.write(f"### Selected Mode: {st.session_state.current_state['evaluation_style']}")
        if st.button("Generate Questions"):
            with st.spinner("🔄 Generating questions..."):
                try:
                    result = workflow.invoke(
                        st.session_state.current_state,
                        retrievers={"en_retriever": st.session_state.en_retriever},
                        config={"configurable": {"thread_id": st.session_state.thread_id}},
                    )
                    st.session_state.current_state = result
                    if result.get("questions"):
                        st.success(f"✅ Generated {len(result['questions'])} questions!")
                        st.rerun()
                    else:
                        st.warning("No questions generated.")
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")

    # Step 4: Start Quiz Button
    if (st.session_state.current_state.get("questions") and 
        not st.session_state.current_state.get("quiz_started") and
        st.session_state.current_state.get("mode") != "Quiz"):
        
        st.write("### 🎯 Ready to Start Quiz!")
        st.write(f"**Questions prepared:** {len(st.session_state.current_state['questions'])}")
        st.write(f"**Mode:** {st.session_state.current_state['evaluation_style']}")
        
        if st.button("🚀 Start Quiz"):
            # Initialize quiz state
            st.session_state.current_state["quiz_started"] = True
            st.session_state.current_state["question_no"] = 0
            st.session_state.current_state["current_points"] = 0
            st.session_state.current_state["mode"] = "Quiz"
            st.session_state.current_state["current_question"] = None
            st.session_state.current_state["feedback_text"] = ""
            st.rerun()
# Step 5: Quiz Interface
    if st.session_state.current_state.get("quiz_started") and st.session_state.current_state.get("questions"):
        state = st.session_state.current_state
        q_no = state["question_no"]
        total_qs = len(state["questions"])

        if q_no < total_qs:
            # Load current question directly from list
            if not state.get("current_question"):
                current_question = state["questions"][q_no]
                st.session_state.current_state["current_question"] = current_question
                st.rerun()

            # Display current question
            if state.get("current_question"):
                question = state["current_question"]
                
                # Progress bar
                progress = q_no / total_qs
                st.progress(progress, text=f"Question {q_no + 1} of {total_qs}")
                
                st.write(f"### Question {q_no + 1}")
                st.markdown(question["question"])

                # Show feedback if available
                if state.get("feedback_text"):
                    if state.get("answer_correct"):
                        st.success(state["feedback_text"])
                    else:
                        st.error(state["feedback_text"])
                    
                    st.info(f"**Current Score:** {state['current_points']} points")
                    
                    if st.button("Next Question", key=f"next_{q_no}"):
                        # Clear feedback and move to next question
                        st.session_state.current_state["feedback_text"] = ""
                        st.session_state.current_state["current_question"] = None
                        st.session_state.current_state["answer_correct"] = False
                        st.rerun()
                else:
                    # Collect answer based on question type
                    if state["evaluation_style"] == "MCQ":
                        # MCQ: Display 4 option buttons
                        options = question["options"]
                        correct_answer = question["answer"]
                        
                        st.write("**Select your answer:**")
                        
                        # Create columns for better layout
                        col1, col2 = st.columns(2)
                        
                        for idx, (key, value) in enumerate(options.items()):
                            col = col1 if idx % 2 == 0 else col2
                            
                            with col:
                                if st.button(f"{key.upper()}) {value}", key=f"option_{key}_{q_no}"):
                                    # Check if answer is correct
                                    if key == correct_answer:
                                        st.session_state.current_state["answer_correct"] = True
                                        st.session_state.current_state["feedback_text"] = "Correct answer!" + st.session_state.current_state["current_question"]["explanation"]
                                        st.session_state.current_state["current_points"] += question["difficulty"].value
                                    else:
                                        st.session_state.current_state["answer_correct"] = False
                                        correct_option = options[correct_answer]
                                        st.session_state.current_state["feedback_text"] = f"Incorrect. Correct answer is {correct_answer.upper()}) {correct_option}"
                                    
                                    # Move to next question
                                    st.session_state.current_state["question_no"] += 1
                                    st.rerun()
                    
                    else:  # Short Questions
                        # SQ: Display text input
                        user_answer = st.text_area("Your answer:", key=f"sq_answer_{q_no}", height=100)
                        
                        if st.button("Submit Answer", key=f"submit_sq_{q_no}"):
                            if user_answer.strip():
                                # Evaluate using similarity
                                similarity = calculate_similarity(question["answer"], user_answer)
                                
                                if similarity >= 0.40 and similarity <= 0.65:
                                    st.session_state.current_state["answer_correct"] = True
                                    st.session_state.current_state["feedback_text"] = f"Partial Correct! You can do better. \n {question["explanation"]})"
                                    st.session_state.current_state["current_points"] += question["difficulty"].value/2
                                elif similarity > 0.65:
                                    st.session_state.current_state["answer_correct"] = True
                                    st.session_state.current_state["feedback_text"] = f"Correct! Good Answer. \n {question["explanation"]})"
                                    st.session_state.current_state["current_points"] += question["difficulty"].value

                                else:
                                    st.session_state.current_state["answer_correct"] = False
                                    st.session_state.current_state["feedback_text"] = f"Not quite right. Expected: {question['answer']})"
                                
                                # Move to next question
                                st.session_state.current_state["question_no"] += 1
                                st.rerun()
                            else:
                                st.warning("Please enter an answer before submitting.")

        else:
            # Quiz completed
            st.write("### Quiz Completed!")
            
            max_points = st.session_state.current_state["max_points"]
            current_points = state.get("current_points")
            
            st.write(f"**Final Score:** {current_points}/{max_points} points")
            
            if max_points > 0:
                percentage = (current_points / max_points) * 100
                st.write(f"**Percentage:** {percentage:.1f}%")
                
                if percentage >= 80:
                    st.success("Excellent work! You've mastered this topic!")
                elif percentage >= 60:
                    st.info("Good job! You have a solid understanding.")
                else:
                    st.warning("Keep studying! Review the material and try again.")
            
            if st.button("Start Over"):
                st.session_state.current_state = initial_state.copy()
                st.session_state.thread_id = f"tutor-session-{int(time.time())}"
                st.rerun()
