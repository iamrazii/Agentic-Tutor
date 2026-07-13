from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from backend.models import QueryRequest, StartTutorRequest, TopicRequest, EvaluationStyleRequest, AnswerRequest,StartSessionRequest
from backend.state_manager import active_sessions, create_new_session, get_session
from backend.tutor_core import init_tutor_agent
from backend.querymode import SettingUp, build_chain, format_chat_history
from backend.graph import workflow 


app = FastAPI(title="Tutor AI API")



@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Handles file uploads, builds retrievers, and provisions a session."""
    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 files allowed.")

    session_id = create_new_session()
    
    try:

            #  TODO CHECK THIS SHIT
        class StreamlitMockFile:
            def __init__(self, name, content):
                self.name = name
                self._content = content
            
            def read(self):
                return self._content
                
            def getvalue(self):
                return self._content

        # Read the contents and wrap them
        mock_files = []
        for f in files:
            content = await f.read()
            # FastAPI uses .filename, we map it to .name for SettingUp
            mock_files.append(StreamlitMockFile(name=f.filename, content=content))

        # Note: You may need to save UploadFile to temp directories if SettingUp expects file paths.
        chunks, en_retriever, strict_retriever = SettingUp(mock_files)
        query_chain = build_chain(chunks, en_retriever)
        
        # Build the safe retriever
        _, _, safe_ret, _, _ = init_tutor_agent(en_retriever, strict_retriever)
        
        # Save to user's locker
        session_data = active_sessions[session_id]
        session_data["en_retriever"] = en_retriever
        session_data["strict_retriever"] = strict_retriever
        session_data["safe_ret"] = safe_ret
        session_data["query_chain"] = query_chain
        
        return {"message": "Documents processed!", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(req: QueryRequest):
    """Standard document Q&A."""
    try:
        session_data = get_session(req.session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    chain = session_data["query_chain"]
    history = session_data["history"]

    response = chain.invoke({
        "question": req.question,
        "chat_history": format_chat_history(history)
    })
    
    history.append((req.question, response))
    return {"answer": response, "chat_history": history}

# --- TUTOR ENDPOINTS ---

def get_graph_config(req) -> dict:
    """Helper to inject dependencies into LangGraph based on the session."""
    session_data = get_session(req.session_id)
    return {
        "configurable": {
            "thread_id": req.thread_id,
            "en_retriever": session_data["en_retriever"],
            "safe_ret": session_data["safe_ret"]
        }
    }

@app.post("/tutor/start")
async def start_session(req: StartSessionRequest):

    config = get_graph_config(req)
    
    initial_input = {
        "topic": "",
        "mode": "Overview",
        "evaluation_style": "",
        "quiz_started": False,
        "questions": [],
        "question_no": 0,
        "max_points": 0,
        "current_points": 0
    }
    
    result = workflow.invoke(initial_input, config)
    return {"message": "Session started. Please provide a topic."}


@app.post("/tutor/topic")
async def set_topic(req: TopicRequest):
    config = get_graph_config(req)
    
    current_state = workflow.get_state(config)
    if not current_state:
        raise HTTPException(status_code=404, detail="Session not found. Call /start first.")

    result = workflow.invoke({"topic": req.topic}, config)

    print(result["overview_text"])
    
    if result.get("branch") == "invalid":
        return {"message": "Topic not found in knowledge base."}
        
    return {
        "message": "Topic set and overview generated.",
        "overview": result.get("overview_text")
    }


@app.post("/tutor/set_evaluation_style")
async def set_evaluation_style(req: EvaluationStyleRequest):
    config = get_graph_config(req)
    
    result = workflow.invoke({"evaluation_style": req.evaluation_style}, config, as_node="evaluation_mode")
    
    return {
        "message": f"{req.evaluation_style} questions generated.",
        "total_questions": result.get("total_questions"),
        "first_question": result.get("questions")[0] if result.get("questions") else None
    }


@app.post("/tutor/submit_answer")
async def submit_answer(req: AnswerRequest):
    config = get_graph_config(req)
    
    state = workflow.get_state(config).values
    if not state.get("questions"):
        raise HTTPException(status_code=400, detail="No questions generated yet.")
    
    formation_node = "MCQFormation" if state.get("evaluation_style") == "MCQ" else "SQFormation"

    workflow.update_state(config, {"user_answer": req.user_answer}, as_node=formation_node)


    # In a fully connected graph, you'd invoke to let it naturally route to ConductTest nodes.
    # Since they aren't connected yet, you can force the state update for now or connect them in tutor.py.
    result = workflow.invoke(None, config) 
    q_no = result.get("question_no", 0)
    total = result.get("total_questions", 1)
    
    return {
        "feedback": result.get("feedback_text", "No feedback available."),
        "current_points": result.get("current_points", 0),
        "is_correct": result.get("answer_correct", False),
        "max_points": result.get("max_points", 1),
        "next_question": result.get("questions")[q_no] if q_no < total else None,
        "is_complete": q_no >= total
    }

@app.get("/state/{thread_id}")
async def get_current_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state_data = workflow.get_state(config)
    
    if not state_data:
        raise HTTPException(status_code=404, detail="State not found")
        
    return {"current_state": state_data.values}