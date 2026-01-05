from typing import TypedDict, Literal,Union
from enum import Enum
from langgraph.graph import StateGraph, START, END
from langchain.docstore.document import Document
from backend.textprocessing import Preprocessor
from langchain.prompts import PromptTemplate
from langchain_google_genai import  ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
import numpy as np
from dotenv import load_dotenv
import os,sys,json
from backend.querymode import SettingUp

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_TOKEN")

class Difficulty(Enum):
   Basic = 1
   Easy = 2
   Medium = 3
   Hard = 4


class ShortQuestion(TypedDict):
    question:str
    answer:str
    difficulty: Difficulty
    explanation:str

class MultipleChoiceQuestion(TypedDict):
    question:str
    answer:Literal["a","b","c","d"]
    options:dict[str,str]
    difficulty:Difficulty
    explanation:str

class TutorState(TypedDict):
   topic: str
   mode: str
   evaluation_style: Literal["MCQ", "SQ"]
   questions: Union[list[MultipleChoiceQuestion] , list[ShortQuestion]] 
   question_no: int
   max_points: int
   current_points:int
   total_questions: int
   difficulty: Difficulty
   retrieved_docs: list[Document]
   overview_text: str  
   branch: str  
   current_question: Union[MultipleChoiceQuestion, ShortQuestion, dict, None]
   user_answer: str
   answer_correct: bool
   feedback_text: str


user_queries = []
def init_tutor_agent(retrevier1,retirver2):
    preprocessor = Preprocessor()
    en_retriever = retrevier1
    strict_retriever = retirver2
    safe_ret = preprocessor.build_safe_retriever(en_retriever, strict_retriever)
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=1.0)
    embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return en_retriever, strict_retriever,safe_ret, llm, embedding



def calculate_similarity(question, answer):
    # Get embeddings
    embedding = st.session_state.embedding
    question_embedding = embedding.embed_query(question)
    answer_embedding = embedding.embed_query(answer)
    
    # Calculate cosine similarity
    dot_product = np.dot(question_embedding, answer_embedding)
    norm_q = np.linalg.norm(question_embedding)
    norm_a = np.linalg.norm(answer_embedding)
    
    return dot_product / (norm_q * norm_a)

def overview_node(state: TutorState) -> TutorState:  
    topic = state['topic']
    # Fetch relevant document chunks using global retriever
    en_retriever = st.session_state.en_retriever
    retrieved_docs: list[Document] = en_retriever.invoke(topic)
    docs_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # LLM prompt
    prompt_template = PromptTemplate(
        template=""" You are an expert tutor.
        Provide a detailed overview of the topic: {topic}.
        Use the following documents as reference:
            {docs_text}

        Structure the answer clearly with:
            - Definition / explanation
            - Key points / details
            - Examples if applicable """,
        input_variables=['topic','docs_text']
    )
    llm = st.session_state.llm
    prompt = prompt_template.invoke({'topic': topic , 'docs_text': docs_text})
    overview_text = llm.invoke(prompt)
    
    # Update state
    state["overview_text"] = overview_text.content if hasattr(overview_text, 'content') else str(overview_text)
    state['mode'] = "Overview"
    state['question_no'] = 0
    return state    

def topic_exists(retriever,topic:str):
   retrieved_docs = retriever.invoke(topic)
   if not retrieved_docs:
      return False,[]
   
   return True, retrieved_docs

def topic_selection(state: TutorState, user_queries: list[str]):
    topic = state.get("topic")
    retriever = st.session_state.safe_ret
    if not topic:
        # If frontend has not provided a topic yet
        state["branch"] = "awaiting_topic"
        return state

    exist, docs = topic_exists(retriever, topic)

    if not exist:
        state["branch"] = "invalid"
        return state

    # If topic exists
    user_queries.append(topic)
    state["retrieved_docs"] = docs
    state["branch"] = "valid"
    return state

def evaluation_mode(state: TutorState) -> TutorState:
    state["evaluation_style"] = state.get("evaluation_style")
    state["mode"] = "Evaluation"
    return state

def SQFormation(state:TutorState):
    state["mode"] = "SQFormation"
    topic = state["topic"]
    en_retriever = st.session_state.en_retriever
    content = en_retriever.invoke(topic)
    document = "\n\n---\n\n".join([doc.page_content for doc in content])

    prompt = PromptTemplate.from_template(
        """
        You are an exam question generator.
        Based on the following study material, generate exact 15 Questions a user can answer.Question's nature should be conceptual.

        Requirements:
        - Difficulty distribution: 10% basic, 30% easy, 40% medium, 20% hard.
        - Focus on "how", "why", or "explain the importance of" questions. 
        - Avoid simple factual recall like "list" or "define".
        - Try to keep answers generalized with simple and common explanation which aligns with how a common user would respond
         - Format: 
            "question": "...","answer": "...","difficulty": "basic|easy|medium|hard","explanation":"..."
        - For explanation in the format, try to give brief but accurate explanation of the correct answer    
        - Try to cover each chunk at least once.
        
        Study Material:
        {document}

        """
    )

    # Call LLM
    llm = st.session_state.llm
    response = llm.invoke(prompt.format(document=document))
    response.content = response.content.removeprefix("```json").removesuffix("```")
    questions = json.loads(response.content)
    # Mapping difficulty to enum
    for q in questions:
        diff_str = q["difficulty"].capitalize()  
        q["difficulty"] = Difficulty[diff_str]

    points = state["difficulty"]
    totalQs = state["total_questions"]
    sorted_questions = sorted(questions , key = lambda x: x["difficulty"].value)
    state["questions"] = sorted_questions
    state["question_no"] = 0
    state["total_questions"] = len(sorted_questions)
    state["current_points"] = 0
    state["max_points"] = (0.1*totalQs)*points.Basic.value + (0.3*totalQs)*points.Easy.value + (0.4*totalQs)*points.Medium.value + (0.2*totalQs)*points.Hard.value 
    state["mode"] = "Evaluation"
    return state

def MCQFormation(state: TutorState):
    state["mode"] = "MCQFormation"
    # Retrieve relevant chunks
    topic = state["topic"]
    en_retriever = st.session_state.en_retriever
    content = en_retriever.invoke(topic)  # en_retriever should already be configured
    document = "\n\n---\n\n".join([doc.page_content for doc in content])

    # Prompt
    prompt = PromptTemplate.from_template(
        """
        You are an exam question generator.
        Based on the following study material, generate around 15 MCQs.

        Requirements:
        - Difficulty distribution: 10% basic, 30% easy, 40% medium, 20% hard.
        - Create more conceptual questions rather than factual oriented questions which depend majorly on the provided documents
        - Each question must have 4 options (a, b, c, d). Only one correct answer.
        - Format: 
            "question": "...","options": {{"a": "...", "b": "...", "c": "...", "d": "..."}},"answer": "a|b|c|d","difficulty": "basic|easy|medium|hard","explanation":"..."
        - Try to cover each chunk at least once.
        - Add a brief but accurate explanation in the format object "explanation"
        
        Study Material:
        {document}

        """
    )

    # Call LLM
    llm = st.session_state.llm
    response = llm.invoke(prompt.format(document=document))
    response.content = response.content.removeprefix("```json").removesuffix("```")
    questions = json.loads(response.content)
    # Mapping difficulty to enum
    for q in questions:
        diff_str = q["difficulty"].capitalize()  
        q["difficulty"] = Difficulty[diff_str]

    points = state["difficulty"]
    sorted_questions = sorted(questions , key = lambda x: x["difficulty"].value)
    state["questions"] = sorted_questions
    state["question_no"] = 0
    state["total_questions"] = len(sorted_questions)
    totalQs = state["total_questions"]
    state["current_points"] = 0
    state["max_points"] = (0.1*totalQs)*points.Basic.value + (0.3*totalQs)*points.Easy.value + (0.4*totalQs)*points.Medium.value + (0.2*totalQs)*points.Hard.value  
    print(f"Max points are {state["max_points"]}")
    state["mode"] = "Evaluation"
    return state

def ConductingTestSQ(state: TutorState):
    
    state["mode"] = "SQTest"
    question = state["questions"][state["question_no"]] # Get current question
    state["current_question"] = question
    # Check if user has provided an answer
    user_answer = state.get("user_answer")
    if not user_answer:
        return state
    
    # Process the answer using your existing similarity function
    similarity = calculate_similarity(question["answer"], user_answer)
    
    # Update points if similarity is good
    if similarity > 0.50:
        state["current_points"] = state["current_points"] + question["difficulty"].value
        state["answer_correct"] = True
        state["feedback_text"] = f"✅ Correct! Good answer. You got {question["difficulty"].value} points. (Similarity: {similarity:.2f})"
    else:
        state["answer_correct"] = False
        state["feedback_text"] = f"❌ Not quite right. Expected: {question['answer']} (Similarity: {similarity:.2f})"
    
    state["question_no"] = state["question_no"] + 1 # Move to next question
    state["user_answer"] = "" # Clear user answer for next question
    
    return state

def ConductingTestMCQ(state: TutorState):
    # Get current question
    state["mode"] = "MCQTest"
    question = state["questions"][state["question_no"]]
    state["current_question"] = question
    
    # Check if user has provided an answer
    user_answer = state.get("user_answer")
    if not user_answer:
        return state
    
    # Process the answer (convert to lowercase for comparison)
    user_answer_lower = user_answer.lower()
    correct_answer = question["answer"]  # This is already "a", "b", "c", or "d"
    
    # Update points and provide feedback
    if user_answer_lower == correct_answer:
        state["current_points"] = state["current_points"] + question["difficulty"].value
        state["answer_correct"] = True
        state["feedback_text"] = f"✅ Correct answer!. You got {question["difficulty"].value} points."
    else:
        state["answer_correct"] = False
        correct_option = question["options"][correct_answer]  # Get the text of correct option
        state["feedback_text"] = f"❌ Incorrect. Correct answer is {correct_answer.upper()}) {correct_option}"
    
    state["question_no"] = state["question_no"] + 1
    state["user_answer"] = ""
    return state


def continue_or_end(state: TutorState) -> str:
    if state["question_no"] < state["total_questions"]:
        return "continue"
    return "done"

def Feedback(state:TutorState):
    percentage = (state["current_points"]/state["max_points"] )*100
   


# Graph setup
graph = StateGraph(TutorState)

# Settign up nodes

graph.add_node("select_topic", lambda state: topic_selection(state, user_queries))
graph.add_node("overview", overview_node , checkpoint =True)
graph.add_node("evaluation_mode" , evaluation_mode,checkpoint=True)
graph.add_node("MCQFormation",MCQFormation)
graph.add_node("SQFormation" , SQFormation)
graph.add_node("ConductingTestMCQ",ConductingTestMCQ)
graph.add_node("ConductingTestSQ",ConductingTestSQ)

# Settign up edges / flow

graph.add_edge(START,'select_topic')
graph.add_conditional_edges(
    "select_topic",
    lambda state: state.get("branch", "invalid"),  # branch key
    {
        "valid": "overview",
        "invalid": END,  # loop back
    },
)
graph.add_edge('overview',"evaluation_mode")
graph.add_conditional_edges(
    "evaluation_mode",
    lambda state: state.get("evaluation_style") if state.get("evaluation_style") else "wait",
    {
        "MCQ": "MCQFormation",
        "SQ": "SQFormation",
        "wait" : END
    }
)
graph.add_edge("MCQFormation", END)
graph.add_edge("SQFormation", END)

memory = MemorySaver()
workflow = graph.compile(checkpointer=memory)
# Updated initial state
initial_state = {
   "topic": "",
   "mode": "Overview", 
   "evaluation_style": "",
   "questions": [],
   "question_no": 0,
   "max_points":0,
   "current_points": 0,
   "total_questions": 0,
   "difficulty": Difficulty.Basic,
   "overview_text": "",
   "branch": "",
   "current_question": None,
   "user_answer": "",
   "answer_correct": False,
   "feedback_text": "",
   "quiz_started": False
}