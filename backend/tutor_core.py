
from backend.models import TutorState,Difficulty
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_google_genai import  ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.config import RunnableConfig
# import streamlit as st
import numpy as np
from dotenv import load_dotenv
import os,sys,json
from backend.textprocessing import Preprocessor

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_TOKEN")

GLOBAL_LLM = ChatGoogleGenerativeAI(model='gemini-3-flash-preview', temperature=1.0)
GLOBAL_EMBEDDING = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

user_queries =[]

def init_tutor_agent(retrevier1,retirver2):
    preprocessor = Preprocessor()
    en_retriever = retrevier1
    strict_retriever = retirver2
    safe_ret = preprocessor.build_safe_retriever(en_retriever, strict_retriever)
    llm = ChatGoogleGenerativeAI(model='gemini-3-flash-preview', temperature=1.0)
    embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return en_retriever, strict_retriever,safe_ret, llm, embedding


def calculate_similarity(question, answer):
    # Get embeddings
    embedding = GLOBAL_EMBEDDING
    question_embedding = embedding.embed_query(question)
    answer_embedding = embedding.embed_query(answer)
    
    # Calculate cosine similarity
    dot_product = np.dot(question_embedding, answer_embedding)
    norm_q = np.linalg.norm(question_embedding)
    norm_a = np.linalg.norm(answer_embedding)
    
    return dot_product / (norm_q * norm_a)

def overview_node(state: TutorState, config:RunnableConfig) -> TutorState:  
    topic = state['topic']
    # Fetch relevant document chunks using global retriever
    en_retriever = config["configurable"]["en_retriever"] 
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
    llm = GLOBAL_LLM
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

def topic_selection(state: TutorState, config:RunnableConfig):
    topic = state.get("topic")
    retriever = config["configurable"]["safe_ret"]
    if not topic:
        # If frontend has not provided a topic yet
        state["branch"] = "awaiting_topic"
        return state

    exist, docs = topic_exists(retriever, topic)

    if not exist:
        state["branch"] = "invalid"
        return state

    # If topic exists
    # user_queries.append(topic)
    state["retrieved_docs"] = docs
    state["branch"] = "valid"
    return state

def evaluation_mode(state: TutorState) -> TutorState:
    state["evaluation_style"] = state.get("evaluation_style")
    state["mode"] = "Evaluation"
    return state

def SQFormation(state: TutorState, config: RunnableConfig):
    state["mode"] = "SQFormation"
    # Retrieve relevant chunks
    topic = state["topic"]
    en_retriever = config["configurable"]["en_retriever"] 
    content = en_retriever.invoke(topic)  
    document = "\n\n---\n\n".join([doc.page_content for doc in content])

    # Prompt (Notice the explicit JSON array instructions)
    prompt = PromptTemplate.from_template(
        """
        You are an exam question generator.
        Based on the following study material, generate around 15 Short Questions (SQ).

        Requirements:
        - Difficulty distribution: 10% basic, 30% easy, 40% medium, 20% hard.
        - Create conceptual questions that require a brief written answer.
        - Format: You MUST output ONLY a valid JSON array of objects. Example:
          [
            {{
              "question": "...",
              "answer": "...",
              "difficulty": "medium",
              "explanation": "..."
            }}
          ]
        - Try to cover each chunk at least once.
        
        Study Material:
        {document}
        """
    )

    # Call LLM
    llm = GLOBAL_LLM
    response = llm.invoke(prompt.format(document=document))

    # Clean the markdown
    cleaned_response = response.content.replace("```json", "").replace("```", "").strip()
    questions = json.loads(cleaned_response)

    # Mapping difficulty to enum
    for q in questions:
        diff_str = q["difficulty"].capitalize()  
        q["difficulty"] = Difficulty[diff_str]

    # FIX 1: Sort using the Difficulty enum directly
    sorted_questions = sorted(questions, key=lambda x: x["difficulty"].value)
    
    state["questions"] = sorted_questions
    state["question_no"] = 0
    state["total_questions"] = len(sorted_questions)
    
    totalQs = state["total_questions"]
    state["current_points"] = 0
    
    # FIX 2: Use the Difficulty enum class directly for the point values
    state["max_points"] = (0.1 * totalQs) * Difficulty.Basic.value + \
                          (0.3 * totalQs) * Difficulty.Easy.value + \
                          (0.4 * totalQs) * Difficulty.Medium.value + \
                          (0.2 * totalQs) * Difficulty.Hard.value  
                          
    # FIX 3: Use single quotes inside the f-string
    print(f"Max points are {state['max_points']}")
    
    state["mode"] = "Evaluation"
    return state
def MCQFormation(state: TutorState, config: RunnableConfig):
    state["mode"] = "MCQFormation"
    # Retrieve relevant chunks
    topic = state["topic"]
    en_retriever = config["configurable"]["en_retriever"] 
    content = en_retriever.invoke(topic)  
    document = "\n\n---\n\n".join([doc.page_content for doc in content])

    # Prompt (Added explicit instructions for a JSON Array)
    prompt = PromptTemplate.from_template(
        """
        You are an exam question generator.
        Based on the following study material, generate around 15 MCQs.

        Requirements:
        - Difficulty distribution: 10% basic, 30% easy, 40% medium, 20% hard.
        - Create more conceptual questions rather than factual oriented questions which depend majorly on the provided documents
        - Each question must have 4 options (a, b, c, d). Only one correct answer.
        - Format: You MUST output ONLY a valid JSON array of objects. Example:
          [
            {{
              "question": "...",
              "options": {{"a": "...", "b": "...", "c": "...", "d": "..."}},
              "answer": "a",
              "difficulty": "medium",
              "explanation":"..."
            }}
          ]
        - Try to cover each chunk at least once.
        
        Study Material:
        {document}
        """
    )

    # Call LLM
    llm = GLOBAL_LLM
    response = llm.invoke(prompt.format(document=document))

    # Clean the markdown
    cleaned_response = response.content.replace("```json", "").replace("```", "").strip()
    questions = json.loads(cleaned_response)

    # Mapping difficulty to enum
    for q in questions:
        diff_str = q["difficulty"].capitalize()  
        q["difficulty"] = Difficulty[diff_str]

    # FIX: Sort using the Difficulty enum directly, no need for state["difficulty"]
    sorted_questions = sorted(questions, key=lambda x: x["difficulty"].value)
    
    state["questions"] = sorted_questions
    state["question_no"] = 0
    state["total_questions"] = len(sorted_questions)
    
    totalQs = state["total_questions"]
    state["current_points"] = 0
    
    # FIX: Use the Difficulty enum class directly for the point values
    state["max_points"] = (0.1 * totalQs) * Difficulty.Basic.value + \
                          (0.3 * totalQs) * Difficulty.Easy.value + \
                          (0.4 * totalQs) * Difficulty.Medium.value + \
                          (0.2 * totalQs) * Difficulty.Hard.value  
                          
    # FIX: Use single quotes inside the f-string dictionary lookup
    print(f"Max points are {state['max_points']}")
    
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
    user_answer_lower = user_answer.lower().strip()
    correct_answer = question["answer"].lower().strip()
    
    # Safely get the explanation (defaulting just in case the AI forgot it)
    explanation = question.get("explanation", "No explanation provided.")
    
    # Update points and provide feedback
    if user_answer_lower == correct_answer:
        state["current_points"] = state["current_points"] + question["difficulty"].value
        state["answer_correct"] = True
        
        # FIX: Single quotes inside the F-string, and added the explanation!
        state["feedback_text"] = f"✅ **Correct!** You got {question['difficulty'].value} points.\n\n**Explanation:** {explanation}"
    else:
        state["answer_correct"] = False
        correct_option = question["options"][correct_answer]
        
        # Added the explanation here too!
        state["feedback_text"] = f"❌ **Incorrect.** Correct answer is **{correct_answer.upper()}) {correct_option}**.\n\n**Explanation:** {explanation}"
    
    # Increment for the next round
    state["question_no"] = state["question_no"] + 1
    
    # Clear the user answer so it doesn't accidentally trigger again
    state["user_answer"] = ""
    
    return state

def continue_or_end(state: TutorState) -> str:
    if state["question_no"] < state["total_questions"]:
        return "continue"
    return "done"

def Feedback(state:TutorState):
    percentage = (state["current_points"]/state["max_points"] )*100
