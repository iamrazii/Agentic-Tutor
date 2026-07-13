
from pydantic import BaseModel
from typing import TypedDict, Literal,Union
from enum import Enum
from langgraph.graph import StateGraph, START, END
from langchain.docstore.document import Document

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


class StartSessionRequest(BaseModel):
    thread_id: str
    session_id: str

class TopicRequest(BaseModel):
    session_id: str
    thread_id: str
    topic: str

class EvaluationStyleRequest(BaseModel):
    thread_id: str
    session_id: str
    evaluation_style: Literal["MCQ", "SQ"]

class AnswerRequest(BaseModel):
    thread_id: str
    session_id: str
    user_answer: str



class QueryRequest(BaseModel):
    session_id: str
    question: str

class StartTutorRequest(BaseModel):
    session_id: str # We use session_id to fetch their retrievers
    thread_id: str  # LangGraph's specific memory ID