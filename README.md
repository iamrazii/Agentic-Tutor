
# Agentic Tutor

Agentic Tutor is an intelligent exam-preparation system designed to teach and evaluate, and adapt to a student’s understanding of the material provided by him/her.
The system is built using an agentic architecture, and structured learning flows.




## Key Features

    1. Dual Interaction Modes

        Query Mode:
        Concise, exam-ready answers for quick revision.

        Tutor Mode:
        Interactive teaching flow with explanation → questioning → evaluation → feedback.

    2. Evaluation Methods:

        It relies on two modes: 
         1. Short Questions 
         2. Multiple Choice Questions. 
        
        Short Questions responses are compared with expected general responses and scored accordingly.
        For MCQs, Four possible answers are provided to choose from.

    3. Adaptive Evaluation Engine

        It Tracks:
            > Topic
            > Question type
            > Difficulty level
            > Learner performance
            > Uses weighted scoring (correctness × difficulty)
    
    4. Intent-Aware Routing
    
        > Automatically classifies user intent (e.g., factual query vs learning session)
        > Routes interactions to the appropriate agent chain

    5. Agentic Design

        > Modular, tool-based architecture
        > Clear separation between:
            > Reasoning
            > Teaching
            > Evaluation
            >Response generation
## System Architecture

    1. Intent Classifier:
        
        > Determines whether the user wants:
            > Quick answer
            > Comprehensive Answer
            > General Information about document

    2. Document Parsing:
        
        > Parses Document based on sections and headings. 
        > It uses GoLang to extract text from the document and feeds chunks to embedding models


    3. Storage and retrieval: 

        > Use FAISS vector store to store embeddings of chunks
        > Use hybrid retrieval techniques(Ensemble+BM25)


    4. Evaluation Module

        > Scores answers based on difficulty
        > Maintains learner performance profile


    5. Response Generator

        > Ensures answers are:
            > Exam-oriented
            > Concise
            > Conceptually correct
## Tech Stack

    > Langchain
        > text splitters
        > vector stores i.e FAISS
        > Huggingface Embeddings
        > Gemini LLM
        > Docstore

    > LangGraph
        
    > Steamlit -> frontend

    > Numpy


## NOTE

Since im using hugging face embeddings and free llm model of gemini, there are certain limitations on speed and accuracy
