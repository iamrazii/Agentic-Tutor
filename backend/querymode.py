from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableBranch
from dotenv import load_dotenv
import os
import nest_asyncio
from backend.textprocessing import Preprocessor

# Combined initialization
nest_asyncio.apply()
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_TOKEN")

def SettingUp(fileobjs):
    preprocessor = Preprocessor()
    chunks = preprocessor.load_and_process(fileobjs)
    en_retriever, strict_retriever = preprocessor.build_retriever(chunks)
    return chunks, en_retriever,strict_retriever

def build_chain(chunks, hybrid_retriever):
    # Single LLM instance with higher temperature for faster processing
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=1.0)
    parser = StrOutputParser()
    
    # Pre-compute full context once
    full_context = "\n\n".join([doc.page_content for doc in chunks])[:20000]
    
    # Simplified single prompt for intent classification with combined logic
    intent_prompt = PromptTemplate.from_template(
        "Classify query as: summary, general, detail, or factual. Query: {question}. Intent:"
    )
    
    # Single unified prompt template for all response types
    unified_prompt = PromptTemplate.from_template(
        "History: {chat_history}\nContext: {context}\n"
        "Intent: {intent}\nQuery: {question}\n"
        "Response based on intent - summary: overview, general: author/title info, "
        "detail: comprehensive answer, factual: precise facts:"
    )
    
    # Combined chains
    intent_chain = intent_prompt | llm | parser
    
    # Fast intent classification with single call
    def get_intent_and_context(x):
        intent = intent_chain.invoke({"question": x["question"]}).strip().lower()
        
        # Skip rewriting step and use original query directly for speed
        if intent in ["summary", "general", "detail"]:
            context = full_context
        else:  # factual
            # Direct retrieval without rewriting
            retrieved_docs = hybrid_retriever.invoke(x["question"])
            context = "\n\n".join([
                f"[{doc.metadata.get('heading', '')}"
                f"{'#p' + str(doc.metadata.get('page')) if doc.metadata.get('page') is not None else ''}] "
                f"{doc.page_content}"
                for doc in retrieved_docs
            ])
        
        return {
            "context": context,
            "question": x["question"],
            "chat_history": x.get("chat_history", ""),
            "intent": intent
        }
    
    # Single unified chain instead of multiple branches
    fast_chain = RunnableMap({
        "unified_input": get_intent_and_context
    }) | (lambda x: x["unified_input"]) | unified_prompt | llm | parser
    
    return fast_chain

def format_chat_history(history):
    return "\n".join([f"User: {q}\nBot: {a}" for q, a in history])