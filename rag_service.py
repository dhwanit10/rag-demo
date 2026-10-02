import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from elevenlabs.client import ElevenLabs

PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "book_rag"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_rag():
    """
    Initializes and returns the RAG chain and doc count.
    """
    load_dotenv()
    
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is missing from environment variables.")
        
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"ChromaDB directory not found at {PERSIST_DIRECTORY}. Please run ingest.py first.")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    
    doc_count = vectorstore._collection.count()
    if doc_count == 0:
        raise ValueError("Vector store is empty. Please run ingest.py first.")
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # LLM initialization
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.3)

    template = """You are a knowledgeable assistant answering questions based strictly on the provided book context.
If the answer is not in the context, say "I don't have enough information in the book to answer that question."

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, doc_count, retriever

def generate_tts_audio(text: str) -> bytes | None:
    """
    Generates speech audio bytes using ElevenLabs API.
    Returns bytes or None if key is missing/error occurs.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return None
        
    try:
        client = ElevenLabs(api_key=api_key)
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
            model_id="eleven_flash_v2_5"
        )
        # Collect audio chunks into bytes
        audio_bytes = b"".join(chunk for chunk in audio_stream)
        return audio_bytes
    except Exception as e:
        print(f"ElevenLabs TTS Error: {e}")
        return None
