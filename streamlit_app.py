import os
import streamlit as st
from rag_service import initialize_rag, generate_tts_audio

# Page Configuration
st.set_page_config(
    page_title="Book Knowledge Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom Styling (Strictly Emerald Green, Warm Amber, Slate, and Dark Charcoal - NO blue/violet/pink/purple)
CUSTOM_CSS = """
<style>
    /* Root & Main Container Background */
    .stApp {
        background-color: #0B0F0D;
        color: #E5E7EB;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        color: #10B981;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #9CA3AF;
        font-size: 0.95rem;
    }
    
    /* Stat Badge */
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    .badge-emerald {
        background-color: #064E3B;
        color: #A7F3D0;
        border: 1px solid #059669;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-amber {
        background-color: #451A03;
        color: #FDE68A;
        border: 1px solid #D97706;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Chat Messages Styling */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
    }
    
    div[data-testid="stChatMessageContent"] {
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
    }

    /* User Message Bubble */
    div[data-testid="stChatMessage"]:nth-child(even) div[data-testid="stChatMessageContent"] {
        background-color: #14231B !important;
        border: 1px solid #059669 !important;
        color: #ECFDF5 !important;
    }

    /* Assistant Message Bubble */
    div[data-testid="stChatMessage"]:nth-child(odd) div[data-testid="stChatMessageContent"] {
        background-color: #1C1F1A !important;
        border: 1px solid #D97706 !important;
        color: #FEF3C7 !important;
    }

    /* Input Box & Controls */
    .stChatInputContainer input {
        background-color: #131916 !important;
        color: #F3F4F6 !important;
        border: 1px solid #059669 !important;
        border-radius: 10px !important;
    }
    
    .stChatInputContainer input:focus {
        border-color: #10B981 !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #059669 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.4) !important;
    }

    /* Audio Player Custom Styling */
    audio {
        filter: sepia(20%) saturate(120%) hue-rotate(90deg) contrast(90%);
        border-radius: 8px;
        margin-top: 0.5rem;
        width: 100%;
    }

    /* Hide Streamlit Header/Footer Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Alert Boxes */
    .stAlert {
        background-color: #1A221E !important;
        border: 1px solid #059669 !important;
        color: #D1FAE5 !important;
        border-radius: 10px !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize RAG Pipeline (Cached in Session State to avoid reload)
@st.cache_resource(show_spinner=False)
def load_rag():
    return initialize_rag()

try:
    with st.spinner("Connecting knowledge base..."):
        rag_chain, doc_count, retriever = load_rag()
    rag_ready = True
except Exception as e:
    rag_ready = False
    rag_error = str(e)

# Header Section
st.markdown(
    """
    <div class="main-header">
        <h1>📚 Book Intelligence Assistant</h1>
        <p>Ask any question about your document — powered by Gemini & ChromaDB RAG</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Badges & Status
if rag_ready:
    has_elevenlabs = bool(os.environ.get("ELEVENLABS_API_KEY"))
    tts_badge = '<span class="badge-emerald">🎙️ Voice Output Enabled</span>' if has_elevenlabs else '<span class="badge-amber">🔇 Voice Disabled (Missing ElevenLabs Key)</span>'
    st.markdown(
        f"""
        <div class="badge-container">
            <span class="badge-emerald">📖 {doc_count} Chunks Indexed</span>
            {tts_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.error(f"Failed to initialize RAG Pipeline: {rag_error}")
    st.info("Please verify `.env` configuration and make sure you ran `python ingest.py` first.")

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am your book assistant. Ask me anything about the loaded book, and I will search the document to provide an accurate answer.",
            "audio": None
        }
    ]

# Display Chat Messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3")

# Chat Input & Processing
if rag_ready:
    if user_query := st.chat_input("Ask a question about the book..."):
        # Append and show user message
        st.session_state["messages"].append({"role": "user", "content": user_query, "audio": None})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating answer..."):
                try:
                    answer = rag_chain.invoke(user_query)
                except Exception as ex:
                    answer = f"Error generating answer: {ex}"

                st.markdown(answer)

                # Generate Audio if TTS key available
                audio_bytes = None
                if os.environ.get("ELEVENLABS_API_KEY"):
                    with st.spinner("Generating speech..."):
                        audio_bytes = generate_tts_audio(answer)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")

            # Save to chat history
            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "audio": audio_bytes
            })
