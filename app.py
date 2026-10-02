import os
import tempfile
import pygame
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import base64
from sarvamai import SarvamAI

# ANSI colors
CYAN = '\033[96m'
MAGENTA = '\033[95m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def split_text(text, limit=2400):
    chunks, current = [], ""
    for sentence in text.replace("\n", " ").split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence += ". "
        if len(current) + len(sentence) > limit and current:
            chunks.append(current.strip())
            current = ""
        current += sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks

def play_audio(audio_b64_list):
    try:
        pygame.mixer.init()
        for audio_b64 in audio_b64_list:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(base64.b64decode(audio_b64))
                temp_path = temp_audio.name

            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            pygame.mixer.music.unload()
            os.remove(temp_path)
        pygame.mixer.quit()
    except Exception as e:
        print(f"{RED}Error playing audio: {e}{RESET}")

def main():
    print(f"{CYAN}Initializing RAG Application...{RESET}")
    load_dotenv()
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print(f"{RED}Error: GOOGLE_API_KEY not found in .env{RESET}")
        return
        
    if not os.environ.get("SARVAM_API_KEY"):
        print(f"{YELLOW}Warning: SARVAM_API_KEY not found. TTS will not work.{RESET}")
        has_tts = False
    else:
        has_tts = True
        sarvam_client = SarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY"))

    persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")

    if not os.path.exists(persist_directory):
        print(f"{RED}Error: ChromaDB directory not found. Please run ingest.py first.{RESET}")
        return

    print(f"{YELLOW}Loading vector store...{RESET}")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="book_rag",
    )
    
    doc_count = vectorstore._collection.count()
    print(f"{CYAN}Vector store loaded: {doc_count} chunks available{RESET}")
    if doc_count == 0:
        print(f"{RED}Error: Vector store is empty! Run ingest.py first.{RESET}")
        return
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print(f"{YELLOW}Setting up LLM and Chain...{RESET}")
    llm = ChatGoogleGenerativeAI(model="models/gemini-3.5-flash-lite")

    template = """You are a helpful assistant that answers questions based on the provided context from a book.
If the answer is not in the context, say "I don't have enough information in the book to answer that question. if you dont have the context, say I do not have context."

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

    print(f"{GREEN}Ready!{RESET}")
    print("-" * 50)
    print(f"{CYAN}Welcome to the Book Assistant! (Type 'quit' or 'exit' to stop){RESET}")
    
    while True:
        try:
            user_input = input(f"\n{MAGENTA}You: {RESET}")
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            print(f"{YELLOW}Thinking...{RESET}")
            
            # Generate answer
            answer = rag_chain.invoke(user_input)
            
            print(f"\n{GREEN}Assistant: {RESET}{answer}")
            
            # Text to Speech
            if has_tts:
                print(f"{YELLOW}(Generating speech...){RESET}")
                try:
                    audio_parts = []
                    for chunk in split_text(answer):
                        response = sarvam_client.text_to_speech.convert(
                            text=chunk,
                            language_code="en-IN",
                            speaker="shubh",
                            model="bulbul:v3",
                        )
                        audio_parts.extend(response.audios)
                    play_audio(audio_parts)
                except Exception as e:
                    print(f"{RED}TTS Error: {e}{RESET}")
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"{RED}Error processing request: {e}{RESET}")

if __name__ == "__main__":
    main()
