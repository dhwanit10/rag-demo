import os
import glob
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
COLLECTION_NAME = "book_rag"
BATCH_SIZE = 50  # Chunks per batch to avoid Gemini rate limits


def main():
    print(f"{YELLOW}Starting ingestion process...{RESET}")
    load_dotenv()

    if not os.environ.get("GOOGLE_API_KEY"):
        print(f"{RED}Error: GOOGLE_API_KEY not found in .env{RESET}")
        return

    if not os.path.exists(DOCUMENTS_DIR):
        print(f"Directory {DOCUMENTS_DIR} does not exist. Creating it...")
        os.makedirs(DOCUMENTS_DIR)
        print("Please add PDF files to the 'documents' directory and run again.")
        return

    pdf_files = glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf"))
    if not pdf_files:
        print(f"{RED}No PDF files found in {DOCUMENTS_DIR}.{RESET}")
        return

    # --- Load all PDFs ---
    all_docs = []
    for pdf_file in pdf_files:
        print(f"Loading {os.path.basename(pdf_file)}...")
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"  -> Loaded {len(docs)} pages")

    print(f"\n{YELLOW}Total pages loaded: {len(all_docs)}{RESET}")

    # --- Split into chunks ---
    print(f"{YELLOW}Splitting text into chunks...{RESET}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(all_docs)

    # Filter out empty/whitespace-only chunks
    chunks = [c for c in chunks if c.page_content.strip()]
    print(f"Created {len(chunks)} non-empty chunks.")

    # --- Clear old vector store to avoid duplicates ---
    if os.path.exists(PERSIST_DIR):
        print(f"{YELLOW}Removing old vector store...{RESET}")
        shutil.rmtree(PERSIST_DIR)

    # --- Create embeddings and store in batches ---
    print(f"{YELLOW}Creating embeddings and storing in ChromaDB...{RESET}")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Process in batches to respect API rate limits
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    vectorstore = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=PERSIST_DIR,
                collection_name=COLLECTION_NAME,
            )
        else:
            vectorstore.add_documents(batch)

    print(f"\n{GREEN}✓ Successfully ingested {len(pdf_files)} file(s)!{RESET}")
    print(f"{GREEN}✓ Created {len(chunks)} chunks in collection '{COLLECTION_NAME}'{RESET}")
    print(f"{GREEN}✓ Vector store saved to {PERSIST_DIR}{RESET}")


if __name__ == "__main__":
    main()
