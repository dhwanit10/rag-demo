# RAG Demo with Voice Output

A full end-to-end Retrieval-Augmented Generation (RAG) application that answers questions based on a provided PDF book and speaks the answer using ElevenLabs Text-to-Speech.

## Tech Stack
- **LangChain**: Orchestration framework
- **Google Gemini**: LLM (`gemini-2.0-flash`) and Embeddings (`models/embedding-001`)
- **ChromaDB**: Local Vector Store
- **ElevenLabs**: Text-to-Speech
- **PyPDF**: PDF document loading
- **Pygame**: Audio playback

## Setup Instructions

1. **Install Dependencies**
   Make sure you have Python 3.13 installed. Then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **API Keys**
   Add your API keys to the `.env` file:
   - `GOOGLE_API_KEY`: Get it from Google AI Studio
   - `SARVAM_API_KEY`: Get it from ElevenLabs (optional, but needed for voice)

3. **Prepare Documents**
   Create a `documents/` folder and place your PDF books inside it.
   ```bash
   mkdir documents
   # Move your PDF files into this directory
   ```

4. **Ingest Documents**
   Run the ingestion script to process the PDFs and create the vector database:
   ```bash
   python ingest.py
   ```

5. **Run the Application**
   - **Terminal UI**:
     ```bash
     python app.py
     ```
   - **Streamlit Web UI**:
     ```bash
     streamlit run streamlit_app.py
     ```
