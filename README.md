# Document Chatbot API

A FastAPI-based chatbot that allows you to ask questions about content from Word documents using RAG (Retrieval-Augmented Generation).

## Features

- **Document Processing**: Extracts and chunks text from .docx files
- **Vector Search**: Uses Pinecone for semantic search
- **AI Chat**: OpenAI GPT-4o-mini for generating answers
- **RAG Implementation**: Only answers from document content

## Setup Instructions

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Copy the environment template
copy env_template.txt .env

# Edit .env file with your API keys
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=chatbot-index
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Process Documents (Optional)
```bash
python process_docs.py
```
This will process all .docx files in the `data/` folder and create chunks.

### 4. Run the API Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Chat Endpoint
- **POST** `/chat/`
- **Body**: `{"query": "your question here"}`
- **Response**: `{"answer": "AI generated answer"}`

### Ingest Endpoint
- **POST** `/ingest/`
- **Body**: `{"transcript": "text to add to knowledge base"}`
- **Response**: `{"status": "ingested", "id": "document_id"}`

## Usage Examples

### Using curl:
```bash
# Ask a question
curl -X POST "http://localhost:8000/chat/" -H "Content-Type: application/json" -d "{\"query\": \"What is the main topic of chapter 1?\"}"

# Add new content
curl -X POST "http://localhost:8000/ingest/" -H "Content-Type: application/json" -d "{\"transcript\": \"New content to add\"}"
```

### Using Python requests:
```python
import requests

# Ask a question
response = requests.post("http://localhost:8000/chat/", json={"query": "What is discussed in episode 1?"})
print(response.json()["answer"])

# Add content
response = requests.post("http://localhost:8000/ingest/", json={"transcript": "New content"})
print(response.json())
```

## What Happens When You Run It

1. **Startup**: 
   - Loads environment variables
   - Initializes Pinecone client
   - Creates Pinecone index if it doesn't exist
   - Starts FastAPI server

2. **First Request**:
   - If no data in Pinecone, returns "Sorry, this is not in the transcripts"
   - You need to ingest documents first

3. **Document Processing**:
   - Extracts text from .docx files
   - Chunks text into manageable pieces
   - Creates embeddings using OpenAI
   - Stores in Pinecone vector database

4. **Chat Functionality**:
   - Converts your question to embedding
   - Searches Pinecone for similar content
   - Uses found content as context for AI response
   - Returns answer based only on document content

## File Structure
```
backend/
├── app/
│   ├── main.py              # FastAPI app
│   ├── routers/
│   │   ├── chat.py          # Chat endpoint
│   │   └── ingest.py        # Ingest endpoint
│   ├── services/
│   │   ├── pinecone_client.py  # Pinecone setup
│   │   ├── embeddings.py    # OpenAI embeddings
│   │   └── llm.py          # OpenAI chat completion
│   └── utils/
│       └── docx_parser.py   # Document processing
├── data/                    # Your .docx files
├── process_docs.py         # Document processing script
└── requirements.txt        # Dependencies
```

