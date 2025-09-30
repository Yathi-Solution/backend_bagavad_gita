# 🕉️ Bhagavad Gita AI Chat Application

A modern web application that allows you to chat with an AI assistant trained on Bhagavad Gita Chapter 1 teachings by Chinna Jeeyar Swamiji. This application uses RAG (Retrieval-Augmented Generation) to provide accurate and contextual responses based on the sacred texts.

## ✨ Features

- **AI-Powered Chat**: Ask questions about Bhagavad Gita teachings with intelligent responses
- **Semantic Search**: Find relevant passages using vector embeddings and Pinecone
- **Multiple Query Types**: Handles various question formats:
  - Direct Shloka/Chapter queries
  - Concept/Definition questions
  - Philosophical comparisons
  - Real-world applications
  - Character/Context questions
  - Source/Citation queries
- **Modern UI**: Beautiful, responsive web interface
- **Real-time Responses**: Fast AI-generated answers using OpenAI GPT-4o-mini
- **Vector Database**: Uses Pinecone for efficient semantic search

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd backend_bagavad_gita
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX=bhagavad-gita-chapter1
   ```

4. **Run the Application**
   ```bash
   cd app1
   python run_app.py
   ```

5. **Access the Application**
   - **Frontend**: http://localhost:8000/app
   - **API Docs**: http://localhost:8000/docs
   - **API**: http://localhost:8000

## 📁 Project Structure

```
backend_bagavad_gita/
├── app1/                           # Main application folder
│   ├── frontend/
│   │   └── index.html             # Web interface
│   ├── services/
│   │   ├── llm_service.py         # AI response generation
│   │   ├── pinecone_services.py   # Vector database operations
│   │   ├── json_processor.py      # Data processing
│   │   ├── embeddings.py          # Text embedding generation
│   │   └── doc_extract.py         # Document extraction utilities
│   ├── endpoints/
│   │   └── chat.py               # FastAPI endpoints
│   ├── models/
│   │   └── __init__.py           # Pydantic models
│   ├── data/
│   │   ├── chapter_1_episodes.json    # Bhagavad Gita structured data
│   │   ├── chapter1/              # Original .docx files
│   │   └── chapter1_extracted_text.txt # Extracted text
│   ├── run_app.py               # Application starter
│   └── README.md               # App-specific documentation
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🔧 API Endpoints

- `GET /` - API status and health check
- `GET /app` - Serve frontend application
- `POST /chat/llm` - Chat with AI assistant (RAG-powered)
- `POST /search` - Search passages using semantic similarity
- `POST /chat` - Simple chat endpoint
- `POST /process-json` - Process and store JSON data in Pinecone
- `GET /episodes` - Get episode information and statistics
- `GET /health` - Health check with Pinecone connection status

## 💬 Example Queries

- "What is the purpose of Krishna's incarnation?"
- "I'm facing failure in my business. How does the Gita suggest I cope?"
- "What did Sanjaya see on the battlefield?"
- "How does Bhakti differ from Jnana?"
- "What is the meaning of dharma?"

## 🛠️ Development

### First Time Setup
1. **Process Data**: After starting the server, call `/process-json` to load Bhagavad Gita data into Pinecone
2. **Test Connection**: Visit `/health` to verify Pinecone connection
3. **Start Chatting**: Use `/app` to access the web interface

### Adding New Data
1. Add new JSON data to `app1/data/` directory
2. Update `json_processor.py` to handle new format if needed
3. Run `/process-json` endpoint to update embeddings

### Customizing Responses
- Modify `llm_service.py` for AI behavior and response style
- Update `frontend/index.html` for UI changes
- Adjust chunk size and overlap in `json_processor.py` for better accuracy

## 🔄 How It Works

1. **Data Processing**: 
   - JSON data is chunked into manageable pieces with overlap
   - Text embeddings are created using OpenAI's text-embedding-3-large
   - Embeddings are stored in Pinecone vector database

2. **Query Processing**:
   - User queries are converted to embeddings
   - Semantic search finds relevant passages in Pinecone
   - Retrieved context is used for AI response generation

3. **Response Generation**:
   - AI preserves exact quotes from Swamiji's teachings
   - Responses are conversational and engaging
   - Context is limited to Bhagavad Gita Chapter 1 content

## 📝 License

This project is for educational and spiritual learning purposes.

## 🙏 Acknowledgments

- Bhagavad Gita teachings by Chinna Jeeyar Swamiji
- OpenAI for AI capabilities
- Pinecone for vector database
- FastAPI for backend framework

