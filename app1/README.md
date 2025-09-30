# 🕉️ Bhagavad Gita AI Chat Application

A modern web application that allows you to chat with an AI assistant trained on Bhagavad Gita Chapter 1 teachings by Chinna Jeeyar Swamiji.

## ✨ Features

- **AI-Powered Chat**: Ask questions about Bhagavad Gita teachings
- **Semantic Search**: Find relevant passages using vector embeddings
- **Multiple Query Types**: Handles various question formats:
  - Direct Shloka/Chapter queries
  - Concept/Definition questions
  - Philosophical comparisons
  - Real-world applications
  - Character/Context questions
  - Source/Citation queries
- **Modern UI**: Beautiful, responsive web interface
- **Real-time Responses**: Fast AI-generated answers

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key

### Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX=bhagavad-gita-chapter1
   ```

3. **Process Data (First Time Only)**
   ```bash
   # Process JSON data and create embeddings
   python -c "import requests; requests.post('http://localhost:8000/process-json')"
   ```

4. **Run the Application**
   ```bash
   python run_app.py
   ```

5. **Access the Application**
   - **Frontend**: http://localhost:8000/app
   - **API Docs**: http://localhost:8000/docs
   - **API**: http://localhost:8000

## 📁 Project Structure

```
app1/
├── frontend/
│   └── index.html          # Web interface
├── services/
│   ├── llm_service.py      # AI response generation
│   ├── pinecone_services.py # Vector database operations
│   ├── json_processor.py   # Data processing
│   └── embeddings.py       # Text embedding generation
├── endpoints/
│   └── chat.py            # FastAPI endpoints
├── models/
│   └── __init__.py        # Pydantic models
├── data/
│   └── chapter_1_episodes.json # Bhagavad Gita data
└── run_app.py             # Application starter
```

## 🔧 API Endpoints

- `GET /` - API status
- `GET /app` - Serve frontend application
- `POST /chat` - Chat with AI assistant
- `POST /search` - Search passages
- `POST /process-json` - Process data
- `GET /episodes` - Get episode information
- `GET /health` - Health check

## 💬 Example Queries

- "What is the purpose of Krishna's incarnation?"
- "I'm facing failure in my business. How does the Gita suggest I cope?"
- "What did Sanjaya see on the battlefield?"
- "How does Bhakti differ from Jnana?"
- "What is the meaning of dharma?"

## 🛠️ Development

### Adding New Data
1. Add new JSON data to `data/` directory
2. Update `json_processor.py` to handle new format
3. Run `/process-json` endpoint

### Customizing Responses
- Modify `llm_service.py` for AI behavior
- Update `frontend/index.html` for UI changes

## 📝 License

This project is for educational and spiritual learning purposes.

## 🙏 Acknowledgments

- Bhagavad Gita teachings by Chinna Jeeyar Swamiji
- OpenAI for AI capabilities
- Pinecone for vector database
- FastAPI for backend framework

