# URP Student Agent - AI-Powered Q&A System

AI-powered student agent system that uses Retrieval-Augmented Generation (RAG) to answer multiple-choice questions based on study materials. The system creates personalized student agents that learn from uploaded documents and can answer questions with varying levels of confidence.

## 🚀 Features

- **Intelligent Student Agents**: Create AI agents that simulate student knowledge based on studied materials
- **RAG-Enhanced Answering**: Uses advanced retrieval mechanisms to find relevant information from study documents
- **Multiple Answer Strategies**: Supports both RAG-based and system prompt-based answer generation
- **Document Processing**: Automatically processes and indexes PDF and PowerPoint files
- **Confidence Scoring**: Provides confidence levels for each answer
- **Performance Tracking**: Tracks agent performance and logs results to CSV files
- **Web Interface**: User-friendly Gradio interface for managing agents and questions
- **RESTful API**: FastAPI-based backend for programmatic access

## 🏗️ Architecture

The system consists of several key components:

### Backend (FastAPI)
- **`app/main.py`**: Main API server with endpoints for agent management and question answering
- **MongoDB Integration**: Stores agents, questions, and processed documents
- **Vector Database**: Uses MongoDB Atlas Vector Search for semantic similarity

### AI Agents
- **`agents/student_rag.py`**: RAG-based question answering with document retrieval
- **`agents/student_systemprompt.py`**: Direct LLM-based question answering
- **`agents/file_extractor.py`**: Document processing and embedding generation
- **`agents/semantic_chunker.py`**: Intelligent text chunking for better retrieval

### Frontend
- **`frontend/app.py`**: Gradio-based web interface for user interaction
- **Agent Management**: Create, view, and delete student agents
- **Question Interface**: Browse and answer questions
- **File Upload**: Upload study materials

## 📦 Installation

### Prerequisites
- Python 3.12
- MongoDB Atlas account
- OpenAI API key
- Ollama (for local LLM inference)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd URP_student_agent
   ```

2. **Create environment variables**
   Create a `.env` file under agents/ directory:
   ```env
   MONGODB_URL=your_mongodbatlas_uri
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama and pull the model**
    Under agents/models/ there are different Python scripts to create different student agents
   ```bash
   # Install Ollama (visit https://ollama.ai for installation instructions)
   python base_rag_model.py # or your preferred model
   ```

5. **Set up MongoDB Atlas**
   - Create a MongoDB Atlas cluster
   - Create a database named `agents_db`
   - Set up a vector search index named `vector_index` on the `documents` collection

## 🚀 Usage

### Starting the Backend
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8080`

### Starting the Frontend
```bash
cd frontend
python app.py
```
The web interface will be available at `http://localhost:7860`

### API Endpoints

#### Agent Management
- `POST /create_student` - Create a new student agent
- `GET /agents/{agent_id}` - Get agent details
- `GET /all_agents` - List all agents
- `DELETE /agents/{agent_id}` - Delete an agent

#### Question Answering
- `POST /answer/{agent_id}` - Answer a single question
- `POST /answer_questions/{agent_id}` - Answer multiple questions
- `GET /all_questions` - Get all available questions

#### File Management
- `POST /files` - Upload and process study materials

### Example Usage

1. **Create a Student Agent**
   ```python
   import requests
   
   agent_data = {
       "name": "Student1",
       "studied": ["Week1.pptx", "Week2.pptx"]
   }
   
   response = requests.post("http://localhost:8080/create_student", json=agent_data)
   agent_id = response.json()["student_id"]
   ```

2. **Answer Questions**
   ```python
   question_data = {
       "qn_ids": ["Q001", "Q002", "Q003"]
   }
   
   response = requests.post(f"http://localhost:8080/answer_questions/{agent_id}", json=question_data)
   results = response.json()
   ```

## 📊 Data Models

### Student Agent
```python
{
    "name": "Student1",
    "studied": ["Week1.pptx", "Week2.pptx"],
    "messages": [
        {
            "query": "What is machine learning?",
            "response": "Machine learning is...",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    ]
}
```

### Question
```python
{
    "qn_id": "Q001",
    "qn_week": 1,
    "qn_stem": "What is the primary purpose of data analytics?",
    "qn_options": "A. To collect data B. To analyze data C. To visualize data D. To store data",
    "correct_option": "B"
}
```

## 🔧 Configuration

### MongoDB Collections
- `agents`: Student agent profiles
- `documents`: Processed study materials with embeddings
- `questions`: Available questions (LLM-generated)
- `questions_manual`: Manually created questions

Restore data with data in dump/

## 📈 Performance Monitoring

The system automatically logs performance metrics:
- Answer accuracy
- Confidence scores
- Results are saved to CSV files under results/

## 🛠️ Development

### Project Structure
```
URP_student_agent/
├── app/                    # FastAPI backend
│   └── main.py            # Main API server
├── agents/                # AI agent components
│   ├── student_rag.py     # RAG-based answering
│   ├── student_systemprompt.py # Direct LLM answering
│   ├── file_extractor.py  # Document processing
│   └── semantic_chunker.py # Text chunking
├── frontend/              # Web interface
│   └── app.py            # Gradio interface
├── resources/             # Static resources
├── dump/                  # Database dumps
└── results/              # Performance logs
```

### Adding New Features
1. **New Agent Types**: Extend the base agent classes in the `agents/` directory
2. **Custom Prompts**: Modify the `PROMPTS` dictionary in agent files
3. **New Endpoints**: Add routes to `app/main.py`
4. **UI Components**: Extend the Gradio interface in `frontend/app.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for AI orchestration
- Uses [Ollama](https://ollama.ai/) for local LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) for the backend API
- [Gradio](https://gradio.app/) for the web interface
- [MongoDB Atlas](https://www.mongodb.com/atlas) for data storage

## 🐛 Troubleshooting

### Common Issues

1. **MongoDB Connection Error**
   - Ensure your MongoDB Atlas cluster is running
   - Check your connection string in the `.env` file
   - Verify network access settings in Atlas

2. **Ollama Model Not Found**
   - Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
   - Pull the required model: `ollama pull llama3.2:3b`

3. **OpenAI API Errors**
   - Verify your API key is correctly set in the `.env` file
   - Check your OpenAI account has sufficient credits

4. **File Processing Issues**
   - Ensure uploaded files are in supported formats (PDF, PPTX)
   - Check file size limits and available storage

### Getting Help

If you encounter issues:
1. Check the logs in the console output
2. Verify all environment variables are set correctly
3. Ensure all dependencies are installed
4. Check the MongoDB Atlas connection and vector search index setup

For additional support, please open an issue in the repository.
