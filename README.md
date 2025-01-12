# LLM-Chatbot
Chatbot build using llama model


### Repo Structure
grade-specific-chatbot/
│
├── data/
│   ├── books/
│   │   ├── Grade4A.pdf   # PDF for Grade 4 Book A
│   │   ├── Grade4B.pdf   # PDF for Grade 4 Book B
│   │   ├── Grade5A.pdf   # PDF for Grade 5 Book A
│   │   ├── Grade5B.pdf   # PDF for Grade 5 Book B
│   │
│   ├── processed/
│       ├── Grade4.json   # Preprocessed and structured content for Grade 4
│       ├── Grade5.json   # Preprocessed and structured content for Grade 5
│
├── src/
│   ├── backend/
│   │   ├── main.py           # Entry point for backend (FastAPI/Flask app)
│   │   ├── retrieval.py      # Handles retrieval logic (RAG pipeline)
│   │   ├── embedding.py      # Functions for text embedding and indexing
│   │   ├── model.py          # Interfaces with the LLM for answer generation
│   │   ├── utils.py          # Utility functions (e.g., text preprocessing)
│   │   ├── database/
│   │       ├── faiss_index.pkl   # Prebuilt FAISS index for retrieval
│   │       ├── chromadb/         # Chroma database files (if used)
│   │
│   ├── frontend/
│   │   ├── app.py          # Gradio/Streamlit app for chatbot UI
│   │   ├── components.py   # Custom UI components or reusable functions
│   │
│   ├── config/
│       ├── settings.yaml   # Configuration file (API keys, model settings, paths)
│
├── docker/
│   ├── Dockerfile          # Dockerfile to containerize the app
│   ├── docker-compose.yml  # Compose file (if using multiple services)
│
├── tests/
│   ├── test_retrieval.py   # Unit tests for retrieval module
│   ├── test_model.py       # Unit tests for model interactions
│   ├── test_ui.py          # UI tests for the frontend
│
├── notebooks/
│   ├── data_preprocessing.ipynb  # Notebook for extracting and preprocessing book content
│   ├── testing_embeddings.ipynb  # Notebook for testing embeddings and retrieval
│
├── requirements.txt        # Python dependencies
├── README.md               # Documentation for the project
├── LICENSE                 # License file (if applicable)
├── .gitignore              # Files and directories to ignore in Git
└── run.sh                  # Script to build and run the Docker container
