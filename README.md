# RAG with re-ranking

A retrieval-augmented generation (RAG) pipeline over a local document, enhanced with **BGE cross-encoder re-ranking**.  

## Prerequisites

- Python 3.12 or higher
- uv
- API keys for:
  - OpenAI


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Pranavisriya/rag-with-re-ranking.git
cd rag-with-re-ranking
```

2. Install `uv` in the environment if it is not present
```bash
pip install uv
```

3. Create a virtual python environment in this repo
```bash
uv init
uv venv -p 3.12
```

Any other method can also be used to create python environment.

4. Activate python environment
```bash
source .venv/bin/activate
```


5. Install dependencies using uv:
```bash
uv add -r requirements.txt
```

6. Create a `.env` file in the project root with your API keys:
```
OPEN_API_KEY=your_openaiapi_key
```

## Usage

Run the fastapi and frontend:
```bash
uvicorn backend:app --reload --port 8000
streamlit run frontend.py
```


## Features

- RAG over a local document (`data/demo.txt`) using Chroma as a vector store  
- BGE cross-encoder re-ranking (`BAAI/bge-reranker-v2-m3`) on top of embedding-based retrieval  
- Final answer generation using OpenAI chat models (e.g., `gpt-4o-mini`) with context from re-ranked chunks  
- Clean separation of concerns:
  - `backend.py` – FastAPI API exposing a `/rag` endpoint
  - `frontend.py` – Streamlit app that calls the API and shows only the final answer
- Environment-based configuration via `.env` (no secrets committed to git)

  
## License

This project is licensed under the terms included in the LICENSE file.

## Author

Pranavi Sriya (pranavisriyavajha9@gmail.com)




