# api.py

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# import your rag_answer from the code you pasted
# if your file is named differently, change "main" below
from main import rag_answer


# ---------- Pydantic models ----------

class ChunkOut(BaseModel):
    rank: int
    chunk_id: Optional[int] = None
    score: float
    preview: str


class RagRequest(BaseModel):
    question: str
    k: int = 4


class RagResponse(BaseModel):
    question: str
    final_answer: str
    baseline_answer: str
    reranked_answer: str
    baseline_chunks: List[ChunkOut]
    reranked_chunks: List[ChunkOut]


# ---------- FastAPI app ----------

app = FastAPI(title="RAG + BGE Rerank API")


@app.post("/rag", response_model=RagResponse)
def rag_endpoint(payload: RagRequest) -> RagResponse:
    """
    FastAPI endpoint that wraps your rag_answer() function.
    final_answer = reranked_answer (if present), otherwise baseline.
    """
    result = rag_answer(payload.question, k=payload.k)

    # choose final answer = reranked (fallback to baseline)
    final_answer = result.get("reranked_answer") or result.get("baseline_answer", "")

    return RagResponse(
        question=result["question"],
        final_answer=final_answer,
        baseline_answer=result["baseline_answer"],
        reranked_answer=result["reranked_answer"],
        baseline_chunks=[ChunkOut(**c) for c in result["baseline_chunks"]],
        reranked_chunks=[ChunkOut(**c) for c in result["reranked_chunks"]],
    )
