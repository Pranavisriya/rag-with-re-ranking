# from langchain_community.vectorstores import Chroma  # type: ignore
# from langchain_community.document_loaders import TextLoader  # type: ignore
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from dotenv import load_dotenv
# import os

# load_dotenv()


# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("Please set OPENAI_API_KEY or OPEN_API_KEY in your .env file")

# MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

# loader = TextLoader("data/demo.txt", encoding="utf-8")
# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=0,
# )
# texts = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=OPENAI_API_KEY,
# )

# vectordb = Chroma.from_documents(
#     texts,
#     embeddings,
# )

# llm = ChatOpenAI(
#     model=MODEL_NAME,
#     temperature=0,
#     api_key=OPENAI_API_KEY,
# )

# retriever = vectordb.as_retriever(search_kwargs={"k": 4})


# def rag_answer(question: str) -> str:
#     #docs = retriever.get_relevant_documents(question)
#     docs = retriever.invoke(question)
#     if not docs:
#         return "I couldn't find anything relevant in the indexed documents."

#     context = "\n\n---\n\n".join(d.page_content for d in docs)

#     prompt = f"""
# You are a helpful assistant that answers questions using ONLY the context below.

# Context:
# {context}

# Question: {question}

# Answer clearly and concisely based only on the context. If the answer is not in the context, say you don't know.
# """

#     resp = llm.invoke(prompt)
#     return resp.content


# if __name__ == "__main__":
#     query = "tell me about the operation mahadev?"
#     response = rag_answer(query)
#     print(response)
from langchain_community.vectorstores import Chroma  # type: ignore
from langchain_community.document_loaders import TextLoader  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder  # type: ignore
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Tuple

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY or OPEN_API_KEY in your .env file")

MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

# -----------------------------
# Load and chunk documents
# -----------------------------
loader = TextLoader("data/demo.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
)
texts = text_splitter.split_documents(documents)

# Give each chunk a stable id
for idx, doc in enumerate(texts):
    if doc.metadata is None:
        doc.metadata = {}
    doc.metadata["chunk_id"] = idx

# -----------------------------
# Vector store & LLM
# -----------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

vectordb = Chroma.from_documents(
    texts,
    embeddings,
)

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# -----------------------------
# BGE reranker (cross encoder)
# -----------------------------
# You can override this via env: BGE_RERANKER_MODEL
BGE_RERANKER_MODEL = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

reranker = HuggingFaceCrossEncoder(
    model_name=BGE_RERANKER_MODEL,
    model_kwargs={"device": "cpu"},  # change to "cuda" if you have GPU
)


# -----------------------------
# Helper functions
# -----------------------------
def baseline_retrieve(question: str, k: int = 4):
    """
    Baseline: pure vector similarity search from Chroma.
    Returns list of (Document, score).
    """
    return vectordb.similarity_search_with_score(question, k=k)


def rerank_with_bge(
    question: str,
    docs: List[Any],
) -> List[Tuple[Any, float]]:
    """
    Use BGE cross-encoder reranker to re-score the initial chunks.
    Returns list of (Document, score) sorted by score DESC.
    """
    if not docs:
        return []

    # Build (query, chunk_text) pairs
    text_pairs = [(question, d.page_content) for d in docs]
    scores = reranker.score(text_pairs)  # higher = more relevant

    # Attach scores and sort
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return doc_score_pairs


def answer_from_chunks(question: str, docs: List[Any]) -> str:
    """
    Generate an answer using ONLY the provided chunks.
    """
    if not docs:
        return "I couldn't find anything relevant in the indexed documents."

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a helpful assistant that answers questions using ONLY the context below.

Context:
{context}

Question: {question}

Answer clearly and concisely based only on the context.
If the answer is not in the context, say you don't know.
"""

    resp = llm.invoke(prompt)
    return resp.content


def rag_answer(question: str, k: int = 4) -> Dict[str, Any]:
    """
    Full pipeline:
    1) Baseline retrieval from Chroma
    2) BGE reranking over the retrieved chunks
    3) Answer from baseline & reranked chunks
    4) Return everything, including chunk-level rankings
    """
    # 1) baseline retrieval
    baseline_docs_scores = baseline_retrieve(question, k=k)
    baseline_docs = [d for d, _ in baseline_docs_scores]

    # 2) rerank with BGE
    reranked_docs_scores = rerank_with_bge(question, baseline_docs)

    # 3) answers
    baseline_answer = answer_from_chunks(question, baseline_docs)
    reranked_answer = answer_from_chunks(
        question, [d for d, _ in reranked_docs_scores]
    )

    # 4) format chunk-level outputs
    baseline_chunks = []
    for rank, (doc, score) in enumerate(baseline_docs_scores, start=1):
        baseline_chunks.append(
            {
                "rank": rank,
                "chunk_id": doc.metadata.get("chunk_id"),
                "score": score,
                "preview": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
            }
        )

    reranked_chunks = []
    for rank, (doc, score) in enumerate(reranked_docs_scores, start=1):
        reranked_chunks.append(
            {
                "rank": rank,
                "chunk_id": doc.metadata.get("chunk_id"),
                "score": score,
                "preview": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
            }
        )

    return {
        "question": question,
        "baseline_answer": baseline_answer,
        "reranked_answer": reranked_answer,
        "baseline_chunks": baseline_chunks,
        "reranked_chunks": reranked_chunks,
    }


# -----------------------------
# CLI demo
# -----------------------------
if __name__ == "__main__":
    query = "tell me about the operation mahadev?"
    result = rag_answer(query, k=4)

    print("\n================ QUESTION ================\n")
    print(result["question"])

    print("\n=========== BASELINE ANSWER =============\n")
    print(result["baseline_answer"])

    print("\n=========== RERANKED ANSWER =============\n")
    print(result["reranked_answer"])

    print("\n====== BASELINE CHUNK RANKING (Chroma) ======\n")
    for ch in result["baseline_chunks"]:
        print(
            f"Rank {ch['rank']} | chunk_id={ch['chunk_id']} | score={ch['score']:.4f}"
        )
        print(ch["preview"])
        print("-" * 80)

    print("\n====== RERANKED CHUNK RANKING (BGE) ======\n")
    for ch in result["reranked_chunks"]:
        print(
            f"Rank {ch['rank']} | chunk_id={ch['chunk_id']} | score={ch['score']:.4f}"
        )
        print(ch["preview"])
        print("-" * 80)
