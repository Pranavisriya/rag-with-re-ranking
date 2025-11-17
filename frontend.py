# # streamlit_app.py

# import requests
# import streamlit as st

# API_URL = "http://127.0.0.1:8000/rag"  # FastAPI endpoint


# st.set_page_config(
#     page_title="RAG + BGE Rerank (FastAPI + Streamlit)",
#     layout="wide",
# )

# st.title("RAG + BGE Reranking Demo")

# st.markdown(
#     """
# This app calls a **FastAPI backend** that runs your RAG pipeline with **BGE reranking**.

# - Backend: `/rag` endpoint (FastAPI)
# - Frontend: Streamlit
# - Final answer shown is the **reranked answer**.
# """
# )

# default_query = "tell me about the operation mahadev?"
# question = st.text_input("Enter your question:", value=default_query)

# k = st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=4)

# if st.button("Get Answer"):
#     if not question.strip():
#         st.warning("Please enter a question.")
#     else:
#         with st.spinner("Querying RAG backend..."):
#             try:
#                 resp = requests.post(
#                     API_URL,
#                     json={"question": question, "k": k},
#                     timeout=60,
#                 )
#                 resp.raise_for_status()
#                 data = resp.json()
#             except Exception as e:
#                 st.error(f"Error calling API: {e}")
#             else:
#                 # ---- Final answer (reranked) ----
#                 st.subheader("Final Answer (BGE-reranked)")
#                 st.write(data["final_answer"])

#                 # ---- Optional: show baseline vs reranked answers ----
#                 with st.expander("Show baseline and reranked answers"):
#                     st.markdown("**Baseline answer (Chroma only):**")
#                     st.write(data["baseline_answer"])
#                     st.markdown("**Reranked answer (BGE):**")
#                     st.write(data["reranked_answer"])

#                 # ---- Optional: chunk-level views ----
#                 col1, col2 = st.columns(2)

#                 with col1:
#                     st.markdown("### Baseline chunks (Chroma)")
#                     for ch in data["baseline_chunks"]:
#                         with st.expander(
#                             f"Rank {ch['rank']} | chunk_id={ch['chunk_id']} | score={ch['score']:.4f}"
#                         ):
#                             st.write(ch["preview"])

#                 with col2:
#                     st.markdown("### Reranked chunks (BGE)")
#                     for ch in data["reranked_chunks"]:
#                         with st.expander(
#                             f"Rank {ch['rank']} | chunk_id={ch['chunk_id']} | score={ch['score']:.4f}"
#                         ):
#                             st.write(ch["preview"])
# streamlit_app.py

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/rag"  # FastAPI endpoint

st.set_page_config(
    page_title="RAG + BGE Rerank (Final Answer)",
    layout="centered",
)

st.title("RAG + BGE Reranking")

st.markdown(
    """
Ask a question based on the indexed document.  
The backend runs RAG + BGE reranking and returns a **single final answer**.
"""
)

default_query = "tell me about the operation mahadev?"
question = st.text_input("Enter your question:", value=default_query)

# You can keep k configurable or hardcode it; keeping it here is fine for now
k = st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=4)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Querying backend..."):
            try:
                resp = requests.post(
                    API_URL,
                    json={"question": question, "k": k},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"Error calling API: {e}")
            else:
                st.subheader("Final Answer")
                st.write(data.get("final_answer", "No answer returned by backend."))
