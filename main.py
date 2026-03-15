#!/usr/bin/env python3
"""
FastAPI backend for the Kapilopadesha Lecture Q&A app.
Loads pre-built vector store at startup and serves Q&A via semantic search + GPT.
"""

import json
import os
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ─── Configuration ────────────────────────────────────────────────────────────
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "vector_store.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4.1-mini"
TOP_K = 5  # number of chunks to retrieve per query

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Kapilopadesha Lecture Q&A API",
    description="Ask questions about the Kapilopadesha lecture series by Swami Ishatmananda",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────
vector_store = None
embeddings_matrix = None
embedding_model = None
openai_client = None


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    video_filter: Optional[str] = None  # optional: filter by video_id


class SourceCitation(BaseModel):
    video_title: str
    timestamp: str
    youtube_url: str
    excerpt: str
    relevance_score: float


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceCitation]


class VideoInfo(BaseModel):
    video_id: str
    title: str
    url: str
    chunk_count: int
    playlist_index: int


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global vector_store, embeddings_matrix, embedding_model, openai_client

    print("Loading vector store...")
    with open(VECTOR_STORE_PATH) as f:
        vector_store = json.load(f)

    chunks = vector_store["chunks"]
    print(f"Loaded {len(chunks)} chunks from {len(vector_store['videos'])} videos")

    # Build numpy matrix for fast cosine similarity
    print("Building embeddings matrix...")
    embeddings_matrix = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_matrix = embeddings_matrix / norms
    print(f"Embeddings matrix shape: {embeddings_matrix.shape}")

    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")

    # Initialize OpenAI client (uses OPENAI_API_KEY env var)
    openai_client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    print("OpenAI client initialized.")
    print("Backend ready!")


# ─── Helper Functions ─────────────────────────────────────────────────────────
def embed_query(text: str) -> np.ndarray:
    """Embed a query string and return normalized vector."""
    vec = embedding_model.encode([text])[0].astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def semantic_search(query: str, top_k: int = TOP_K, video_filter: str = None) -> List[dict]:
    """Find top-k most relevant chunks using cosine similarity."""
    query_vec = embed_query(query)
    chunks = vector_store["chunks"]

    # Filter by video if requested
    if video_filter:
        indices = [i for i, c in enumerate(chunks) if c["video_id"] == video_filter]
        if not indices:
            indices = list(range(len(chunks)))
        filtered_matrix = embeddings_matrix[indices]
        scores = filtered_matrix @ query_vec
        top_local = np.argsort(scores)[::-1][:top_k]
        top_indices = [indices[i] for i in top_local]
        top_scores = scores[top_local]
    else:
        scores = embeddings_matrix @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]

    results = []
    for idx, score in zip(top_indices, top_scores):
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


def build_prompt(question: str, chunks: List[dict]) -> str:
    """Build the RAG prompt for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Excerpt {i} — {chunk['title']}, at {chunk['timestamp']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    return f"""You are a knowledgeable assistant helping students understand the Kapilopadesha lecture series — teachings on Sankhya philosophy and Vedanta by Swami Ishatmananda, based on the Bhagavata Purana.

Answer the question below using ONLY the provided lecture excerpts. Be precise, clear, and educational. If the excerpts do not contain enough information to answer the question, say so honestly. Always refer to specific concepts from the lectures.

LECTURE EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


# ─── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "Kapilopadesha Lecture Q&A API",
        "status": "running",
        "total_chunks": len(vector_store["chunks"]) if vector_store else 0,
        "total_videos": len(vector_store["videos"]) if vector_store else 0
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model": EMBEDDING_MODEL_NAME, "llm": LLM_MODEL}


@app.get("/videos", response_model=List[VideoInfo])
async def list_videos():
    """Return all ingested lecture videos."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not loaded")
    return vector_store["videos"]


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Answer a question using semantic search + GPT."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not loaded")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Semantic search
    relevant_chunks = semantic_search(question, top_k=TOP_K, video_filter=request.video_filter)

    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="No relevant content found")

    # Build prompt and call LLM
    prompt = build_prompt(question, relevant_chunks)

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert on Vedanta and Sankhya philosophy, specifically the Kapilopadesha teachings from the Bhagavata Purana as explained by Swami Ishatmananda."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=800,
            temperature=0.3
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Build source citations
    sources = []
    seen_chunks = set()
    for chunk in relevant_chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id not in seen_chunks:
            seen_chunks.add(chunk_id)
            sources.append(SourceCitation(
                video_title=chunk["title"],
                timestamp=chunk["timestamp"],
                youtube_url=chunk["youtube_url_with_time"],
                excerpt=chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                relevance_score=round(chunk["score"], 4)
            ))

    return AnswerResponse(
        question=question,
        answer=answer,
        sources=sources
    )


@app.get("/search")
async def search_transcripts(q: str, top_k: int = 5):
    """Raw semantic search without LLM generation."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not loaded")
    results = semantic_search(q, top_k=top_k)
    return {
        "query": q,
        "results": [
            {
                "title": r["title"],
                "timestamp": r["timestamp"],
                "url": r["youtube_url_with_time"],
                "text": r["text"][:300] + "...",
                "score": round(r["score"], 4)
            }
            for r in results
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
