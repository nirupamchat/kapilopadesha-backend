"""
FastAPI backend for the Kapilopadesha Lecture Q&A app.
Uses Google Gemini for BOTH embeddings and answer generation.
No local ML models — starts instantly so Render can detect the port.
"""

import json
import os
import asyncio
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai as google_genai
from google.genai import types as genai_types

# ─── Configuration ────────────────────────────────────────────────────────────
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "vector_store.json")
EMBEDDING_MODEL = "models/gemini-embedding-001"   # Gemini embedding model
LLM_MODEL       = "gemini-2.5-flash"
TOP_K = 5

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Kapilopadesha Lecture Q&A API",
    description="Ask questions about the Kapilopadesha lecture series by Swami Ishatmananda",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────
vector_store      = None
embeddings_matrix = None
gemini_client     = None
_ready            = False   # set to True once everything is loaded


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    video_filter: Optional[str] = None


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


# ─── Background Loader ────────────────────────────────────────────────────────
async def _load_resources():
    """Load vector store and init Gemini client in the background."""
    global vector_store, embeddings_matrix, gemini_client, _ready

    print("Loading vector store...")
    with open(VECTOR_STORE_PATH) as f:
        vector_store = json.load(f)

    chunks = vector_store["chunks"]
    print(f"Loaded {len(chunks)} chunks from {len(vector_store['videos'])} videos")

    # Build numpy matrix for fast cosine similarity
    print("Building embeddings matrix...")
    embeddings_matrix = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_matrix = embeddings_matrix / norms
    print(f"Embeddings matrix shape: {embeddings_matrix.shape}")

    # Initialize Gemini client
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("WARNING: GEMINI_API_KEY not set — Q&A will not work!")
    else:
        gemini_client = google_genai.Client(api_key=gemini_api_key)
        print("Gemini client initialized.")

    _ready = True
    print("Backend fully ready!")


@app.on_event("startup")
async def startup_event():
    """
    Start the background loader WITHOUT awaiting it.
    This lets uvicorn bind the port immediately, satisfying Render's health check,
    while the vector store and Gemini client load in the background.
    """
    asyncio.create_task(_load_resources())
    print("Server started — loading resources in background...")


# ─── Helper Functions ─────────────────────────────────────────────────────────
def embed_query(text: str) -> np.ndarray:
    """Embed a query string using Gemini and return a normalized vector."""
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    vec = np.array(response.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def semantic_search(query: str, top_k: int = TOP_K, video_filter: Optional[str] = None) -> List[dict]:
    """Find the most semantically similar chunks to the query."""
    chunks = vector_store["chunks"]
    query_vec = embed_query(query)

    if video_filter:
        indices = [i for i, c in enumerate(chunks) if c["video_id"] == video_filter]
        if not indices:
            return []
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
        "status": "ready" if _ready else "loading",
        "total_chunks": len(vector_store["chunks"]) if vector_store else 0,
        "total_videos": len(vector_store["videos"]) if vector_store else 0
    }


@app.get("/health")
async def health():
    return {
        "status": "ready" if _ready else "loading",
        "llm": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "gemini_ready": gemini_client is not None
    }


@app.get("/videos", response_model=List[VideoInfo])
async def list_videos():
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store still loading, please retry in a moment")
    return vector_store["videos"]


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not _ready:
        raise HTTPException(status_code=503, detail="Backend still loading, please retry in a moment")
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client not initialized. Please set GEMINI_API_KEY.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Semantic search using Gemini embeddings
    relevant_chunks = semantic_search(question, top_k=TOP_K, video_filter=request.video_filter)
    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="No relevant content found")

    # Build prompt and call Gemini
    prompt = build_prompt(question, relevant_chunks)
    try:
        response = gemini_client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=800,
                temperature=0.3,
            )
        )
        answer = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

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
    if not _ready:
        raise HTTPException(status_code=503, detail="Backend still loading, please retry in a moment")
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
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
