"""Naija Law RAG -- FastAPI service: hardened, streaming, cited answers.

Wraps the framework-free retrieval+generation engine (../rag) in a web product:
  GET  /                UI (static chat page)
  GET  /health          liveness probe (reports the active store)
  POST /chat            one-shot cited answer         (JSON)
  POST /chat/stream     token-by-token SSE + citations

Public mode (PUBLIC_MODE=1, the default) lets anyone try the live demo with no
key, behind three locks: a per-IP rate limit (per minute and per day), a
question length cap, and a daily spending cap on estimated AI cost. A request
WITH an X-API-Key must carry the right one; it skips the per-IP limits and the
spending cap (it is the owner), but still counts toward the day's spend.

The RAG brain is unchanged; this is the body around it -- auth, rate limiting,
clean error JSON, CORS, streaming, and a UI.

The retrieval store is chosen by an env var, not hardcoded:
  ENGINE=qdrant  -> local Qdrant files in ../rag/index_data  (dev default)
  ENGINE=pg      -> pgvector over DATABASE_URL               (production)
A container's filesystem is ephemeral, so production points at an external
pgvector database; retrieval parity between the two stores is proven, so the
swap is invisible to everything above the store.

All config comes from env vars (secrets never baked into the image):
  API_KEY, ENGINE, DATABASE_URL, RATE_LIMIT, RATE_WINDOW, ALLOWED_ORIGINS,
  ANTHROPIC_API_KEY, VOYAGE_API_KEY, PUBLIC_MODE, PUBLIC_RATE_LIMIT,
  PUBLIC_RATE_WINDOW, PUBLIC_DAILY_LIMIT, DAILY_BUDGET_USD, TRUST_PROXY.

Run:  uvicorn app:app --reload          (from the app/ folder)
"""

import hashlib
import json
import os
import pathlib
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- Wire in the retrieval+generation engine (../rag) ---------------------
HERE = pathlib.Path(__file__).resolve().parent                 # .../app
ROOT = HERE.parent                                             # repo root
RAG = ROOT / "rag"
sys.path.insert(0, str(RAG))
load_dotenv(ROOT / ".env")                                     # no-op in the container

from chat import (                                             # noqa: E402
    answer, build_documents, _parse, cost_of,
    SYSTEM, MODEL, DONT_KNOW, THRESHOLD, TOP_K, COVERAGE_FLOOR,
)

# --- Pick the retrieval store from the environment ------------------------
ENGINE = os.environ.get("ENGINE", "qdrant").lower()
if ENGINE == "pg":
    from pg_engine import PgRetrievalEngine as Engine          # noqa: E402
else:
    from engine import RetrievalEngine as Engine               # noqa: E402


# --- Config (secrets from env) --------------------------------------------
API_KEY = os.environ["API_KEY"]
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "10"))
RATE_WINDOW = float(os.environ.get("RATE_WINDOW", "60"))
# Public demo: keyless access, rate-limited per IP and capped by daily spend.
PUBLIC_MODE = os.environ.get("PUBLIC_MODE", "1") == "1"
PUBLIC_RATE_LIMIT = int(os.environ.get("PUBLIC_RATE_LIMIT", "5"))        # per IP ...
PUBLIC_RATE_WINDOW = float(os.environ.get("PUBLIC_RATE_WINDOW", "60"))   # ... per N seconds
PUBLIC_DAILY_LIMIT = int(os.environ.get("PUBLIC_DAILY_LIMIT", "30"))     # per IP per 24h
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "1.00"))
MAX_QUESTION = 500
# On Render every request arrives through its proxy, so without trusting
# X-Forwarded-For all visitors would share one rate-limit bucket. A client can
# forge that header to dodge the per-IP limits, which is why the daily spending
# cap, not the rate limit, is the hard ceiling on cost. Set 0 when not behind a proxy.
TRUST_PROXY = os.environ.get("TRUST_PROXY", "1") == "1"
# Comma-separated in prod (e.g. your public URL); localhost is the dev default.
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",") if o.strip()
]


# --- Pydantic schemas -----------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION, description="A question about Nigerian law.")


class Citation(BaseModel):
    n: int
    title: str
    quote: str


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    gated: bool
    top_score: float
    coverage: float
    flagged: bool
    citations: list[Citation]


# --- Lock 1: API-key auth (optional in public mode) -----------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthError(Exception):
    """Raised when a request has no valid API key."""


class RateLimitError(Exception):
    """Raised when a caller exceeds its per-window quota."""


class BudgetError(Exception):
    """Raised when the public demo has spent today's budget."""


# --- Lock 2: in-memory rate limiter ---------------------------------------
# NOTE: correct for a single process. A multi-instance deploy would move this
# to a shared store (e.g. Redis) so the counter is shared across instances.
# IPs are hashed and only held in memory.
_hits: dict[tuple[str, str], deque[float]] = {}


def enforce_rate_limit(bucket_id: tuple[str, str], limit: int, window: float,
                       message: str) -> None:
    now = time.monotonic()
    bucket = _hits.setdefault(bucket_id, deque())
    while bucket and now - bucket[0] > window:
        bucket.popleft()
    if len(bucket) >= limit:
        raise RateLimitError(message)
    bucket.append(now)


def client_ip(request: Request) -> str:
    if TRUST_PROXY:
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# --- Lock 3: daily spending cap -------------------------------------------
class Budget:
    """Estimated USD spent on AI calls today (UTC). Resets at midnight UTC.

    In-memory: a restart resets the count, which errs on the side of serving
    visitors; the provider-side monthly limit is the hard backstop.
    """

    def __init__(self, limit):
        self.limit = limit
        self.day = None
        self.spent = 0.0

    def _roll(self):
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day, self.spent = today, 0.0

    def check(self) -> None:
        self._roll()
        if self.spent >= self.limit:
            raise BudgetError("The public demo has reached today's limit. "
                              "It resets at midnight UTC; please try again tomorrow.")

    def add(self, usd: float) -> None:
        self._roll()
        self.spent += usd


budget = Budget(DAILY_BUDGET_USD)


def guard(request: Request, key: str | None = Depends(_api_key_header)) -> None:
    """Owner (valid key) or public visitor (no key, if PUBLIC_MODE)."""
    import secrets
    if key:
        if not secrets.compare_digest(key, API_KEY):
            raise AuthError("Invalid API key.")
        enforce_rate_limit(("key", "owner"), RATE_LIMIT, RATE_WINDOW,
                           f"Rate limit exceeded: max {RATE_LIMIT} requests per "
                           f"{int(RATE_WINDOW)}s.")
        return
    if not PUBLIC_MODE:
        raise AuthError("Missing API key.")
    ip = hashlib.sha256(client_ip(request).encode()).hexdigest()[:16]
    enforce_rate_limit(("min", ip), PUBLIC_RATE_LIMIT, PUBLIC_RATE_WINDOW,
                       "You're asking quickly. Please wait a minute and try again.")
    enforce_rate_limit(("day", ip), PUBLIC_DAILY_LIMIT, 86400,
                       f"You've reached the demo limit of {PUBLIC_DAILY_LIMIT} "
                       "questions a day. Please come back tomorrow.")
    budget.check()


# --- App lifecycle --------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] retrieval store: {ENGINE}", file=sys.stderr)
    app.state.engine = Engine()
    app.state.client = anthropic.Anthropic()
    yield
    app.state.engine.close()


app = FastAPI(title="Naija Law RAG", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# --- Lock 4: structured error handlers ------------------------------------
@app.exception_handler(AuthError)
async def _auth_handler(request: Request, exc: AuthError):
    return JSONResponse(status_code=401, content={"error": str(exc)})


@app.exception_handler(RateLimitError)
async def _rate_handler(request: Request, exc: RateLimitError):
    return JSONResponse(status_code=429, content={"error": str(exc)})


@app.exception_handler(BudgetError)
async def _budget_handler(request: Request, exc: BudgetError):
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
    return JSONResponse(status_code=500, content={"error": "Internal server error."})


# --- UI routes (unauthenticated: just the page; the API it calls is not) --
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "engine": ENGINE}


# --- API (public demo or owner key) --------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _guard: None = Depends(guard)):
    result = answer(app.state.engine, app.state.client, req.question)
    budget.add(result["cost"])
    return ChatResponse(
        answer=result["text"], refused=result["refused"], gated=result["gated"],
        top_score=result["top_score"], coverage=result["coverage"],
        flagged=result["flagged"],
        citations=[Citation(**c) for c in result["citations"]],
    )


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def stream_pipeline(engine, client, query):
    passages = engine.search(query, k=TOP_K)
    top_score = passages[0][1] if passages else 0.0

    if top_score < THRESHOLD:
        yield _sse({"type": "token", "text": DONT_KNOW})
        yield _sse({"type": "done", "refused": True, "gated": True,
                    "top_score": top_score, "coverage": 0.0,
                    "flagged": False, "citations": []})
        return

    with client.messages.stream(
        model=MODEL, max_tokens=1024, system=SYSTEM,
        messages=[{
            "role": "user",
            "content": build_documents(passages) + [
                {"type": "text", "text": f"Question: {query}"}
            ],
        }],
    ) as stream:
        for text in stream.text_stream:
            yield _sse({"type": "token", "text": text})
        final = stream.get_final_message()
    budget.add(cost_of(final.usage))

    text, citations, cited_chars, total_chars = _parse(final)
    refused = text.strip().startswith(DONT_KNOW)   # sentinel may be followed by a reason
    coverage = (cited_chars / total_chars) if total_chars else 0.0
    flagged = (not refused) and coverage < COVERAGE_FLOOR
    yield _sse({"type": "done", "refused": refused, "gated": False,
                "top_score": top_score, "coverage": coverage,
                "flagged": flagged, "citations": citations})


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, _guard: None = Depends(guard)):
    gen = stream_pipeline(app.state.engine, app.state.client, req.question)
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
