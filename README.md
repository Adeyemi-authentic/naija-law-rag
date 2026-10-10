# Naija Law RAG

A document-intelligence assistant that answers questions about **Nigerian law**
the 1999 Constitution, federal tax legislation, and Lagos State tenancy law
**strictly from the source documents**, cites every claim to a verified span,
**refuses** what the documents do not cover, and **streams** its answer behind a
hardened API.

It is legal *information*, not legal advice.

> Built on the reusable grounded-RAG playbook (see `docs/SKILL.md`): the retrieval
> brain and the production body carry over from the reference build; the corpus,
> the gate threshold, the system prompt, and the evaluation set are new.

## What it does

- **PDF + text ingestion** — drop `.pdf`, `.txt`, or `.md` into `rag/docs/`; the
  ingester extracts, de-hyphenates, and paragraph-reflows the text before indexing.
- **Hybrid retrieval + reranking** — dense (Voyage embeddings) + BM25 keyword,
  fused by Reciprocal Rank Fusion, then re-scored by a cross-encoder reranker.
- **Confidence gate** — an out-of-scope question is refused *before* any
  generation call (no cost, no hallucination surface).
- **Cited answers** — each retrieved passage is a citable document block; the
  model API attaches the exact source span behind each claim, so citations are
  copied from the source, not typed by the model.
- **Coverage check** — flags a confident answer whose grounding is thin.
- **Streaming UI** — tokens stream live over Server-Sent Events; citations land
  when the answer completes.

## Architecture

```
INGEST (offline)   docs/*.pdf|txt|md -> clean+reflow -> smart chunk
                   -> embed (Voyage) -> Qdrant (local) / pgvector (prod)  (+ BM25)

QUERY (per request)
   question (+ history)
      -> contextualize (rewrite an elliptical follow-up into a standalone query)
      -> retrieve: dense + BM25 fused by RRF -> top-N candidates
      -> rerank (cross-encoder) -> top-k, best first
      -> confidence gate: top-1 score < THRESHOLD  ->  refuse (no model call)
      -> generate with native citations -> coverage check
      -> stream answer + verified sources
```

## Corpus

| File | Source | Notes |
|------|--------|-------|
| `constitution.txt` | 1999 Constitution of the Federal Republic of Nigeria (public interactive edition) | verbatim operative provisions; cover page + table of contents removed |
| `tax_administration_act_2025.txt` | Nigeria Tax Administration Act, 2025 (Act No. 5), FRN Official Gazette No. 117 | **verbatim** operative provisions: registration, returns, assessment, enforcement, offences and penalties |
| `lagos_tenancy_law.txt` | Lagos State Tenancy Law 2011 (No. 14), Lagos Official Gazette No. 37 | **verbatim** operative provisions: application, advance rent, notice to quit, recovery |
| `nigerian_tax.txt` | Federal tax rates: PIT, CIT, VAT | plain-language summary reflecting the **Nigeria Tax Act 2025** (in force 1 Jan 2026), which the Administration Act does not restate; figures as at 2026 |

The two Acts are verbatim, cleaned from gazette PDFs by `rag/ingest.py` +
`scripts/`-style prep (front matter, page furniture and running headers removed,
provisions reflowed from section/subsection markers). `nigerian_tax.txt` is a
short rates summary because the *substantive* rates live in the separate Nigeria
Tax Act, not the Administration Act. Drop more official PDFs into `rag/docs/` and
rebuild to extend coverage.

## Quickstart (local, Qdrant)

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY, VOYAGE_API_KEY, API_KEY
python rag/engine.py build    # chunk + embed + store docs/ (once)
python rag/tune_gate.py       # measure score bands, set THRESHOLD in chat.py
python rag/evaluate.py        # reproduce the eval numbers
uvicorn app:app --reload --app-dir app
```

Then open http://localhost:8000 and ask a question (no key needed in public mode).

## Evaluation

Deterministic gold-substring + structural eval over a labelled set of answerable
and unanswerable questions spanning all four documents (`rag/evaluate.py`):

| Metric | Result |
|--------|--------|
| Answer correctness (answerable) | 8/9 (89%) |
| Citation accuracy (verified spans) | 11/11 (100%) |
| Refusal accuracy (unanswerable) | 5/5 (100%) |
| False refusals | 1/9 |

The single false refusal is the "qualifications to be President" question: the
Constitution states the same "School Certificate" education requirement for four
different offices, and the assistant declines rather than risk attributing the
House-of-Assembly rule to the President — cautious behaviour that is the right
default for law. Rerank scores vary slightly run to run, so numbers can move by
one; delete `rag/eval_cache.json` to re-measure.

## Public demo mode

The live page is self-serve: visitors ask questions with no key. `PUBLIC_MODE=1` (the default) protects it
with three locks:

| Lock | Default | Env var |
|---|---|---|
| Questions per visitor (IP) per minute | 5 | `PUBLIC_RATE_LIMIT`, `PUBLIC_RATE_WINDOW` |
| Questions per visitor per day | 30 | `PUBLIC_DAILY_LIMIT` |
| Estimated AI spend per day, all visitors | $1.00 | `DAILY_BUDGET_USD` |

Questions are capped at 500 characters. An `X-API-Key` header with the right `API_KEY` skips the per-visitor
limits and the spending cap (owner access); a wrong key is rejected. Set `PUBLIC_MODE=0` to require the key
again. Behind Render's proxy the visitor IP comes from `X-Forwarded-For` (`TRUST_PROXY=1`, the default); the
daily spending cap is the hard ceiling on cost either way.

## Deploy (Render + managed pgvector)

Production swaps the local Qdrant files for an external pgvector store (a
container filesystem is ephemeral). The image and code are identical; only
`ENGINE=pg` and `DATABASE_URL` change.

1. Create a managed Postgres with the `vector` extension (e.g. Neon free tier);
   copy its `DATABASE_URL`.
2. Build the prod index once, locally, against that DB:
   `ENGINE=pg DATABASE_URL=... python rag/pg_engine.py build`
3. Push this repo to GitHub, create a Render **Docker** web service from it, set
   env secrets (`ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `API_KEY`, `ENGINE=pg`,
   `DATABASE_URL`, `ALLOWED_ORIGINS=https://<your-app>.onrender.com`), health
   check path `/health`. The container serves; it does not re-ingest.

Free tiers sleep when idle, so the first request after a nap is a ~30-60s cold
start — fine for a portfolio demo.

## Regenerating the corpus

The committed `rag/docs/*.txt` are cleaned from source gazette markdown by the
one-time prep scripts in `scripts/` (`prep_constitution.py`, `prep_statute.py`):
they strip front matter, page furniture and running headers, then reflow
provisions from section/subsection markers. Re-run them if you re-source a
document, then `python rag/engine.py build`.

## Stack

Generation: Claude `claude-haiku-4-5` with native Citations. Embeddings: Voyage
`voyage-3.5-lite` (1024-dim). Reranker: Voyage `rerank-2-lite`. Store: Qdrant
(local) / pgvector (prod). Serving: FastAPI + SSE, Docker, deployable to any
Docker host with a managed Postgres.

## Disclaimer

This tool provides legal *information* drawn from public legal documents. It is
not legal advice and may be incomplete or out of date. Verify against the
official gazetted text and consult a qualified legal practitioner for any
specific matter.
