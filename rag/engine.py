"""Semantic search engine over company documents (Week-2 capstone).

End-to-end retrieval pipeline:

    ingest docs -> smart chunk -> embed (Voyage) -> store (Qdrant local)
                -> hybrid retrieve (BM25 + dense, fused by RRF)
                -> rerank (Voyage cross-encoder) -> top passages with source + score

This is the retrieval CORE that a RAG app (generation on top) will wrap later.
Self-contained: needs only voyage_client.py, qdrant-client, rank-bm25, numpy, and
a VOYAGE_API_KEY in a .env file.

CLI:
    python engine.py build        # chunk+embed+store the docs/ folder (once)
    python engine.py "my question"   # search and print reranked top passages
"""

import pathlib
import re
import sys

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from voyage_client import embed, rerank
from ingest import load_documents

DOCS_DIR = HERE / "docs"
STORE_PATH = HERE / "index_data"
COLLECTION = "naija_law"
VECTOR_SIZE = 1024          # voyage-3.5-lite
# Statutory text is dense and provision-structured. 600 keeps most sections /
# subsections intact as one chunk (the natural retrieval unit for law) while
# staying small enough that the embedding signal is not diluted. Retune per D3.
CHUNK_SIZE = 600
OVERLAP = 0.15
RRF_K = 60


# --- ingest -----------------------------------------------------------------

def _split_long(para, size):
    """Break a single paragraph longer than `size` on sentence boundaries.

    A statutory subsection can run to thousands of characters; without this a
    single paragraph would refuse to split (the packer only breaks BETWEEN
    paragraphs) and become one oversized chunk. Sentence-split first, then
    hard-cut any lone sentence that is still absurdly long.
    """
    if len(para) <= size:
        return [para]
    sentences = re.split(r"(?<=[.;:])\s+", para)
    pieces, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > size:
            pieces.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s) if cur else s
    if cur.strip():
        pieces.append(cur.strip())
    out = []
    for p in pieces:
        while len(p) > size * 1.5:
            # Cut at the last space before `size` so no word is severed.
            cut = p.rfind(" ", 0, size)
            if cut <= 0:
                cut = size
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p.strip():
            out.append(p)
    return out


_SENT_BOUNDARY = re.compile(r"(?<=[.;:])\s+|\n\n")


def _overlap_tail(text, budget):
    """The longest suffix of `text` within `budget` chars that starts cleanly.

    Prefers whole trailing sentences; if even the last sentence is longer than
    the budget, falls back to whole trailing words. A raw character slice would
    start the next chunk mid-word ("onths notice..."), and that fragment is
    what a citation quotes back to the user.
    """
    if budget <= 0 or not text:
        return ""
    starts = [m.end() for m in _SENT_BOUNDARY.finditer(text)]
    for s in starts:
        if len(text) - s <= budget:
            return text[s:].strip()
    for m in re.finditer(r"\s+", text):
        if len(text) - m.end() <= budget:
            return text[m.end():].strip()
    return ""


def smart_chunks(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Paragraph-aware chunking with ~overlap carry-over (see Day 3).

    Oversized single paragraphs are pre-split on sentence boundaries so one long
    provision cannot become a single giant chunk.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        for piece in _split_long(para, size):
            if current and len(current) + len(piece) + 2 > size:
                chunks.append(current.strip())
                tail = _overlap_tail(current, int(size * overlap))
                current = (tail + "\n\n" + piece) if tail else piece
            else:
                current = (current + "\n\n" + piece) if current else piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def tokenize(text):
    """Lowercase alphanumeric tokens; keeps codes/IDs (e.g. 0x0000011b) whole."""
    return re.findall(r"[a-z0-9]+", text.lower())


# --- engine -----------------------------------------------------------------

class RetrievalEngine:
    def __init__(self, store_path=STORE_PATH, collection=COLLECTION):
        self.client = QdrantClient(path=str(store_path))
        self.collection = collection
        self._ids = self._sources = self._texts = self._bm25 = None

    def close(self):
        self.client.close()

    # ingest --------------------------------------------------------------
    def build(self, docs_dir=DOCS_DIR):
        """Chunk + embed + store every .txt in docs_dir. Idempotent."""
        if self.client.collection_exists(self.collection):
            n = self.client.count(self.collection).count
            print(f"Index already built: '{self.collection}' has {n} chunks.")
            return

        records = []   # (source, chunk_text)
        docs = list(load_documents(docs_dir))
        for name, text in docs:
            for chunk in smart_chunks(text):
                records.append((name, chunk))

        texts = [c for _s, c in records]
        print(f"Embedding {len(texts)} chunks from {len(docs)} docs...")
        vectors, tokens = embed(texts, input_type="document")
        print(f"  embedded ({tokens} tokens).")

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        self.client.upsert(self.collection, points=[
            models.PointStruct(id=i, vector=v, payload={"source": s, "chunk_text": c})
            for i, ((s, c), v) in enumerate(zip(records, vectors))
        ])
        print(f"Stored {len(records)} chunks in '{self.collection}'.")

    # corpus cache --------------------------------------------------------
    def _load(self):
        if self._ids is not None:
            return
        pts, _ = self.client.scroll(self.collection, limit=10000, with_payload=True)
        pts.sort(key=lambda p: p.id)
        self._ids = [p.id for p in pts]
        self._sources = {p.id: p.payload["source"] for p in pts}
        self._texts = {p.id: p.payload["chunk_text"] for p in pts}
        self._bm25 = BM25Okapi([tokenize(self._texts[i]) for i in self._ids])

    def source(self, pid):
        self._load(); return self._sources[pid]

    def text(self, pid):
        self._load(); return self._texts[pid]

    # retrieval stages ----------------------------------------------------
    def _dense(self, query):
        qv, _ = embed([query], input_type="query")
        res = self.client.query_points(self.collection, query=qv[0], limit=10000)
        return [h.id for h in res.points]

    def _bm25_rank(self, query):
        self._load()
        scores = self._bm25.get_scores(tokenize(query))
        return [pid for pid, _ in sorted(zip(self._ids, scores), key=lambda p: -p[1])]

    def hybrid(self, query, top_n=10):
        """Stage 1: dense + BM25 fused by RRF; return top_n point ids."""
        self._load()
        rankings = [self._dense(query), self._bm25_rank(query)]
        fused = {}
        for ranking in rankings:
            for rank, pid in enumerate(ranking, start=1):
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (RRF_K + rank)
        ordered = [pid for pid, _ in sorted(fused.items(), key=lambda x: -x[1])]
        return ordered[:top_n]

    def search(self, query, top_n=25, k=5, do_rerank=True):
        """Full pipeline. Returns [(id, score, source, text), ...] best-first.

        score is the reranker's relevance (do_rerank=True) or the stage-1 rank
        position's RRF order (do_rerank=False, score is just descending rank).
        """
        self._load()
        candidates = self.hybrid(query, top_n=top_n)
        if not do_rerank:
            return [(pid, None, self._sources[pid], self._texts[pid]) for pid in candidates[:k]]

        docs = [self._texts[pid] for pid in candidates]
        reranked = rerank(query, docs, top_k=k)
        return [
            (candidates[r["index"]], r["relevance_score"],
             self._sources[candidates[r["index"]]], self._texts[candidates[r["index"]]])
            for r in reranked
        ]


def _flat(text, n=110):
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "..."


def main():
    args = sys.argv[1:]
    engine = RetrievalEngine()
    try:
        if not args or args[0] == "build":
            engine.build()
            if not args:
                print('\nUsage: python engine.py "your question"')
            return

        query = " ".join(args)
        print(f'Query: "{query}"\n')
        for rank, (pid, score, source, text) in enumerate(engine.search(query), 1):
            print(f"#{rank}  rel={score:0.4f}  [{source}]")
            print(f"    {_flat(text)}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
