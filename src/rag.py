"""RAG layer — retrieval over real discharge-guideline documents.

    python -m src.rag --build                 # guidelines/ -> chunk -> embed -> artifacts/chroma
    python -m src.rag --query "follow-up call" # sanity-check the index
    python -m src.alert                       # retrieval is wired in (use --no-rag to disable)

Design (the mentor's Week 6 spec):
- **Corpus:** ``guidelines/*.md`` — AHRQ Re-Engineered Discharge (RED) Toolkit
  documents, committed to the repo (``scripts/fetch_guidelines.py`` re-fetches
  them; each file records its source URL for citation).
- **Chunking:** ~512 tokens with ~50 tokens of overlap, split at paragraph
  boundaries first, sentence boundaries when a paragraph is too big.
- **Embeddings:** OpenAI ``text-embedding-3-small`` (needs ``OPENAI_API_KEY``).
  Vectors are always passed to Chroma explicitly — no default embedder is ever
  invoked — so tests inject a fake embedder and CI never touches the network.
- **Store:** ``artifacts/chroma`` (Chroma ``PersistentClient``, cosine space).
- **Fail loud:** querying an index that was never built stops with build
  instructions; there is no empty-result silent fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from . import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
)
logger = logging.getLogger("rag")

COLLECTION = "discharge_guidelines"
EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHARS_PER_TOKEN = 4  # English prose approximation — no tokenizer dependency
EMBED_BATCH = 100


# ------------------------------------------------------------------ chunking


def estimate_tokens(text: str) -> int:
    """Approximate token count (``len / 4``) — good enough for chunk sizing."""
    return len(text) // CHARS_PER_TOKEN


def _split_long(text: str, target: int, overlap: int) -> list[str]:
    """Split an oversized paragraph at sentence/word boundaries with overlap."""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + target, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("; "), window.rfind(" "))
            if cut > target * 0.5:
                end = start + cut + 1
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_text(text: str, chunk_tokens: int = 512, overlap_tokens: int = 50) -> list[str]:
    """Split text into ~``chunk_tokens`` chunks with ~``overlap_tokens`` overlap.

    Paragraphs are packed greedily; the tail of every chunk is carried into the
    next so context spanning a boundary is not lost. Oversized paragraphs are
    split at sentence boundaries first.
    """
    target = chunk_tokens * CHARS_PER_TOKEN
    overlap = overlap_tokens * CHARS_PER_TOKEN
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        while len(paragraph) > target:  # a paragraph larger than one chunk
            if buffer:
                chunks.append(buffer)
                buffer = ""
            pieces = _split_long(paragraph, target, overlap)
            chunks.extend(pieces[:-1])
            paragraph = pieces[-1]
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= target:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            buffer = paragraph
    if buffer:
        chunks.append(buffer)

    # Carry the overlap tail forward between packed chunks.
    if overlap > 0 and len(chunks) > 1:
        carried: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = carried[i - 1][-overlap:]
            carried.append(tail + "\n" + chunks[i] if not chunks[i].startswith(tail) else chunks[i])
        chunks = carried
    return [c for c in chunks if c.strip()]


def load_guidelines(directory: Path | None = None) -> list[dict]:
    """Read every ``guidelines/*.md`` into ``{doc, title, url, text}``."""
    directory = Path(directory or config.GUIDELINE_DIR)
    if not directory.is_dir():
        raise SystemExit(
            f"guideline directory not found: {directory}\n"
            "Run: python scripts/fetch_guidelines.py"
        )
    documents = []
    for path in sorted(directory.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title_match = re.match(r"#\s+(.+)", raw)
        url_match = re.search(r">\s*Source:\s*(\S+)", raw)
        documents.append(
            {
                "doc": path.stem,
                "title": (title_match.group(1) if title_match else path.stem).strip(),
                "url": url_match.group(1) if url_match else "",
                "text": raw,
            }
        )
    if not documents:
        raise SystemExit(
            f"no guideline documents in {directory}\n"
            "Run: python scripts/fetch_guidelines.py"
        )
    return documents


def build_chunks(directory: Path | None = None) -> list[dict]:
    """Chunk every guideline document into indexed records with stable ids."""
    records = []
    for document in load_guidelines(directory):
        for position, text in enumerate(
            chunk_text(
                document["text"],
                chunk_tokens=config.RAG_CHUNK_TOKENS,
                overlap_tokens=config.RAG_CHUNK_OVERLAP_TOKENS,
            )
        ):
            records.append(
                {
                    "id": f"{document['doc']}#{position:03d}",
                    "text": text,
                    "doc": document["doc"],
                    "title": document["title"],
                    "url": document["url"],
                    "chunk_index": position,
                    "tokens": estimate_tokens(text),
                }
            )
    return records


# ---------------------------------------------------------------- embedding


def _openai_client():
    """Build an OpenAI client, resolving the key exactly like ``src.alert``.

    Imported lazily: ``src.alert`` imports this module, so importing it back at
    module level would be circular.
    """
    from .alert import resolve_api_key

    resolve_api_key()
    from openai import OpenAI

    return OpenAI()


class OpenAIEmbedder:
    """``text-embedding-3-small`` embeddings, batched. The only online piece."""

    def __init__(self, model: str = EMBEDDING_MODEL) -> None:
        self.model = model
        self._client = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            self._client = _openai_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            response = self._client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors


# --------------------------------------------------------------------- store


def _client(chroma_dir: Path | None = None):
    import chromadb

    path = Path(chroma_dir or config.CHROMA_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def build_index(
    directory: Path | None = None,
    chroma_dir: Path | None = None,
    embedder=None,
) -> dict:
    """Chunk + embed + write the index. Rebuilds from scratch every time."""
    records = build_chunks(directory)
    embedder = embedder or OpenAIEmbedder()
    logger.info("embedding %d chunks from %s", len(records), directory or config.GUIDELINE_DIR)
    vectors = embedder.embed([r["text"] for r in records])

    client = _client(chroma_dir)
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001 - collection simply does not exist yet
        pass
    collection = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[r["id"] for r in records],
        embeddings=vectors,
        documents=[r["text"] for r in records],
        metadatas=[
            {k: r[k] for k in ("doc", "title", "url", "chunk_index", "tokens")}
            for r in records
        ],
    )
    summary = {
        "documents": len({r["doc"] for r in records}),
        "chunks": len(records),
        "collection": COLLECTION,
        "embedding_model": getattr(embedder, "model", "custom"),
        "path": str(Path(chroma_dir or config.CHROMA_DIR)),
    }
    logger.info("index built: %s", summary)
    return summary


def require_index(chroma_dir: Path | None = None) -> None:
    """Fail loudly (before any OpenAI call) if the index was never built."""
    path = Path(chroma_dir or config.CHROMA_DIR)
    if not path.is_dir():
        raise SystemExit(
            f"RAG index not found at {path}\n"
            "Build it first:\n"
            "  python -m src.rag --build\n"
            "or run the alert stage with --no-rag (no guideline grounding)."
        )
    collection = _client(chroma_dir).get_or_create_collection(COLLECTION)
    if collection.count() == 0:
        raise SystemExit(
            f"RAG index at {path} is empty\n"
            "Rebuild it:\n"
            "  python -m src.rag --build"
        )


def retrieve(
    query: str,
    k: int = config.RAG_TOP_K,
    chroma_dir: Path | None = None,
    embedder=None,
) -> list[dict]:
    """Top-``k`` guideline chunks for ``query``, ranked by cosine similarity.

    Returns records shaped ``{rank, id, text, doc, title, url, score}`` where
    ``rank`` (1-based) is what callers turn into ``[S1]``, ``[S2]``, ... labels.
    """
    require_index(chroma_dir)
    embedder = embedder or OpenAIEmbedder()
    vector = embedder.embed([query])[0]
    collection = _client(chroma_dir).get_or_create_collection(COLLECTION)
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for rank, (chunk_id, text, meta, distance) in enumerate(
        zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ),
        start=1,
    ):
        hits.append(
            {
                "rank": rank,
                "id": chunk_id,
                "text": text,
                "doc": meta.get("doc", ""),
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                # cosine distance -> similarity in [0, 1]
                "score": round(1.0 - float(distance), 4),
            }
        )
    logger.info("retrieved %d chunks for %r", len(hits), query[:80])
    return hits


# ----------------------------------------------------------------------- cli


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or query the guideline RAG index")
    parser.add_argument("--build", action="store_true", help="chunk + embed guidelines/ into artifacts/chroma")
    parser.add_argument("--query", help="run a query against the index")
    parser.add_argument("--k", type=int, default=config.RAG_TOP_K, help=f"top-k (default {config.RAG_TOP_K})")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    args = parser.parse_args()

    if args.build:
        summary = build_index()
        print(json.dumps(summary, indent=2))
        return 0

    if args.query:
        hits = retrieve(args.query, k=args.k)
        if args.json:
            print(json.dumps(hits, indent=2))
        else:
            for hit in hits:
                print(f"[S{hit['rank']}] {hit['title']} (score {hit['score']})")
                print("      " + hit["text"][:200].replace("\n", " ") + "...")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
