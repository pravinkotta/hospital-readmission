"""RAG layer tests — everything runs offline (no OpenAI, no network).

The index round-trip injects a deterministic fake embedder, so CI can build and
query a throwaway Chroma store without any credentials. These tests guard the
three things that would silently break grounding: chunking, the [Sx] prompt
contract, and the fail-loud behaviour when the index is missing.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import pytest

from src import config, rag
from src.alert import SYSTEM_PROMPT, build_messages, retrieval_query

ROOT = Path(__file__).resolve().parents[1]


class FakeEmbedder:
    """Deterministic bag-of-words embeddings — no network, no model files."""

    dim = 64
    model = "fake-bow"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                vector[zlib.crc32(token.encode()) % self.dim] += 1.0
            vectors.append(vector)
        return vectors


# ------------------------------------------------------------------ chunking


def test_chunking_respects_size_and_overlap() -> None:
    text = "\n\n".join(f"Paragraph {i}: " + ("discharge follow-up content. " * 30) for i in range(12))
    target = config.RAG_CHUNK_TOKENS * rag.CHARS_PER_TOKEN
    overlap = config.RAG_CHUNK_OVERLAP_TOKENS * rag.CHARS_PER_TOKEN

    chunks = rag.chunk_text(text, config.RAG_CHUNK_TOKENS, config.RAG_CHUNK_OVERLAP_TOKENS)

    assert len(chunks) > 1, "a 12-paragraph document must produce multiple chunks"
    for chunk in chunks:
        assert chunk.strip(), "no empty chunks"
        # the overlap tail is carried forward, so one chunk may exceed target
        assert len(chunk) <= target + overlap + 16
    # consecutive chunks share context at the boundary (the whole point of overlap)
    for previous, current in zip(chunks, chunks[1:]):
        tail = previous[-overlap:]
        assert current.startswith(tail[:20]), "chunks must overlap at the boundary"


def test_chunking_covers_long_unbroken_paragraphs() -> None:
    """A single paragraph longer than one chunk still splits with overlap."""
    text = "Sentence about follow-up calls. " * 500  # one huge paragraph
    chunks = rag.chunk_text(text, chunk_tokens=512, overlap_tokens=50)

    assert len(chunks) >= 2
    assert all(c.strip() for c in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        shared = previous[-80:]
        assert shared in current, "sentence-level split must keep the overlap"


def test_guidelines_produce_indexed_chunks() -> None:
    records = rag.build_chunks()
    assert len(records) >= 10, "the AHRQ corpus should yield at least 10 chunks"

    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids), "chunk ids must be unique (Chroma keys)"
    assert all(r["url"].startswith("https://") for r in records), "every chunk cites its source URL"
    assert all(r["title"] for r in records)
    assert all(r["tokens"] > 0 for r in records)
    assert {r["doc"] for r in records} >= {"ahrq-red-tool-1-overview"}


# --------------------------------------------------------------- index roundtrip


def make_corpus(tmp_path: Path) -> Path:
    guidelines = tmp_path / "guidelines"
    guidelines.mkdir()
    (guidelines / "followup-guidance.md").write_text(
        "# Follow-Up Call Guidance\n\n"
        "> Source: https://example.org/followup\n\n"
        "A postdischarge telephone call within 3 business days confirms "
        "medications were obtained and symptoms are controlled.\n",
        encoding="utf-8",
    )
    (guidelines / "medication-guidance.md").write_text(
        "# Medication Reconciliation\n\n"
        "> Source: https://example.org/meds\n\n"
        "Medication reconciliation compares the discharge list against what "
        "the patient was taking before admission.\n",
        encoding="utf-8",
    )
    return guidelines


def test_index_roundtrip_with_fake_embedder(tmp_path: Path) -> None:
    guidelines = make_corpus(tmp_path)
    chroma_dir = tmp_path / "chroma"

    summary = rag.build_index(directory=guidelines, chroma_dir=chroma_dir, embedder=FakeEmbedder())
    assert summary["chunks"] == 2
    assert summary["embedding_model"] == "fake-bow"

    hits = rag.retrieve(
        "postdischarge telephone follow-up call",
        k=1,
        chroma_dir=chroma_dir,
        embedder=FakeEmbedder(),
    )
    assert len(hits) == 1
    hit = hits[0]
    assert hit["rank"] == 1
    assert hit["id"].startswith("followup-guidance#")
    assert "telephone call" in hit["text"]
    assert hit["url"] == "https://example.org/followup"
    assert 0.0 <= hit["score"] <= 1.0


# ------------------------------------------------------------------ fail loud


def test_retrieve_without_index_stops_with_instructions(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        rag.retrieve("anything", chroma_dir=tmp_path / "never-built")
    assert "python -m src.rag --build" in str(excinfo.value)


def test_empty_index_stops_with_rebuild_instructions(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)  # directory exists but nothing was indexed
    with pytest.raises(SystemExit) as excinfo:
        rag.require_index(chroma_dir=tmp_path)
    assert "empty" in str(excinfo.value)


def test_missing_guidelines_directory_stops(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        rag.build_chunks(directory=tmp_path / "nope")
    assert "fetch_guidelines" in str(excinfo.value)


# -------------------------------------------------------------- prompt contract


def make_case() -> dict:
    return {
        "probability": 0.71,
        "threshold": 0.57,
        "flagged": True,
        "risk_band": "High",
        "recommended_action": "Escalate: transition-of-care follow-up",
        "facts": {
            "primary_diagnosis": "circulatory",
            "prior_inpatient_visits": 3,
            "discharge_destination": "Home with home health",
            "age_band": "70-80",
        },
        "drivers": [
            {"feature": "number_inpatient", "description": "inpatient visits in the prior 6 months: 3 (SHAP +0.81 log-odds)", "contribution": 0.8123, "value": 3}
        ],
        "guideline_chunks": [
            {"rank": 1, "id": "ahrq-red-tool-5-followup#002", "title": "AHRQ RED Toolkit — Tool 5", "url": "https://example.org/t5", "text": "Call the patient within 3 business days."},
            {"rank": 2, "id": "ahrq-red-tool-1-overview#000", "title": "AHRQ RED Toolkit — Tool 1", "url": "https://example.org/t1", "text": "The RED has 12 mutually reinforcing actions."},
        ],
    }


def test_system_prompt_requires_five_sections_with_sources() -> None:
    sections = re.findall(r"^\s*## (.+)$", SYSTEM_PROMPT, flags=re.M)
    assert sections == [
        "Risk summary",
        "Why the model flagged this patient",
        "Suggested discharge actions",
        "Sources",
        "Caveat",
    ]
    assert "[S1]" in SYSTEM_PROMPT, "the prompt must teach the citation labels"
    assert "insufficient guidance" in SYSTEM_PROMPT, "the no-chunk escape hatch is required"


def test_build_messages_grounds_actions_in_retrieved_chunks() -> None:
    case = make_case()
    messages = build_messages(case)
    assert [m["role"] for m in messages] == ["system", "user"]

    user = messages[1]["content"]
    payload = user.split("\n\n", 1)[1]
    assert '"retrieved_guideline_chunks"' in payload
    assert '"label": "[S1]"' in payload
    assert '"label": "[S2]"' in payload
    assert "Call the patient within 3 business days." in payload
    assert "cite them as [S1], [S2]" in user


def test_build_messages_without_chunks_still_builds() -> None:
    case = make_case()
    case.pop("guideline_chunks")
    messages = build_messages(case)
    assert '"retrieved_guideline_chunks": []' in messages[1]["content"]


def test_retrieval_query_carries_patient_context() -> None:
    query = retrieval_query(make_case())
    assert "High" in query
    assert "circulatory" in query
    assert "3 prior inpatient visits" in query
