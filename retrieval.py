"""
SupportPilot — Knowledge Retrieval Agent backend.

Uses TF-IDF + cosine similarity over section-level chunks of the policy
markdown files in kb/. This keeps the capstone dependency-light (no
external embedding API calls needed for retrieval itself) while still
exercising the "chunk by section, keep metadata for traceability"
pattern from Week 3.

Swap this for a real embedding model (OpenAI text-embedding-3-small,
SentenceTransformer) + FAISS/Chroma if you want to extend the capstone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = Path(__file__).parent / "kb"


@dataclass
class Chunk:
    source_doc: str
    section: str
    text: str


def _load_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for md_file in sorted(kb_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        # Split on level-2 headers ("## Section Title")
        parts = re.split(r"\n(?=## )", content)
        for part in parts:
            part = part.strip()
            if not part or part.startswith("# "):
                # capture the doc-level title line separately, skip as its own chunk
                if part.startswith("## "):
                    pass
                continue
            header_match = re.match(r"##\s+(.+)", part)
            section_title = header_match.group(1).strip() if header_match else "Introduction"
            chunks.append(Chunk(source_doc=md_file.name, section=section_title, text=part))
    return chunks


class KnowledgeRetriever:
    """Builds a TF-IDF index over KB chunks once, then answers queries."""

    def __init__(self, kb_dir: Path = KB_DIR):
        self.chunks = _load_chunks(kb_dir)
        self._texts = [c.text for c in self.chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._texts) if self._texts else None

    def retrieve(self, query: str, top_k: int = 3, exclude_internal: bool = True) -> list[dict]:
        if not self._texts:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked_idx = scores.argsort()[::-1]

        results = []
        for idx in ranked_idx:
            chunk = self.chunks[idx]
            if exclude_internal and chunk.source_doc == "escalation_criteria.md":
                # internal-only doc: usable for validation grounding, not customer answers
                continue
            if scores[idx] <= 0:
                continue
            results.append({
                "source_doc": chunk.source_doc,
                "section": chunk.section,
                "text": chunk.text,
                "relevance_score": round(float(scores[idx]), 4),
            })
            if len(results) >= top_k:
                break
        return results

    def retrieve_internal(self, query: str, top_k: int = 2) -> list[dict]:
        """Retrieval scoped to escalation_criteria.md only, for the validation/escalation agents."""
        if not self._texts:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked_idx = scores.argsort()[::-1]
        results = []
        for idx in ranked_idx:
            chunk = self.chunks[idx]
            if chunk.source_doc != "escalation_criteria.md":
                continue
            results.append({
                "source_doc": chunk.source_doc,
                "section": chunk.section,
                "text": chunk.text,
                "relevance_score": round(float(scores[idx]), 4),
            })
            if len(results) >= top_k:
                break
        return results


if __name__ == "__main__":
    retriever = KnowledgeRetriever()
    print(f"Loaded {len(retriever.chunks)} chunks from {KB_DIR}")
    for r in retriever.retrieve("customer wants a refund for a damaged item"):
        print(f"  [{r['relevance_score']}] {r['source_doc']} — {r['section']}")
