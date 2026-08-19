"""HybridMemoryAgent — POC combining three memories (bonus).

Combines:
  - Episodic memory (Qdrant vector store, per-user filter)
  - Stable user profile (Feast feature store)
  - Recent activity (Feast streaming feature view)

Design notes live in ARCHITECTURE.md. The code below is intentionally
linear — clarity over cleverness.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

from app.embeddings import Embedder
from app.metadata import enrich

EPISODIC_COLLECTION = "bonus_memory"
USER_PAYLOAD_KEY = "user_id"


@dataclass
class MemoryHit:
    text: str
    score: float
    user_id: str


@dataclass
class RecallContext:
    user_id: str
    query: str
    profile: dict
    recent_activity: dict
    hits: list[MemoryHit]
    assembled: str
    latency_ms: float


class HybridMemoryAgent:
    """Tiny POC: chunk per message, embed, store per-user, retrieve + profile."""

    def __init__(self, dim: int) -> None:
        self.embedder = Embedder()
        self.client = QdrantClient(":memory:")
        self.dim = dim
        # Per-user collection is overkill; one collection with a user_id
        # payload + payload index scales further and keeps ops simple.
        if EPISODIC_COLLECTION in {c.name for c in self.client.get_collections().collections}:
            self.client.delete_collection(EPISODIC_COLLECTION)
        self.client.create_collection(
            collection_name=EPISODIC_COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        # Local in-memory Qdrant ignores payload indexes but warns loudly
        # every time. Silenced here — real Qdrant server DOES use them
        # for per-user filtered search at scale.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.client.create_payload_index(
                EPISODIC_COLLECTION, USER_PAYLOAD_KEY,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        # Stand-in profile facts — a real impl would query Feast online store.
        self._profiles = {
            "u_001": {
                "preferred_language": "vi",
                "reading_speed_wpm": 220,
                "topic_affinity": "cloud",
                "active_hours_local": "18-23",
            },
            "u_002": {
                "preferred_language": "en",
                "reading_speed_wpm": 280,
                "topic_affinity": "ai_ml",
                "active_hours_local": "9-17",
            },
        }
        # Stand-in recent activity — streaming feature view lives in Feast.
        self._recent = {
            "u_001": {"queries_last_hour": 4, "last_topic": "kubernetes"},
            "u_002": {"queries_last_hour": 1, "last_topic": "transformer"},
        }

    # ── write side ───────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Chunk = 1 message. Embed. Upsert with user_id payload."""
        vec = next(self.embedder.embed([text]))
        self.client.upsert(
            collection_name=EPISODIC_COLLECTION,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=np.asarray(vec, dtype=np.float32).tolist(),
                payload={"user_id": user_id, "text": text, "ts": time.time()},
            )],
        )

    # ── read side ────────────────────────────────────────────────────────
    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> RecallContext:
        """Three-step: vector top-K (filtered by user_id) + profile + recent.
        Return assembled context string, ready for LLM prompt.
        """
        t0 = time.perf_counter()
        qv = next(self.embedder.embed([query]))
        hits_pts = self.client.query_points(
            collection_name=EPISODIC_COLLECTION,
            query=np.asarray(qv, dtype=np.float32).tolist(),
            query_filter=models.Filter(must=[models.FieldCondition(
                key=USER_PAYLOAD_KEY, match=models.MatchValue(value=user_id))]),
            limit=top_k,
        ).points
        hits = [MemoryHit(text=p.payload["text"], score=float(p.score),
                          user_id=p.payload["user_id"]) for p in hits_pts]

        profile = self._profiles.get(user_id, {})
        recent = self._recent.get(user_id, {})
        assembled = self._assemble(query, user_id, profile, recent, hits)
        return RecallContext(
            user_id=user_id, query=query, profile=profile,
            recent_activity=recent, hits=hits, assembled=assembled,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── helpers ──────────────────────────────────────────────────────────
    def _assemble(self, query: str, user_id: str, profile: dict,
                  recent: dict, hits: list[MemoryHit]) -> str:
        """Return the prompt block an LLM would consume.

        Profile first — sets the persona. Recent activity next — sets context.
        Episodic hits last — sets evidence. This order beats the reverse:
        Claude/GPT compress the tail of long prompts, so evidence must be
        concrete and short.
        """
        lines = [
            f"User profile: language={profile.get('preferred_language')}, "
            f"reading_speed={profile.get('reading_speed_wpm')}wpm, "
            f"topic_affinity={profile.get('topic_affinity')}, "
            f"active_hours={profile.get('active_hours_local')}.",
            f"Recent activity: {recent.get('queries_last_hour', 0)} queries last hour, "
            f"last topic={recent.get('last_topic')}.",
            f"User question: {query}",
            "Relevant memories (most relevant first):",
        ]
        for i, h in enumerate(hits, 1):
            short = h.text if len(h.text) < 120 else h.text[:117] + "..."
            lines.append(f"  [{i}] (score={h.score:.3f}) {short}")
        if not hits:
            lines.append("  (no prior memory for this user — answer as fresh conversation)")
        return "\n".join(lines)
