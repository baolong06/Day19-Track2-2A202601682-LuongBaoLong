"""Demo 5 queries for the HybridMemoryAgent.

Run:
    python bonus/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root on path so `from bonus.agent import ...` works from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonus.agent import HybridMemoryAgent


def seed(agent: HybridMemoryAgent) -> None:
    """Seed episodic memory for two users. Order matters — last writes win
    for the same paraphrase, so we keep the list varied."""
    memories = [
        ("u_001", "Kubernetes HPA tự động scale pod theo CPU usage"),
        ("u_001", "Cách cấu hình cluster autoscaler trên GKE"),
        ("u_001", "Tôi đã đọc docs về blue-green deployment cho cloud"),
        ("u_001", "Họp với team ngày mai về cost optimization"),
        ("u_001", "Tự động mở rộng hạ tầng theo lưu lượng — auto-scaling group"),
        ("u_002", "Fine-tuning BERT for Vietnamese sentiment analysis"),
        ("u_002", "Transformer attention mechanism explained"),
        ("u_002", "Reading about LoRA parameter-efficient fine-tuning"),
    ]
    for uid, text in memories:
        agent.remember(text, user_id=uid)


def run_queries(agent: HybridMemoryAgent) -> None:
    demos = [
        ("u_001", "Tôi đã đọc gì về Kubernetes?",
         "Vector hit only — paraphrase `tự động mở rộng` matches HPA."),
        ("u_001", "Recommend tôi đọc gì tiếp theo?",
         "Profile context — `topic_affinity=cloud` primes the LLM."),
        ("u_001", "Tôi đang quan tâm gì gần đây?",
         "Fresh activity — `queries_last_hour`, `last_topic`."),
        ("u_001", "Tài liệu về tự động mở rộng hạ tầng?",
         "Paraphrase — must come from vector (no keyword match)."),
        ("u_001", "Cho tôi summary cloud security",
         "Mixed — vector + profile (topic_affinity=cloud)."),
    ]
    for uid, query, note in demos:
        print("=" * 72)
        print(f"User: {uid}  Query: {query}")
        print(f"  Note: {note}")
        ctx = agent.recall(query, user_id=uid)
        print(f"  Latency: {ctx.latency_ms:.1f}ms  Hits: {len(ctx.hits)}")
        print("  ── assembled context ──")
        for line in ctx.assembled.split("\n"):
            print(f"  {line}")
    print("=" * 72)


def main() -> None:
    print("Loading embedder + building in-memory Qdrant collection...")
    agent = HybridMemoryAgent(dim=384)  # matches bge-small-en
    seed(agent)
    print(f"Seeded 8 memories across 2 users.\n")
    run_queries(agent)
    print("\nOK — demo.py exits clean.")


if __name__ == "__main__":
    main()
