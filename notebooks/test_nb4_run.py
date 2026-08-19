# Test NB4 - Feast Feature Store (direct import)
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl
import pandas as pd

REPO_ROOT = ROOT
FEAST_DIR = REPO_ROOT / "app" / "feast_repo"
FEAST_DATA = FEAST_DIR / "data"
FEAST_DATA.mkdir(exist_ok=True)

NOW = datetime.now(timezone.utc).replace(microsecond=0)

# 1. Generate Parquet files
print("--- Generating Parquet files ---")

def make_user_profile(n_users: int = 100) -> pl.DataFrame:
    return pl.DataFrame({
        "user_id": [f"u_{i:03d}" for i in range(n_users)],
        "reading_speed_wpm": [180 + (i * 7) % 200 for i in range(n_users)],
        "preferred_language": ["vi" if i % 3 != 0 else "en" for i in range(n_users)],
        "topic_affinity": [
            ["ai_ml", "cloud", "security", "database", "devops"][i % 5]
            for i in range(n_users)
        ],
        "event_timestamp": [NOW - timedelta(hours=i % 48) for i in range(n_users)],
    })

def make_item_popularity(n_items: int = 1000) -> pl.DataFrame:
    return pl.DataFrame({
        "doc_id": [f"item_{i:04d}" for i in range(n_items)],
        "click_count_24h": [(i * 13) % 500 for i in range(n_items)],
        "ctr_7d": [round(((i * 7) % 100) / 100.0, 3) for i in range(n_items)],
        "avg_dwell_seconds": [10.0 + (i * 0.7) % 90 for i in range(n_items)],
        "event_timestamp": [NOW - timedelta(minutes=i % 720) for i in range(n_items)],
    })

def make_query_velocity(n_users: int = 100) -> pl.DataFrame:
    return pl.DataFrame({
        "user_id": [f"u_{i:03d}" for i in range(n_users)],
        "queries_last_hour": [(i * 11) % 50 for i in range(n_users)],
        "distinct_topics_24h": [1 + (i * 3) % 10 for i in range(n_users)],
        "event_timestamp": [NOW - timedelta(minutes=i % 30) for i in range(n_users)],
    })

make_user_profile().write_parquet(FEAST_DATA / "user_profile.parquet")
make_item_popularity().write_parquet(FEAST_DATA / "item_popularity.parquet")
make_query_velocity().write_parquet(FEAST_DATA / "query_velocity.parquet")
print(f"Wrote 3 Parquet sources to {FEAST_DATA}")
for p in sorted(FEAST_DATA.glob("*.parquet")):
    print(f"  {p.name}  {p.stat().st_size/1024:.1f} KB")

# 2. Run feast commands using subprocess with proper env
import subprocess
import os

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

print("\n--- feast apply ---")
result = subprocess.run(
    [sys.executable, "-m", "feast", "apply"],
    cwd=str(FEAST_DIR),
    capture_output=True, text=True, env=env,
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print(f"Return code: {result.returncode}")
if result.returncode == 0:
    print("NB4 Part 1: PASS - feast apply succeeded")

# 3. feast materialize
print("\n--- feast materialize-incremental ---")
end_dt = NOW.strftime("%Y-%m-%dT%H:%M:%S")
result = subprocess.run(
    [sys.executable, "-m", "feast", "materialize-incremental", end_dt],
    cwd=str(FEAST_DIR),
    capture_output=True, text=True, env=env,
)
print(result.stdout[-1500:] if result.stdout else "No stdout")
if result.stderr:
    print("STDERR (tail):", result.stderr[-500:])
print(f"Return code: {result.returncode}")
if result.returncode == 0:
    print("NB4 Part 2: PASS - materialize succeeded")

# 4. Online lookup - use the venv python directly
print("\n--- Online Lookup Test ---")
from feast import FeatureStore

fs = FeatureStore(repo_path=str(FEAST_DIR))

REQUEST_FEATURES = [
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "user_profile_features:topic_affinity",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]

# Single lookup
t0 = time.perf_counter()
features = fs.get_online_features(
    features=REQUEST_FEATURES,
    entity_rows=[{"user_id": "u_001"}],
).to_dict()
single_latency_ms = (time.perf_counter() - t0) * 1000
print(f"Single lookup: {single_latency_ms:.2f}ms")
print({k: v[0] for k, v in features.items()})

# 5. Batch latency benchmark
print("\n--- Batch Latency Benchmark (100 lookups) ---")
latencies: list[float] = []
for i in range(100):
    user_id = f"u_{i:03d}"
    t0 = time.perf_counter()
    fs.get_online_features(
        features=REQUEST_FEATURES,
        entity_rows=[{"user_id": user_id}],
    ).to_dict()
    latencies.append((time.perf_counter() - t0) * 1000)

latencies.sort()
p50 = latencies[50]
p95 = latencies[95]
p99 = latencies[99]
print(f"Online lookup latency over 100 calls:")
print(f"  P50 = {p50:.2f}ms")
print(f"  P95 = {p95:.2f}ms")
print(f"  P99 = {p99:.2f}ms")

if p99 < 10:
    print(f"NB4 Part 3: PASS - P99 < 10ms ({p99:.2f}ms)")
else:
    print(f"NB4 Part 3: WARN - P99 = {p99:.2f}ms (threshold is < 10ms)")

# 6. PIT join
print("\n--- PIT Join (Historical Features) ---")
entity_df = pd.DataFrame({
    "user_id": ["u_001", "u_002", "u_003"],
    "event_timestamp": [NOW - timedelta(hours=2), NOW - timedelta(hours=1), NOW],
})

historical = fs.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_profile_features:reading_speed_wpm",
        "user_profile_features:topic_affinity",
    ],
).to_df()
print(historical)
print(f"\nNB4 Part 4: PIT join returned {len(historical)} rows × {len(historical.columns)} columns")
if len(historical) == 3:
    print("NB4 Part 4: PASS - PIT join correct")

print("\n=== NB4 COMPLETE ===")
