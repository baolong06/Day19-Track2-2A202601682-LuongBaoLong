# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB4 — Feast Feature Store: 3 Feature Views
#
# **Mục tiêu:** Định nghĩa 3 feature views, materialize sang SQLite online store,
# đo lookup latency P99 < 10ms.
#
# **WHY Feast?**
# - Đảm bảo **consistency** giữa training (offline) và serving (online)
# - **Point-in-Time join** tự động: tránh data leakage trong training
# - **Materialize** có lịch: giữ online store fresh với offline store

# %% [markdown]
# ## 1. Sinh dữ liệu offline (Parquet) cho 3 feature views
#
# **WHY 3 feature views?**
# - `user_profile` (TTL dài): thông tin ổn định về user
# - `item_popularity` (theo item): tín hiệu về doc/article
# - `query_velocity` (TTL ngắn): hành vi real-time của user
# → Mỗi view phục vụ một mục đích khác nhau, **không trộn lẫn**

# %%
import _setup  # noqa: F401
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(_setup.__file__).resolve().parent.parent
FEAST_DIR = REPO_ROOT / "app" / "feast_repo"
FEAST_DATA = FEAST_DIR / "data"
FEAST_DATA.mkdir(exist_ok=True)

# %%
NOW = datetime.now(timezone.utc).replace(microsecond=0)


def make_user_profile(n_users: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [f"u_{i:03d}" for i in range(n_users)],
        "reading_speed_wpm": [180 + (i * 7) % 200 for i in range(n_users)],
        "preferred_language": ["vi" if i % 3 != 0 else "en" for i in range(n_users)],
        "topic_affinity": [
            ["ai_ml", "cloud", "security", "database", "devops"][i % 5]
            for i in range(n_users)
        ],
        "event_timestamp": [NOW - timedelta(hours=i % 48) for i in range(n_users)],
    })


def make_item_popularity(n_items: int = 1000) -> pd.DataFrame:
    return pd.DataFrame({
        "doc_id": [f"item_{i:04d}" for i in range(n_items)],
        "click_count_24h": [(i * 13) % 500 for i in range(n_items)],
        "ctr_7d": [round(((i * 7) % 100) / 100.0, 3) for i in range(n_items)],
        "avg_dwell_seconds": [10.0 + (i * 0.7) % 90 for i in range(n_items)],
        "event_timestamp": [NOW - timedelta(minutes=i % 720) for i in range(n_items)],
    })


def make_query_velocity(n_users: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [f"u_{i:03d}" for i in range(n_users)],
        "queries_last_hour": [(i * 11) % 50 for i in range(n_users)],
        "distinct_topics_24h": [1 + (i * 3) % 10 for i in range(n_users)],
        "event_timestamp": [NOW - timedelta(minutes=i % 30) for i in range(n_users)],
    })


# **WHY Parquet chứ không CSV?**
# - Parquet: columnar, nén tốt, Feast đọc trực tiếp
# - CSV: dễ đọc, nhưng phải parse → chậm hơn
# - Trong production: Parquet trên S3/GCS rất phổ biến
make_user_profile().to_parquet(FEAST_DATA / "user_profile.parquet", index=False)
make_item_popularity().to_parquet(FEAST_DATA / "item_popularity.parquet", index=False)
make_query_velocity().to_parquet(FEAST_DATA / "query_velocity.parquet", index=False)
print(f"Wrote 3 Parquet sources to {FEAST_DATA}")
for p in sorted(FEAST_DATA.glob("*.parquet")):
    print(f"  {p.name}  {p.stat().st_size / 1024:.1f} KB")

# %% [markdown]
# ## 2. `feast apply` — register 3 feature views
#
# **HOW Feast registries?**
# - `feature_views.py` (Python) định nghĩa schema + entity + source
# - `feast apply` đọc file và ghi metadata vào `registry.db` (SQLite)
# - Registry là **source of truth** cho cả team
# - Nếu đổi definition → chạy lại `feast apply`

# %%
res = subprocess.run(
    ["feast", "apply"],
    cwd=str(FEAST_DIR),
    capture_output=True, text=True, check=False,
)
print("STDOUT:")
print(res.stdout)
print(f"Return code: {res.returncode}")
if res.stderr:
    print("STDERR:")
    print(res.stderr)
if res.returncode != 0:
    raise RuntimeError(f"feast apply failed: {res.stderr}")

# %% [markdown]
# ## 3. `feast materialize` — load offline → online
#
# **WHY materialize tách riêng với apply?**
# - `apply` chỉ register metadata
# - `materialize` chuyển **giá trị mới nhất** từ offline store vào online store
# - Trong production: scheduled job chạy `materialize` mỗi 5-15 phút
#
# **WHY `materialize START END` (full) thay vì `materialize-incremental`?**
# - Incremental cần state `last_materialized_at` trong registry
# - Lần đầu chạy / data corruption → incremental skip hết (0 rows)
# - Full reload chỉ chậm vài giây với lab scale, dễ debug

# %%
# **WHY `feast materialize` (full) thay vì incremental?**
# - `materialize-incremental` cần `last_materialized_at` trong registry,
#   nếu chưa có → skip hết
# - `materialize START END` luôn load toàn bộ range, dễ hiểu, dễ debug
# - Lab này: dữ liệu 100 rows, full reload < 1s nên không tốn
# - Production: chạy incremental từ cron job scheduler
start_dt = (NOW - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
end_dt = (NOW + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
res = subprocess.run(
    ["feast", "materialize", start_dt, end_dt],
    cwd=str(FEAST_DIR),
    capture_output=True, text=True, check=False,
)
print("STDOUT:")
print(res.stdout)
print(f"Return code: {res.returncode}")
if res.stderr:
    print("STDERR:")
    print(res.stderr)
if res.returncode != 0:
    raise RuntimeError(f"materialize failed: {res.stderr}")

# %% [markdown]
# ## 4. Online lookup — đo latency
#
# **WHY đo latency ở lookup?**
# - Online feature lookup phải nhanh: < 10ms là bắt buộc cho mọi request có
#   personalized ranking
# - Nếu chậm → throttle cho toàn bộ inference pipeline
#
# **HOW Feast lookup work?**
# 1. Nhận entity_rows (list of dict, ví dụ `[{"user_id": "u_001"}]`)
# 2. Query online store (SQLite ở lab, Redis ở prod) cho mỗi entity
# 3. Trả về dict {feature_name: value}

# %%
import time

from feast import FeatureStore

fs = FeatureStore(repo_path=str(FEAST_DIR))

REQUEST_FEATURES = [
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "user_profile_features:topic_affinity",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]

t0 = time.perf_counter()
features = fs.get_online_features(
    features=REQUEST_FEATURES,
    entity_rows=[{"user_id": "u_001"}],
).to_dict()
single_latency_ms = (time.perf_counter() - t0) * 1000
print(f"Single lookup: {single_latency_ms:.2f}ms")
print({k: v[0] for k, v in features.items()})

# %% [markdown]
# ## 5. Batch latency benchmark (100 lookups)
#
# **WHY percentile thay vì mean?**
# - Mean bị kéo lên bởi outliers (cold page cache, GC)
# - P99 = 99% requests nhanh hơn con số này → SLA thật
# - 100 lookups đủ để P99 ổn định
#
# **WHY cả online lookup latency?**
# - SQLite local: ~1-5ms (nhanh vì data trong RAM cache)
# - Redis remote: ~2-8ms (thêm network)
# - DynamoDB: ~5-15ms (network xa hơn)

# %%
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
    print(f"PASS — online lookup P99 < 10ms ({p99:.2f}ms)")
else:
    print(f"WARN — P99 = {p99:.2f}ms (SQLite local OK; nếu vẫn > 10ms kiểm tra SQLite locks)")

# %% [markdown]
# ## 6. PIT join (offline) — no data leakage
#
# **WHY PIT join quan trọng?**
# - Training cần feature tại **thời điểm** của sự kiện, không phải feature hiện tại
# - Ví dụ: user click vào 9h, dùng feature từ 8h → đúng
# - Nếu dùng feature 12h (sau khi click) → **data leakage** → training accuracy
#   cao nhưng production tệ hại
#
# **HOW Feast PIT?**
# - `get_historical_features(entity_df, features)`
# - Feast scan offline store, lấy giá trị **gần nhất trước** mỗi `event_timestamp`
# - Internal nó dùng LEFT JOIN với `ASOF` semantics

# %%
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

# %% [markdown]
# ## Diễn giải kết quả
#
# **Think about:**
# - Tại sao `user_profile` TTL dài hơn `query_velocity`?
#   → Profile ổn định; query_velocity đếm-theo-giờ, cần refresh thường xuyên
# - Khi nào cần Redis thay SQLite?
#   → Khi cần lookup > 10K QPS, hoặc multi-region
# - Tại sao dùng PIT join cho training?
#   → Tránh leakage; matching với timestamp của label
#
# **Bài học:**
# 1. Feature store tách **definition** (apply) khỏi **data** (materialize)
# 2. Online lookup latency là SLO quan trọng — benchmark thường xuyên
# 3. PIT join là defensive guard chống training-serving skew
# 4. TTL phải match với business semantics, không phải arbitrary

# %% [markdown]
# ## Deliverable evidence
# 1. Output cell 2: 3 Parquet files generated
# 2. Output cell 3: `feast apply` STDOUT showing "Created feature view × 3"
# 3. Output cell 4: `materialize` log rows materialized to online store
# 4. Output cell 5: 1 online lookup result + latency
# 5. Output cell 6: 100-lookup P50/P95/P99 + PASS line
# 6. Output cell 7: PIT join DataFrame (3 rows × features)