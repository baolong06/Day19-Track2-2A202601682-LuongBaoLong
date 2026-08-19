# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB8 — Feature Engineering: 6 họ + 2 cách leak
#
# **Mục tiêu:** Demo 6 họ feature phổ biến, chứng minh 2 cách leak kinh
# điển (target encoding sai, latest join vs PIT), và on-demand feature view.
#
# **6 họ feature:**
# 1. Windowed aggregation (count/sum/avg over 1h/24h/7d)
# 2. Ratio / normalization (vs user's own baseline)
# 3. Lag & delta (previous value, change)
# 4. Recency (seconds since last event)
# 5. Categorical encoding (frequency + target)
# 6. Embedding as feature (vector as column)
#
# **2 cách leak:**
# - Target encoding trên toàn bộ data (bao gồm test) → train AUC cao, test AUC thấp
# - Latest join thay vì PIT join → feature value từ tương lai
#
# **Deck reference:** §7 "Feature Engineering: 6 Ho Feature Ban Se Viet Di Viet Lai"
#
# **Pass when:**
# - leak gap > 0.30 trên `session_id`
# - ODFV trả 2 giá trị khác nhau cho cùng user_cho cùng `event_timestamp`

# %% [markdown]
# ## 1. Setup — synthetic event log
#
# `app.features.generate_events()` produces a 9k-row search-event log:
# - 200 users × 30 days
# - `clicked` label depends on per-user engagement + topic affinity
# - made-up high-cardinality `session_id` (~2-3 events per session)
#   → ripe for target-encoding leakage

# %%
import _setup  # noqa: F401
import numpy as np
import pandas as pd
from pathlib import Path

from app.features import (
    TOPICS,
    auc,
    generate_events,
    leakage_experiment,
    latest_join,
    pit_join,
    leaked_row_fraction,
    target_encode_naive,
    target_encode_in_fold,
    window_aggregates,
    frequency_encode,
)

print("Generating 200-user × 30-day event log...")
events = generate_events(n_users=200, n_days=30, seed=42)
print(f"events: {len(events)} rows, {events['user_id'].nunique()} users")
print(f"columns: {list(events.columns)}")
print(f"clicked rate: {events['clicked'].mean():.2%}")
print(events.head(3))

# %% [markdown]
# ## 2. Families 1-4 — windowed aggregates, ratio, lag, recency
#
# **Windowed aggregation (Family 1):**
# - For each event, count **causal** events in the past 1h/24h/7d
# - Causal = only strictly previous events (not including current row)
#
# **Ratio (Family 2):**
# - query_len / user's own expanding mean (shifted)
# - Captures "this query is longer than usual for THIS user"
#
# **Lag & delta (Family 3):**
# - previous query_len, delta vs current
# - Captures velocity / burst behavior
#
# **Recency (Family 4):**
# - seconds since user's last event
# - Captures engagement intensity

# %%
feats = window_aggregates(events, windows=("1h", "24h", "7d"))
print(f"feature matrix: {feats.shape}")
print(f"columns: {list(feats.columns)}")
print(feats.head(3).to_string())

# Quick check: do these features predict label?
for col in ("searches_1h", "searches_24h", "searches_7d",
            "query_len_vs_user_avg", "query_len_delta", "seconds_since_last"):
    a = auc(feats[col].fillna(0).values, events["clicked"].values)
    print(f"  {col:30} AUC = {a:.3f}")

# %% [markdown]
# ## 3. Family 5 — categorical encoding (the leaked one)
#
# **Frequency encoding:** safe — depends only on feature distribution
# **Target encoding (naive):** LEAKY — each row's encoding includes its own label
# **Target encoding (in-fold):** correct — encoded using OTHER folds' labels
#
# **Demonstration:** split first, then encode. Compare train vs test AUC.
# - Naive: train AUC inflated, test AUC collapsed → big gap
# - In-fold: train AUC similar to test AUC → small gap

# %%
# Run leak experiment on session_id (high cardinality, leak-prone)
result = leakage_experiment(events, "session_id")
print("\n  Target encoding on session_id (high cardinality):")
print(result.to_string(index=False))

# %% [markdown]
# ## 4. Family 5 — categorical encoding (the one that doesn't leak)
#
# **Frequency encoding** depends only on counts, never on label.
# - Train AUC ≈ Test AUC → small gap
# - Safe for production

# %%
# Compare on topic (low cardinality, less leak-prone)
result_topic = leakage_experiment(events, "topic")
print("\n  Target encoding on topic (low cardinality):")
print(result_topic.to_string(index=False))

# %% [markdown]
# ## 5. PIT vs latest join — the time trap
#
# **PIT (point-in-time) join:** each row sees only features recorded AT OR BEFORE
# that row's timestamp. The only safe join for training.
#
# **Latest join:** `GROUP BY user_id` + tail(1) → silently pulls values recorded
# AFTER the label event. This is the most common data leakage in production.

# %%
# Build a "feature table" with a feature_value column
feature_table = events[["user_id", "event_timestamp"]].copy()
np.random.seed(0)
feature_table["feature_value"] = np.random.randn(len(events))

# PIT join: safe
pit = pit_join(events[["user_id", "event_timestamp", "clicked"]], feature_table)
print(f"PIT join: {pit.shape}, mean feature_value={pit['feature_value'].mean():.3f}")

# Latest join: leaky
latest = latest_join(events[["user_id", "event_timestamp", "clicked"]], feature_table)
print(f"Latest join: {latest.shape}, mean feature_value={latest['feature_value'].mean():.3f}")

# What fraction of training rows have a feature from the FUTURE?
leak_fraction = leaked_row_fraction(events, feature_table)
print(f"\nLeaked row fraction: {leak_fraction:.1%}")
print("  → A 'latest join' on this data would use post-event feature values")

# %% [markdown]
# ## 6. On-Demand Feature View — same user, different results
#
# **ODFV** computes feature values at request time from raw data, unlike
# pre-computed feature views that are materialized to a store.
#
# **Why ODFV matters:**
# - Real-time features (e.g., "events in last 5 minutes") can't be pre-computed
# - Bug: same user_id, same entity_row, but ODFV sees raw data evolving
#
# **Demo:** compute "user's 1h search count" via ODFV at two different times
# for the same user — different ctx means different counts.

# %%
# Demo ODFV: count searches in last 1h FROM A GIVEN TIMESTAMP
def odfv_searches_1h(user_id: str, ctx_timestamp: pd.Timestamp) -> int:
    user_events = events[events["user_id"] == user_id]
    cutoff = ctx_timestamp - pd.Timedelta(hours=1)
    return int(((user_events["event_timestamp"] >= cutoff) &
                (user_events["event_timestamp"] <= ctx_timestamp)).sum())

# Pick a user with multiple events CLOSE TOGETHER (within 1h)
active_user = events["user_id"].value_counts().index[0]
user_events = events[events["user_id"] == active_user].sort_values("event_timestamp").reset_index(drop=True)
# Find a window where 2 events are within 1h of each other
ts1 = user_events["event_timestamp"].iloc[5]
# ts2 = second event after ts1 (should be within 1h typically)
ts2 = user_events["event_timestamp"].iloc[6]

v1 = odfv_searches_1h(active_user, ts1)
v2 = odfv_searches_1h(active_user, ts2)
print(f"User: {active_user}")
print(f"  At {ts1}: 1h search count = {v1}")
print(f"  At {ts2}: 1h search count = {v2}")
print(f"  Different values: {v1 != v2}  (if True, ODFV is time-dependent as required)")
print(f"  Time delta: {(ts2 - ts1).total_seconds():.0f}s")

# %% [markdown]
# ## Diễn giải kết quả
#
# **Học từ output cell 2 (families 1-4):**
# - searches_24h/7d có AUC > 0.55 → genuine signal
# - searches_1h quá sparse → AUC thấp
# - query_len_vs_user_avg: ratio tốt hơn raw value (relative effect)
# - lag/recency: phụ thuộc burst pattern → AUC vừa
#
# **Học từ output cell 3 (target encoding leak):**
# - session_id (high-cardinality): target-naive gap lớn → LEAK
# - in-fold closes gap → CORRECT
# - frequency encoding: gap thấp vì không xài label
#
# **Học từ output cell 4 (topic encoding):**
# - topic (low-cardinality): gap nhỏ hơn vì per-class mean ổn định
# - High cardinality = leak nhiều hơn
#
# **Học từ output cell 5 (PIT vs latest):**
# - Leaked row fraction = X% — train rows đang dùng future feature
# - Cách fix: `merge_asof` với `direction="backward"` (PIT)
#
# **Học từ output cell 6 (ODFV):**
# - Same user, same ctx entity, different timestamps → different feature values
# - ODFV phù hợp real-time features (rolling counts, freshness)
# - Pre-computed feature views không thể cover real-time
#
# **Khi nào dùng họ nào?**
# - **1 (windowed):** activity intensity, fraud detection
# - **2 (ratio):** personalization, anomaly within user
# - **3 (lag/delta):** trend, velocity
# - **4 (recency):** engagement, churn prediction
# - **5 (frequency):** safe baseline; target encoding with care
# - **6 (embedding):** similarity, semantic features
# - **ODFV:** freshness < 1 minute, real-time
#
# **Think about:**
# - Tại sao high-cardinality target encoding leak nhiều?
#   → Per-category mean dominated by 1-2 rows (high variance)
#   → Look up the row's own label → perfect signal cho chính row đó
# - PIT vs ASOF join?
#   → Feast: `point_in_time_join` tự xử lý join_keys + timestamps
#   → Pandas: `merge_asof(direction="backward")` cho single-key PIT
# - ODFV cost?
#   → Compute at request time → thêm latency
#   → Materialize partial results + ODFV cho tail
#
# **Bài học:**
# 1. Leakage đến từ 2 nguồn: encoding (calculator) + join (time)
# 2. Always split before fit encoder; always use PIT for training
# 3. Pre-computed feature views cho batch, ODFV cho real-time
# 4. Audit AUC gap trước khi trust feature

# %% [markdown]
# ## Deliverable evidence
# 1. Output cell 2: 6 features + AUC cho mỗi cái
# 2. Output cell 3: leak gap > 0.30 cho session_id
# 3. Output cell 4: cùng metric cho topic (low cardinality)
# 4. Output cell 5: leaked_row_fraction + PIT vs latest
# 5. Output cell 6: ODFV trả 2 giá trị khác nhau
