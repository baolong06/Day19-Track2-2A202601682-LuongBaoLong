# Test NB8 - Feature Engineering
import sys
import subprocess
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np

from app.features import (auc, frequency_encode, generate_events, latest_join,
                          leakage_experiment, leaked_row_fraction, pit_join,
                          window_aggregates)

print("--- Event Log ---")
events = generate_events(n_users=200, n_days=30, seed=42)
print(f"events   : {len(events)}")
print(f"users    : {events.user_id.nunique()}")
print(f"sessions : {events.session_id.nunique()}  (~{len(events)/events.session_id.nunique():.1f} event/session)")
print(f"click rate: {events.clicked.mean():.3f}")

print("\n--- Window Aggregates (Causal Features) ---")
feat = window_aggregates(events)
cols = ["searches_1h", "searches_24h", "searches_7d",
        "query_len_vs_user_avg", "query_len_delta", "seconds_since_last"]
print(feat[cols].describe().loc[["mean", "50%", "max"]].round(2).to_string())

print("\n--- AUC Train/Holdout (Honest Features) ---")
rng = np.random.default_rng(0)
mask = rng.random(len(feat)) < 0.7
print(f"{'feature':<26}{'train':>8}{'holdout':>9}{'gap':>8}")
for c in ["searches_24h", "searches_7d", "seconds_since_last"]:
    tr = auc(feat.loc[mask, c], feat.loc[mask, "clicked"])
    ho = auc(feat.loc[~mask, c], feat.loc[~mask, "clicked"])
    print(f"{c:<26}{tr:8.3f}{ho:9.3f}{tr-ho:+8.3f}")

print("\nNB8 Part 1: Honest features have small train/holdout gap")

print("\n--- Target Encoding Leakage (session_id vs user_id) ---")
print("-- key = session_id (high cardinality, ~1 event/group) --")
session_results = leakage_experiment(events, "session_id")
print(session_results.round(3).to_string(index=False))

print("\n-- key = user_id (lower cardinality, ~45 events/group) --")
user_results = leakage_experiment(events, "user_id")
print(user_results.round(3).to_string(index=False))

print("\nNB8 Part 2: Target leakage analysis")
print(session_results.to_string(index=False))
# Find the target-naive row
target_naive_row = session_results[session_results['encoding'] == 'target-naive'].iloc[0]
session_gap = target_naive_row['train_auc'] - target_naive_row['test_auc']
print(f"  session_id target-naive gap: {session_gap:.3f}")
if session_gap > 0.30:
    print("  PASS - Large leakage gap detected (>0.30)")
else:
    print(f"  WARN - Gap is {session_gap:.3f} (expected >0.30)")

print("\n--- PIT Join vs Latest Join ---")
fe = events[["user_id", "event_timestamp"]].copy().sort_values("event_timestamp")
fe["feature_value"] = fe.groupby("user_id").cumcount() + 1

rng = np.random.default_rng(1)
ent = (events.loc[rng.random(len(events)) < 0.4,
                  ["user_id", "event_timestamp", "clicked"]]
       .sort_values("event_timestamp").reset_index(drop=True))

lat, pit = latest_join(ent, fe), pit_join(ent, fe)
auc_lat = auc(lat["feature_value"], lat["clicked"])
auc_pit = auc(pit["feature_value"], pit["clicked"])

print(f"training rows                    : {len(ent)}")
print(f"dòng bị rò (giá trị ghi SAU nhãn): {leaked_row_fraction(ent, fe):.1%}")
print(f"\nAUC với latest-value join        : {auc_lat:.3f}")
print(f"AUC với point-in-time join       : {auc_pit:.3f}")
print(f"'lift ảo' sẽ mất khi lên production: {auc_lat - auc_pit:+.3f} AUC")

print("\nNB8 Part 3: PIT vs Latest Join")
print(f"  Leaked rows: {leaked_row_fraction(ent, fe):.1%}")
print(f"  AUC difference: {auc_lat - auc_pit:+.3f}")

print("\n--- On-Demand Feature View ---")
repo = ROOT / "app" / "feast_repo_ondemand"

# Run gen_spend.py
result = subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "gen_spend.py")],
                       capture_output=True, text=True)
if result.returncode == 0:
    print("gen_spend.py OK")
else:
    print(f"gen_spend.py failed: {result.stderr}")

# feast apply
result = subprocess.run([str(ROOT / ".venv" / "Scripts" / "feast"), "apply"],
                         cwd=str(repo), capture_output=True, text=True)
if result.returncode == 0:
    print("feast apply OK")
else:
    print(f"feast apply failed: {result.stderr}")

# materialize
result = subprocess.run([str(ROOT / ".venv" / "Scripts" / "feast"), "materialize-incremental", "2027-01-01T00:00:00"],
                         cwd=str(repo), capture_output=True, text=True)
if result.returncode == 0:
    print("feast materialize OK")
else:
    print(f"feast materialize failed: {result.stderr}")

from feast import FeatureStore

fs = FeatureStore(repo_path=str(repo))
out = fs.get_online_features(
    features=["user_spend_stats:avg_amount_7d",
              "amount_vs_avg:amount_vs_avg", "amount_vs_avg:is_spike"],
    entity_rows=[
        {"user_id": "u_000", "amount": 100_000.0},
        {"user_id": "u_000", "amount": 15_000_000.0},
        {"user_id": "u_001", "amount": 250_000.0},
    ],
).to_dict()

print("\nOn-demand feature results:")
for i in range(3):
    print(f"user={out['user_id'][i]}  avg7d={out['avg_amount_7d'][i]:>12,.0f}  "
          f"ratio={out['amount_vs_avg'][i]:6.2f}  spike={out['is_spike'][i]}")

print("\nNB8 Part 4: On-demand feature view")
if out['amount_vs_avg'][0] != out['amount_vs_avg'][1]:
    print("  PASS - Same user with different amounts gives different ratios")
else:
    print("  WARN - Expected different ratios for different amounts")

print("\n=== NB8 COMPLETE ===")
