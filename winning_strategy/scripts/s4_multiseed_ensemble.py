"""
Strategy 4: Multi-seed TabPFN ensemble + threshold tuning.
  - Use T1+T2 corrected labels (best from S3)
  - Train TabPFN with 5 different random seeds
  - Average their probabilities for more stable predictions
  - Sweep thresholds on the averaged OOF probabilities
  - Also generate individual seed submissions for diversity
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier
from config import TRAIN_PATH, TEST_PATH, ID_COL, TARGET_COL, CAT_COLS, SEED

# ── Load data ──
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train["comorbidity"] = train["comorbidity"].fillna("None")
test["comorbidity"] = test["comorbidity"].fillna("None")

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit(train[CAT_COLS])

def encode(df):
    oh = pd.DataFrame(
        encoder.transform(df[CAT_COLS]),
        columns=encoder.get_feature_names_out(CAT_COLS),
        index=df.index,
    )
    return pd.concat([df.drop(columns=CAT_COLS), oh], axis=1)

train_enc = encode(train)
test_enc = encode(test)

# ── T1+T2 corrected labels ──
with open(os.path.join(os.path.dirname(__file__), "..", "..", "reports", "tiered_noise_labels.json")) as f:
    tiers = json.load(f)

orig_labels = np.array(tiers["original_labels"])
y = orig_labels.copy()
for idx in tiers["tier1_indices"]:
    y[idx] = 1 - orig_labels[idx]
for idx in tiers["tier2_indices"]:
    y[idx] = 1 - orig_labels[idx]

X = train_enc.drop(columns=[ID_COL, TARGET_COL])
X_test = test_enc.drop(columns=[ID_COL])

print(f"Using T1+T2 corrected labels ({len(tiers['tier1_indices'])} + {len(tiers['tier2_indices'])} = {len(tiers['tier1_indices'])+len(tiers['tier2_indices'])} corrections)")

# ── Multi-seed 5-fold CV ──
SEEDS = [42, 123, 314, 777, 2026]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

all_oof_probs = []
all_test_probs = []

for seed in SEEDS:
    print(f"\n{'=' * 60}")
    print(f"SEED = {seed}")
    print(f"{'=' * 60}")
    
    oof_probs = np.zeros(len(y), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        clf = TabPFNClassifier(random_state=seed, device="cpu", ignore_pretraining_limits=True)
        clf.fit(X.iloc[tr_idx], y[tr_idx])
        oof_probs[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]
        fold_acc = accuracy_score(y[va_idx], (oof_probs[va_idx] >= 0.5).astype(int))
        print(f"  Fold {fold}: accuracy = {fold_acc:.4f}")
    
    oof_acc = accuracy_score(y, (oof_probs >= 0.5).astype(int))
    print(f"  OOF accuracy: {oof_acc:.4f}")
    all_oof_probs.append(oof_probs)
    
    # Full train → test
    clf_full = TabPFNClassifier(random_state=seed, device="cpu", ignore_pretraining_limits=True)
    clf_full.fit(X, y)
    test_probs = []
    for i in range(0, len(X_test), 1000):
        test_probs.append(clf_full.predict_proba(X_test.iloc[i:i+1000])[:, 1])
    test_probs = np.concatenate(test_probs)
    all_test_probs.append(test_probs)

# ── Average across seeds ──
avg_oof = np.mean(all_oof_probs, axis=0)
avg_test = np.mean(all_test_probs, axis=0)

print(f"\n{'=' * 60}")
print("MULTI-SEED AVERAGE RESULTS")
print(f"{'=' * 60}")

avg_oof_acc = accuracy_score(y, (avg_oof >= 0.5).astype(int))
avg_oof_noisy = accuracy_score(orig_labels, (avg_oof >= 0.5).astype(int))
print(f"Average OOF accuracy (vs corrected labels): {avg_oof_acc:.4f}")
print(f"Average OOF accuracy (vs original noisy): {avg_oof_noisy:.4f}")

# Threshold sweep
print("\nThreshold sweep on averaged OOF probs:")
best_t, best_a = 0.5, avg_oof_acc
results = []
for t in np.arange(0.20, 0.80, 0.01):
    a = accuracy_score(y, (avg_oof >= t).astype(int))
    results.append((t, a))
    if a > best_a:
        best_a = a
        best_t = t

# Print top 5 thresholds
results.sort(key=lambda x: -x[1])
for t, a in results[:5]:
    print(f"  threshold={t:.2f}, accuracy={a:.4f}")
print(f"\nBest threshold: {best_t:.2f}, accuracy: {best_a:.4f}")

# ── Save ensemble submission ──
preds_avg = (avg_test >= best_t).astype(int)
n0, n1 = (preds_avg == 0).sum(), (preds_avg == 1).sum()
print(f"\nAverage ensemble predictions: {n0} zeros, {n1} ones ({n1/len(preds_avg)*100:.1f}% positive)")

sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: preds_avg})
out_path = os.path.join(os.path.dirname(__file__), "..", "submissions", "s4_multiseed_t1t2.csv")
sub.to_csv(out_path, index=False)
print(f"Saved → {out_path}")

# Also save at 0.50 threshold for comparison
preds_50 = (avg_test >= 0.5).astype(int)
sub_50 = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: preds_50})
out_50 = os.path.join(os.path.dirname(__file__), "..", "submissions", "s4_multiseed_t1t2_t50.csv")
sub_50.to_csv(out_50, index=False)

np.save(os.path.join(os.path.dirname(__file__), "..", "reports", "s4_avg_test_probs.npy"), avg_test)
np.save(os.path.join(os.path.dirname(__file__), "..", "reports", "s4_avg_oof_probs.npy"), avg_oof)

# Compare with S1 and S3
s1_probs = np.load(os.path.join(os.path.dirname(__file__), "..", "reports", "s1_test_probs.npy"))
s3_probs = np.load(os.path.join(os.path.dirname(__file__), "..", "reports", "s3_test_probs.npy"))

s1_preds = (s1_probs >= 0.5).astype(int)
s3_preds = (s3_probs >= 0.5).astype(int)
print(f"\nDiffs vs S1 (T1 baseline): {(s1_preds != preds_avg).sum()}")
print(f"Diffs vs S3 (T1+T2 single seed): {(s3_preds != preds_avg).sum()}")
print(f"Diffs S1 vs S3: {(s1_preds != s3_preds).sum()}")

print(f"\n{'=' * 60}")
print("S4 DONE")
print(f"{'=' * 60}")
