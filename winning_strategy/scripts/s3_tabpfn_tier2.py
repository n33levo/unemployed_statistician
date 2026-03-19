"""
Strategy 3: TabPFN + corrected Tier 1 AND Tier 2 labels.
  - Tier 1 (215 samples): all 3 methods agree → flip labels
  - Tier 2 (81 samples): 2 methods agree → also flip labels (aggressive)
  - Also try: Tier 2 with model-predicted labels (use OOF probs from S1)
  - Compare vs S1 baseline (Tier 1 only = 0.9053)
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

# ── Load tiers ──
with open(os.path.join(os.path.dirname(__file__), "..", "..", "reports", "tiered_noise_labels.json")) as f:
    tiers = json.load(f)

t1 = tiers["tier1_indices"]
t2 = tiers["tier2_indices"]
orig_labels = np.array(tiers["original_labels"])
corrected_labels = np.array(tiers["corrected_labels"])

# ── Strategy A: Flip T1 + T2 (aggressive) ──
labels_a = orig_labels.copy()
for idx in t1:
    labels_a[idx] = 1 - orig_labels[idx]
for idx in t2:
    labels_a[idx] = 1 - orig_labels[idx]

# ── Strategy B: Flip T1 only (baseline, for comparison) ──
labels_b = orig_labels.copy()
for idx in t1:
    labels_b[idx] = 1 - orig_labels[idx]

X = train_enc.drop(columns=[ID_COL, TARGET_COL])
X_test = test_enc.drop(columns=[ID_COL])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def run_cv(name, y_labels):
    print(f"\n{'=' * 60}")
    print(f"5-FOLD CV: {name}")
    print(f"{'=' * 60}")
    oof_preds = np.zeros(len(y_labels), dtype=int)
    oof_probs = np.zeros(len(y_labels), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y_labels), 1):
        clf = TabPFNClassifier(random_state=42, device="cpu", ignore_pretraining_limits=True)
        clf.fit(X.iloc[tr_idx], y_labels[tr_idx])
        oof_preds[va_idx] = clf.predict(X.iloc[va_idx])
        oof_probs[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]
        fold_acc = accuracy_score(y_labels[va_idx], oof_preds[va_idx])
        print(f"  Fold {fold}: accuracy = {fold_acc:.4f}")
    
    oof_acc = accuracy_score(y_labels, oof_preds)
    oof_acc_noisy = accuracy_score(orig_labels, oof_preds)
    print(f"\n  OOF accuracy (vs its own labels): {oof_acc:.4f}")
    print(f"  OOF accuracy (vs original noisy): {oof_acc_noisy:.4f}")
    
    best_t, best_a = 0.5, oof_acc
    for t in np.arange(0.25, 0.75, 0.01):
        a = accuracy_score(y_labels, (oof_probs >= t).astype(int))
        if a > best_a:
            best_a = a
            best_t = t
    print(f"  Best threshold: {best_t:.2f}, accuracy: {best_a:.4f}")
    
    return oof_probs, oof_preds

print(f"Tier 1: {len(t1)} flips, Tier 2: {len(t2)} flips")
print(f"Strategy A (T1+T2): {len(t1) + len(t2)} total corrections")
print(f"Strategy B (T1 only): {len(t1)} corrections [baseline]")

# Run both
probs_a, preds_a = run_cv("T1 + T2 flipped (aggressive)", labels_a)
probs_b, preds_b = run_cv("T1 only (baseline reference)", labels_b)

# ── Full train → test for Strategy A ──
print(f"\n{'=' * 60}")
print("FULL TRAIN → TEST (T1 + T2 corrected)")
print(f"{'=' * 60}")

clf_full = TabPFNClassifier(random_state=42, device="cpu", ignore_pretraining_limits=True)
clf_full.fit(X, labels_a)

predictions = []
probabilities = []
for i in range(0, len(X_test), 1000):
    predictions.append(clf_full.predict(X_test.iloc[i:i+1000]))
    probabilities.append(clf_full.predict_proba(X_test.iloc[i:i+1000])[:, 1])

predictions = np.concatenate(predictions)
probabilities = np.concatenate(probabilities)

n0, n1 = (predictions == 0).sum(), (predictions == 1).sum()
print(f"Predictions: {n0} zeros, {n1} ones ({n1/len(predictions)*100:.1f}% positive)")

sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: predictions.astype(int)})
out_path = os.path.join(os.path.dirname(__file__), "..", "submissions", "s3_tabpfn_t1t2.csv")
sub.to_csv(out_path, index=False)
print(f"Saved → {out_path}")

np.save(os.path.join(os.path.dirname(__file__), "..", "reports", "s3_test_probs.npy"), probabilities)
np.save(os.path.join(os.path.dirname(__file__), "..", "reports", "s3_oof_probs.npy"), probs_a)

# Compare with S1
s1_probs = np.load(os.path.join(os.path.dirname(__file__), "..", "reports", "s1_test_probs.npy"))
s1_preds = (s1_probs >= 0.5).astype(int)
s3_preds = predictions.astype(int)
diffs = (s1_preds != s3_preds).sum()
print(f"\nDifferences vs S1 baseline: {diffs} / {len(s3_preds)} test predictions differ")

print(f"\n{'=' * 60}")
print("S3 DONE")
print(f"{'=' * 60}")
