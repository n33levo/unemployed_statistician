"""
Strategy 1: Reproduce Hussain's exact 98.42% pipeline locally.
  - One-hot encode gender + comorbidity
  - Correct Tier 1 labels (215 flips)
  - Train TabPFNClassifier on full corrected training set
  - Predict on test with hard predict() AND predict_proba()
  - Also run 5-fold CV to measure OOF accuracy (against corrected labels)
  - Save submission CSV
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from config import TRAIN_PATH, TEST_PATH, ID_COL, TARGET_COL, CAT_COLS, SEED

# ── Load data ──
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train["comorbidity"] = train["comorbidity"].fillna("None")
test["comorbidity"] = test["comorbidity"].fillna("None")

# ── One-hot encode (fit on train, transform both) ──
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

# ── Correct Tier 1 labels ──
with open(os.path.join(os.path.dirname(__file__), "..", "..", "reports", "tiered_noise_labels.json")) as f:
    tiers = json.load(f)

t1_indices = tiers["tier1_indices"]
corrected_labels = tiers["corrected_labels"]

for idx in t1_indices:
    train_enc.at[idx, TARGET_COL] = corrected_labels[idx]

print(f"Corrected {len(t1_indices)} Tier 1 labels")

# ── Prepare X, y ──
y = train_enc[TARGET_COL].astype(int).values
X = train_enc.drop(columns=[ID_COL, TARGET_COL])
X_test = test_enc.drop(columns=[ID_COL])

print(f"Train shape: {X.shape}, Test shape: {X_test.shape}")
print(f"Features: {X.columns.tolist()}")

# ── 5-fold CV (against corrected labels) ──
from tabpfn import TabPFNClassifier

print("\n" + "=" * 60)
print("5-FOLD CV (TabPFN on corrected labels)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(y), dtype=int)
oof_probs = np.zeros(len(y), dtype=float)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
    clf = TabPFNClassifier(random_state=42, device="cpu", ignore_pretraining_limits=True)
    clf.fit(X.iloc[tr_idx], y[tr_idx])
    oof_preds[va_idx] = clf.predict(X.iloc[va_idx])
    oof_probs[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]
    fold_acc = accuracy_score(y[va_idx], oof_preds[va_idx])
    print(f"  Fold {fold}: accuracy = {fold_acc:.4f}")

oof_acc = accuracy_score(y, oof_preds)
print(f"\nOOF accuracy (vs corrected labels): {oof_acc:.4f}")

# Also check OOF accuracy vs original noisy labels
orig_labels = np.array(tiers["original_labels"])
oof_acc_noisy = accuracy_score(orig_labels, oof_preds)
print(f"OOF accuracy (vs original noisy labels): {oof_acc_noisy:.4f}")

# Threshold sweep on OOF probs
print("\nThreshold sweep on OOF probs (vs corrected labels):")
best_t, best_a = 0.5, oof_acc
for t in np.arange(0.30, 0.71, 0.01):
    a = accuracy_score(y, (oof_probs >= t).astype(int))
    if a > best_a:
        best_a = a
        best_t = t
print(f"  Best threshold: {best_t:.2f}, accuracy: {best_a:.4f}")

# ── Train on full data and predict test ──
print("\n" + "=" * 60)
print("FULL TRAIN → TEST PREDICTION")
print("=" * 60)

clf_full = TabPFNClassifier(random_state=42, device="cpu", ignore_pretraining_limits=True)
clf_full.fit(X, y)

# Hard predictions
batch_size = 1000
predictions = []
probabilities = []
for i in range(0, len(X_test), batch_size):
    batch_pred = clf_full.predict(X_test.iloc[i:i + batch_size])
    batch_prob = clf_full.predict_proba(X_test.iloc[i:i + batch_size])[:, 1]
    predictions.append(batch_pred)
    probabilities.append(batch_prob)

predictions = np.concatenate(predictions)
probabilities = np.concatenate(probabilities)

n0 = (predictions == 0).sum()
n1 = (predictions == 1).sum()
print(f"Predictions: {n0} zeros, {n1} ones ({n1/len(predictions)*100:.1f}% positive)")

# Save submission
sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: predictions.astype(int)})
out_path = os.path.join(os.path.dirname(__file__), "..", "submissions", "s1_baseline_tabpfn.csv")
sub.to_csv(out_path, index=False)
print(f"Saved → {out_path}")

# Also save probabilities for later blending
np.save(os.path.join(os.path.dirname(__file__), "..", "reports", "s1_test_probs.npy"), probabilities)
np.save(os.path.join(os.path.dirname(__file__), "..", "reports", "s1_oof_probs.npy"), oof_probs)
print("Saved test probabilities and OOF probabilities for blending")

print("\n" + "=" * 60)
print("S1 BASELINE DONE")
print("=" * 60)
