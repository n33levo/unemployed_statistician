"""
Strategy 5: Blend TabPFN (multi-seed, T1+T2) with LR corrected.
  - Train LR with T1+T2 corrected labels (same correction as S3/S4)
  - Get LR OOF probs and test probs
  - Blend LR + TabPFN at various weights
  - Also try: LR with T1-only correction (what got 97.35% on Kaggle)
  - Find optimal blend weight + threshold
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from config import TRAIN_PATH, TEST_PATH, ID_COL, TARGET_COL, CAT_COLS, SEED

# ── Load data ──
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train["comorbidity"] = train["comorbidity"].fillna("None")
test["comorbidity"] = test["comorbidity"].fillna("None")

# ── Load tiers ──
with open(os.path.join(os.path.dirname(__file__), "..", "..", "reports", "tiered_noise_labels.json")) as f:
    tiers = json.load(f)

orig_labels = np.array(tiers["original_labels"])

# T1+T2 corrected labels
y_t1t2 = orig_labels.copy()
for idx in tiers["tier1_indices"]:
    y_t1t2[idx] = 1 - orig_labels[idx]
for idx in tiers["tier2_indices"]:
    y_t1t2[idx] = 1 - orig_labels[idx]

# T1-only corrected labels
y_t1 = orig_labels.copy()
for idx in tiers["tier1_indices"]:
    y_t1[idx] = 1 - orig_labels[idx]

X = train.drop(columns=[ID_COL, TARGET_COL])
X_test_raw = test.drop(columns=[ID_COL])

# ── Build preprocessor for LR ──
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if X[c].dtype == "object"]

def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ], sparse_threshold=0.0)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ── LR OOF (T1+T2) ──
print("=" * 60)
print("LR with T1+T2 corrected labels")
print("=" * 60)

lr_oof_t1t2 = np.zeros(len(y_t1t2), dtype=float)
for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y_t1t2), 1):
    pre = make_pre()
    X_tr = pre.fit_transform(X.iloc[tr_idx])
    X_va = pre.transform(X.iloc[va_idx])
    lr = LogisticRegression(max_iter=2500, C=2.0, random_state=SEED)
    lr.fit(X_tr, y_t1t2[tr_idx])
    lr_oof_t1t2[va_idx] = lr.predict_proba(X_va)[:, 1]
    fold_acc = accuracy_score(y_t1t2[va_idx], (lr_oof_t1t2[va_idx] >= 0.5).astype(int))
    print(f"  Fold {fold}: accuracy = {fold_acc:.4f}")

lr_acc_t1t2 = accuracy_score(y_t1t2, (lr_oof_t1t2 >= 0.5).astype(int))
print(f"LR T1+T2 OOF accuracy: {lr_acc_t1t2:.4f}")

# LR full train → test (T1+T2)
pre_full = make_pre()
X_train_full = pre_full.fit_transform(X)
X_test_full = pre_full.transform(X_test_raw)
lr_full = LogisticRegression(max_iter=2500, C=2.0, random_state=SEED)
lr_full.fit(X_train_full, y_t1t2)
lr_test_t1t2 = lr_full.predict_proba(X_test_full)[:, 1]

# ── LR OOF (T1 only) ──
print(f"\n{'=' * 60}")
print("LR with T1-only corrected labels")
print("=" * 60)

lr_oof_t1 = np.zeros(len(y_t1), dtype=float)
for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y_t1), 1):
    pre = make_pre()
    X_tr = pre.fit_transform(X.iloc[tr_idx])
    X_va = pre.transform(X.iloc[va_idx])
    lr = LogisticRegression(max_iter=2500, C=2.0, random_state=SEED)
    lr.fit(X_tr, y_t1[tr_idx])
    lr_oof_t1[va_idx] = lr.predict_proba(X_va)[:, 1]
    fold_acc = accuracy_score(y_t1[va_idx], (lr_oof_t1[va_idx] >= 0.5).astype(int))
    print(f"  Fold {fold}: accuracy = {fold_acc:.4f}")

lr_acc_t1 = accuracy_score(y_t1, (lr_oof_t1 >= 0.5).astype(int))
print(f"LR T1-only OOF accuracy: {lr_acc_t1:.4f}")

# LR full train → test (T1)
lr_full_t1 = LogisticRegression(max_iter=2500, C=2.0, random_state=SEED)
lr_full_t1.fit(X_train_full, y_t1)
lr_test_t1 = lr_full_t1.predict_proba(X_test_full)[:, 1]

# ── Load TabPFN probs from S4 ──
tabpfn_oof = np.load(os.path.join(os.path.dirname(__file__), "..", "reports", "s4_avg_oof_probs.npy"))
tabpfn_test = np.load(os.path.join(os.path.dirname(__file__), "..", "reports", "s4_avg_test_probs.npy"))

print(f"\nTabPFN multi-seed OOF accuracy: {accuracy_score(y_t1t2, (tabpfn_oof >= 0.5).astype(int)):.4f}")

# ── Blend search ──
print(f"\n{'=' * 60}")
print("BLEND SEARCH: w * TabPFN + (1-w) * LR")
print("=" * 60)

best_overall = (0, 0, 0.5, 0)  # (blend_name, weight, threshold, accuracy)

for lr_name, lr_oof, lr_test, y_ref in [
    ("LR_T1T2", lr_oof_t1t2, lr_test_t1t2, y_t1t2),
    ("LR_T1", lr_oof_t1, lr_test_t1, y_t1t2),  # still evaluate vs T1+T2 corrected
]:
    print(f"\n  --- TabPFN + {lr_name} ---")
    for w in np.arange(0.0, 1.05, 0.05):
        blend_oof = w * tabpfn_oof + (1 - w) * lr_oof
        # Sweep thresholds
        best_t, best_a = 0.5, accuracy_score(y_ref, (blend_oof >= 0.5).astype(int))
        for t in np.arange(0.25, 0.75, 0.01):
            a = accuracy_score(y_ref, (blend_oof >= t).astype(int))
            if a > best_a:
                best_a = a
                best_t = t
        
        acc_50 = accuracy_score(y_ref, (blend_oof >= 0.5).astype(int))
        if w in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
            print(f"    w={w:.2f}: acc@0.50={acc_50:.4f}, best={best_a:.4f}@{best_t:.2f}")
        
        if best_a > best_overall[3]:
            best_overall = (f"TabPFN+{lr_name}", w, best_t, best_a)

print(f"\n{'=' * 60}")
print(f"BEST BLEND: {best_overall[0]}")
print(f"  Weight (TabPFN): {best_overall[1]:.2f}")
print(f"  Threshold: {best_overall[2]:.2f}")
print(f"  OOF accuracy: {best_overall[3]:.4f}")
print(f"{'=' * 60}")

# ── Generate submissions for top blends ──
bw = best_overall[1]
bt = best_overall[2]

# Best blend
if "T1T2" in best_overall[0]:
    blend_test = bw * tabpfn_test + (1 - bw) * lr_test_t1t2
else:
    blend_test = bw * tabpfn_test + (1 - bw) * lr_test_t1

preds = (blend_test >= bt).astype(int)
n0, n1 = (preds == 0).sum(), (preds == 1).sum()
print(f"\nBest blend predictions: {n0} zeros, {n1} ones")

sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: preds})
out = os.path.join(os.path.dirname(__file__), "..", "submissions", "s5_best_blend.csv")
sub.to_csv(out, index=False)
print(f"Saved → {out}")

# compare with s1
s1_probs = np.load(os.path.join(os.path.dirname(__file__), "..", "reports", "s1_test_probs.npy"))
s1_preds = (s1_probs >= 0.5).astype(int)
print(f"\nDiffs vs S1 baseline: {(s1_preds != preds).sum()}")

print("\ndone")
