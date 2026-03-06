import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from config import TRAIN_PATH, TEST_PATH, ID_COL, TARGET_COL, NUM_COLS, SYMPTOM_COLS, CAT_COLS

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

train[TARGET_COL] = train[TARGET_COL].astype(int)

feat_cols = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ---- shapes ----
print(f"\ntrain shape: {train.shape}")
print(f"test shape:  {test.shape}")

# ---- class balance ----
pos = train[TARGET_COL].sum()
neg = len(train) - pos
print(f"\nclass balance: positive={pos} ({pos/len(train):.2%}), negative={neg} ({neg/len(train):.2%})")

# ---- missing values ----
print("\n--- missing values (train) ---")
miss_train = train[feat_cols].isnull().sum()
miss_train = miss_train[miss_train > 0]
if len(miss_train):
    for col, cnt in miss_train.items():
        print(f"  {col}: {cnt} ({cnt/len(train):.1%})")
else:
    print("  none")

print("\n--- missing values (test) ---")
miss_test = test.drop(columns=[ID_COL]).isnull().sum()
miss_test = miss_test[miss_test > 0]
if len(miss_test):
    for col, cnt in miss_test.items():
        print(f"  {col}: {cnt} ({cnt/len(test):.1%})")
else:
    print("  none")

# ---- numeric stats by class ----
print("\n--- numeric features by class ---")
for col in NUM_COLS:
    m0 = train.loc[train[TARGET_COL] == 0, col].mean()
    m1 = train.loc[train[TARGET_COL] == 1, col].mean()
    s = train[col].std()
    eff = (m1 - m0) / s if s > 0 else 0
    print(f"  {col}: mean(neg)={m0:.2f}, mean(pos)={m1:.2f}, cohen_d={eff:.3f}")

# ---- train vs test numeric shift ----
print("\n--- train vs test numeric shift ---")
for col in NUM_COLS:
    tr_mean = train[col].mean()
    te_mean = test[col].mean()
    tr_std = train[col].std()
    te_std = test[col].std()
    print(f"  {col}: train_mean={tr_mean:.2f} (std={tr_std:.2f}), test_mean={te_mean:.2f} (std={te_std:.2f}), delta={te_mean - tr_mean:.2f}")

# note: train has high-precision floats, test has rounded values
print("\n  NOTE: train oxygen_level/body_temperature have many decimal places")
print(f"  e.g. train oxygen_level[0] = {train['oxygen_level'].iloc[0]}")
print(f"       test  oxygen_level[0] = {test['oxygen_level'].iloc[0]}")

# ---- categorical distributions ----
print("\n--- categorical distributions ---")
for col in CAT_COLS:
    print(f"\n  {col}:")
    tr_counts = train[col].fillna("None").value_counts(normalize=True).sort_index()
    te_counts = test[col].fillna("None").value_counts(normalize=True).sort_index()
    all_vals = sorted(set(tr_counts.index) | set(te_counts.index))
    for v in all_vals:
        tp = tr_counts.get(v, 0)
        ep = te_counts.get(v, 0)
        print(f"    {v:20s}  train={tp:.3f}  test={ep:.3f}  diff={ep-tp:+.3f}")

# ---- symptom rates by class ----
print("\n--- symptom prevalence by class ---")
for col in SYMPTOM_COLS:
    r0 = train.loc[train[TARGET_COL] == 0, col].mean()
    r1 = train.loc[train[TARGET_COL] == 1, col].mean()
    print(f"  {col:25s}  neg={r0:.3f}  pos={r1:.3f}  diff={r1-r0:+.3f}")

# ---- duplicate rows ----
print("\n--- duplicate feature rows ---")
dups = train.drop(columns=[ID_COL, TARGET_COL]).duplicated()
n_dup = dups.sum()
print(f"  exact duplicate feature rows: {n_dup}")

if n_dup > 0:
    dup_mask = train.drop(columns=[ID_COL, TARGET_COL]).duplicated(keep=False)
    dup_rows = train[dup_mask].sort_values(feat_cols)
    conflicts = 0
    grouped = dup_rows.groupby(feat_cols)
    for _, grp in grouped:
        if grp[TARGET_COL].nunique() > 1:
            conflicts += len(grp)
    print(f"  duplicate rows w/ conflicting labels: {conflicts}")

# ---- comorbidity x class ----
print("\n--- comorbidity vs covid_result ---")
ct = pd.crosstab(train["comorbidity"].fillna("None"), train[TARGET_COL])
ct["total"] = ct.sum(axis=1)
ct["pos_rate"] = ct[1] / ct["total"]
print(ct.to_string())

print("\n" + "=" * 60)
print("EDA DONE")
print("=" * 60)
