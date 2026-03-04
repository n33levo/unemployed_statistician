"""
the course sample file has 15 rows (patient_id 111-125) that match
test ids 1-15 with known ground truth labels. we can use this to
sanity-check any model's predictions on those 15 test rows.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from config import (
    TRAIN_PATH, TEST_PATH, ID_COL, TARGET_COL,
    NUM_COLS, SEED, BASE_DIR,
)

SAMPLE_PATH = BASE_DIR / "docs" / "sample_covid19_data.csv"

train = pd.read_csv(TRAIN_PATH)
train[TARGET_COL] = train[TARGET_COL].astype(int)
train["comorbidity"] = train["comorbidity"].fillna("Unknown")

test = pd.read_csv(TEST_PATH)
test["comorbidity"] = test["comorbidity"].fillna("Unknown")

sample = pd.read_csv(SAMPLE_PATH)

print("=" * 60)
print("SANITY CHECK: 15 known test labels")
print("=" * 60)

# map patient_id 111-125 -> test id 1-15
sample["mapped_id"] = sample["patient_id"] - 110
known = sample[["mapped_id", "covid_result"]].rename(
    columns={"mapped_id": ID_COL, "covid_result": "true_label"}
)
known["true_label"] = known["true_label"].astype(int)

print(f"\nknown labels (test ids 1-15):")
print(known.to_string(index=False))

# train a catboost on full train, predict on test ids 1-15
y = train[TARGET_COL].values
X = train.drop(columns=[ID_COL, TARGET_COL])
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

X_test = test.drop(columns=[ID_COL])

cb = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.03,
    l2_leaf_reg=3.0, loss_function="Logloss",
    verbose=False, random_seed=SEED,
)
cb.fit(X, y, cat_features=cat_idx)

test_probs = cb.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

# check the 15 known ones
check_ids = known[ID_COL].values
mask = test[ID_COL].isin(check_ids)
check_df = test.loc[mask, [ID_COL]].copy()
check_df["predicted"] = test_preds[mask.values]
check_df["prob"] = test_probs[mask.values]
check_df = check_df.merge(known, on=ID_COL)

print(f"\npredictions vs known labels:")
print(f"{'id':>4} {'pred':>5} {'true':>5} {'prob':>6} {'match':>6}")
correct = 0
for _, row in check_df.iterrows():
    match = "ok" if row["predicted"] == row["true_label"] else "WRONG"
    if match == "ok":
        correct += 1
    print(f"{int(row[ID_COL]):4d} {int(row['predicted']):5d} {int(row['true_label']):5d} {row['prob']:6.3f} {match:>6}")

acc = correct / len(check_df)
print(f"\nsanity check accuracy: {correct}/{len(check_df)} = {acc:.1%}")
if acc >= 0.8:
    print("looks reasonable")
elif acc >= 0.6:
    print("some mismatches -- worth investigating")
else:
    print("something might be off w/ the pipeline")

print("\n" + "=" * 60)
print("SANITY CHECK DONE")
print("=" * 60)
