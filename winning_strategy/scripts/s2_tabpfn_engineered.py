"""
Strategy 2: TabPFN + corrected labels + engineered features.
  - Same Tier 1 correction as baseline
  - Add clinically motivated features (symptom_count, low_oxygen, etc.)
  - Compare OOF accuracy vs the raw-feature baseline (S1 = 0.9053)
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


def add_features(df):
    """Clinically motivated feature engineering."""
    df = df.copy()
    df["comorbidity"] = df["comorbidity"].fillna("None")
    symptom_cols = [
        "fever", "dry_cough", "sore_throat", "fatigue",
        "headache", "shortness_of_breath", "loss_of_smell", "loss_of_taste",
    ]
    df["symptom_count"] = df[symptom_cols].sum(axis=1)
    df["exposure_count"] = df[["travel_history", "contact_with_patient"]].sum(axis=1)
    df["low_oxygen_95"] = (df["oxygen_level"] < 95).astype(int)
    df["low_oxygen_92"] = (df["oxygen_level"] < 92).astype(int)
    df["fever_temp_38"] = (df["body_temperature"] >= 38).astype(int)
    df["smell_or_taste_loss"] = (
        (df["loss_of_smell"] == 1) | (df["loss_of_taste"] == 1)
    ).astype(int)
    df["smell_and_taste_loss"] = (
        (df["loss_of_smell"] == 1) & (df["loss_of_taste"] == 1)
    ).astype(int)
    df["respiratory_distress"] = (
        (df["shortness_of_breath"] == 1) & (df["oxygen_level"] < 95)
    ).astype(int)
    df["temp_oxygen_gap"] = df["body_temperature"] - (df["oxygen_level"] / 10.0)
    df["age_risk"] = ((df["age"] >= 60) | (df["age"] <= 10)).astype(int)
    df["oxygen_age_risk"] = df["age_risk"] * df["low_oxygen_95"]
    df["symptom_exposure_product"] = df["symptom_count"] * (df["exposure_count"] + 1)
    df["oxygen_squared"] = (df["oxygen_level"] - 95.0) ** 2
    df["temp_fever_oxygen"] = df["fever_temp_38"] * df["low_oxygen_95"]
    df["high_symptom_load"] = (df["symptom_count"] >= 5).astype(int)
    df["no_exposure"] = (df["exposure_count"] == 0).astype(int)
    return df


# ── Load + engineer ──
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train = add_features(train)
test = add_features(test)

# ── One-hot encode ──
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

for idx in tiers["tier1_indices"]:
    train_enc.at[idx, TARGET_COL] = tiers["corrected_labels"][idx]

y = train_enc[TARGET_COL].astype(int).values
X = train_enc.drop(columns=[ID_COL, TARGET_COL])
X_test = test_enc.drop(columns=[ID_COL])

print(f"Train shape: {X.shape} (was 1500x20 in baseline)")
print(f"Features ({X.shape[1]}): {X.columns.tolist()}")

# ── 5-fold CV ──
print("\n" + "=" * 60)
print("5-FOLD CV (TabPFN + engineered features + corrected labels)")
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
orig_labels = np.array(tiers["original_labels"])
oof_acc_noisy = accuracy_score(orig_labels, oof_preds)
print(f"\nOOF accuracy (vs corrected labels): {oof_acc:.4f}  [baseline was 0.9053]")
print(f"OOF accuracy (vs original noisy labels): {oof_acc_noisy:.4f}  [baseline was 0.7700]")

# Threshold sweep
best_t, best_a = 0.5, oof_acc
for t in np.arange(0.25, 0.75, 0.01):
    a = accuracy_score(y, (oof_probs >= t).astype(int))
    if a > best_a:
        best_a = a
        best_t = t
print(f"Best threshold: {best_t:.2f}, accuracy: {best_a:.4f}")

# ── Full train → test prediction ──
print("\n" + "=" * 60)
print("FULL TRAIN → TEST PREDICTION")
print("=" * 60)

clf_full = TabPFNClassifier(random_state=42, device="cpu", ignore_pretraining_limits=True)
clf_full.fit(X, y)

predictions = []
probabilities = []
for i in range(0, len(X_test), 1000):
    predictions.append(clf_full.predict(X_test.iloc[i:i+1000]))
    probabilities.append(clf_full.predict_proba(X_test.iloc[i:i+1000])[:, 1])

predictions = np.concatenate(predictions)
probabilities = np.concatenate(probabilities)

n0, n1 = (predictions == 0).sum(), (predictions == 1).sum()
print(f"Predictions: {n0} zeros, {n1} ones ({n1/len(predictions)*100:.1f}% positive)")

# Save
sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: predictions.astype(int)})
out_path = os.path.join(os.path.dirname(__file__), "..", "submissions", "s2_tabpfn_engineered.csv")
sub.to_csv(out_path, index=False)
print(f"Saved → {out_path}")

np.save(os.path.join(os.path.dirname(__file__), "..", "reports", "s2_test_probs.npy"), probabilities)
np.save(os.path.join(os.path.dirname(__file__), "..", "reports", "s2_oof_probs.npy"), oof_probs)

# Compare with baseline
s1_probs = np.load(os.path.join(os.path.dirname(__file__), "..", "reports", "s1_test_probs.npy"))
s1_preds = (s1_probs >= 0.5).astype(int)
s2_preds = predictions.astype(int)
diffs = (s1_preds != s2_preds).sum()
print(f"\nDifferences vs S1 baseline: {diffs} / {len(s2_preds)} test predictions differ")

print("\n" + "=" * 60)
print("S2 DONE")
print("=" * 60)
