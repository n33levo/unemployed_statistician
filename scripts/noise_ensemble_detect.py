"""
Ensemble Noise Detection — 3-method consensus to find mislabelled rows.

Methods:
  1. Cleanlab  — confident learning: flags rows where the label disagrees
                 with the model's confident prediction
  2. AUM (Area Under the Margin) via CatBoost — samples that are hard to
                 learn throughout training have low AUM → likely mislabelled
  3. Dataset Cartography — tracks per-sample confidence + variability across
                 training; "hard-to-learn" samples are likely mislabelled

Consensus tiers:
  Tier 1 (definite flips)   — flagged by ALL 3 methods  → flip label
  Tier 2 (probable noise)   — flagged by 2 methods       → downweight
  Tier 3 (clean)            — flagged by 0 or 1 method   → full weight

Output:
  reports/tiered_noise_labels.json — tier assignments + corrected labels
  5-fold CV results showing improvement from correction
"""
from __future__ import annotations

import json
import os
import sys
import warnings

from runtime_env import configure_runtime

configure_runtime()

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from cleanlab.filter import find_label_issues
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ID_COL, TARGET_COL, TRAIN_PATH, SEED


# ── Feature engineering ──────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
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


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        sparse_threshold=0.0,
    )
    return pre.set_output(transform="pandas")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("ENSEMBLE NOISE DETECTION")
    print("=" * 60)

    train_raw = pd.read_csv(TRAIN_PATH)
    train_raw[TARGET_COL] = train_raw[TARGET_COL].astype(int)
    train_eng = add_features(train_raw)

    y_all = train_raw[TARGET_COL].to_numpy()
    X_df = train_eng.drop(columns=[ID_COL, TARGET_COL])
    n = len(y_all)

    # ── Compute OOF probabilities (needed by Cleanlab + Cartography) ─────────
    print("\nComputing LR OOF probabilities for Cleanlab...")
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(n, dtype=float)
    for tr_idx, va_idx in outer_cv.split(X_df, y_all):
        pre = build_preprocessor(X_df.iloc[tr_idx])
        X_tr = pre.fit_transform(X_df.iloc[tr_idx].reset_index(drop=True))
        X_va = pre.transform(X_df.iloc[va_idx].reset_index(drop=True))
        lr = LogisticRegression(max_iter=2500, C=0.25, random_state=SEED)
        lr.fit(X_tr, y_all[tr_idx])
        oof_probs[va_idx] = lr.predict_proba(X_va)[:, 1]

    pred_probs_matrix = np.column_stack([1 - oof_probs, oof_probs])

    # ── Step 1: Cleanlab ─────────────────────────────────────────────────────
    print("\n[Step 1] Cleanlab...")
    # find_label_issues returns a boolean mask; np.where extracts the true indices
    cl_mask = find_label_issues(y_all, pred_probs_matrix)
    issue_idx_cl = set(np.where(cl_mask)[0].tolist())
    print(f"  Cleanlab flagged {len(issue_idx_cl)} samples ({len(issue_idx_cl)/n*100:.1f}%)")

    # ── Step 2: AUM via CatBoost ─────────────────────────────────────────────
    print("\n[Step 2] AUM via CatBoost...")
    # AUM = area under the margin throughout training
    # Margin for sample i at step t = P(correct class) - max P(other class)
    # Low/negative AUM → model consistently struggles → likely mislabelled

    # Encode categoricals for CatBoost
    X_cb = X_df.copy()
    for col in X_cb.select_dtypes(include="object").columns:
        X_cb[col] = X_cb[col].astype("category").cat.codes

    checkpoints = [100, 200, 300, 400, 500]
    margins_sum = np.zeros(n, dtype=float)

    cb = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
        thread_count=1, allow_writing_files=False,
        save_snapshot=False,
    )
    cb.fit(X_cb.values, y_all)

    # Use leaf values at different iteration snapshots to estimate margin
    # Simpler: use prediction at each checkpoint as proxy for AUM
    for ckpt in checkpoints:
        cb_ckpt = CatBoostClassifier(
            iterations=ckpt, depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
            thread_count=1, allow_writing_files=False,
        )
        cb_ckpt.fit(X_cb.values, y_all)
        probs_ckpt = cb_ckpt.predict_proba(X_cb.values)
        # Margin = P(true class) - P(other class)
        margins = probs_ckpt[np.arange(n), y_all] - probs_ckpt[np.arange(n), 1 - y_all]
        margins_sum += margins

    aum_scores = margins_sum / len(checkpoints)
    AUM_THRESHOLD = 0.40
    issue_idx_aum = set(np.where(aum_scores < AUM_THRESHOLD)[0])
    print(f"  AUM: {len(issue_idx_aum)} samples with score < {AUM_THRESHOLD} ({len(issue_idx_aum)/n*100:.1f}%)")
    print(f"  AUM mean={aum_scores.mean():.3f}, min={aum_scores.min():.3f}, max={aum_scores.max():.3f}")

    # ── Step 3: Dataset Cartography ─────────────────────────────────────────
    print("\n[Step 3] Dataset Cartography (5 seeds × 5 folds)...")
    # For each sample, track mean confidence and variability across multiple runs
    # Hard-to-learn: low mean confidence in true label → likely mislabelled

    confidence_runs = []
    for seed_offset in range(5):
        cv_dc = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + seed_offset * 100)
        run_conf = np.zeros(n, dtype=float)
        for tr_idx, va_idx in cv_dc.split(X_df, y_all):
            pre = build_preprocessor(X_df.iloc[tr_idx])
            X_tr = pre.fit_transform(X_df.iloc[tr_idx].reset_index(drop=True))
            X_va = pre.transform(X_df.iloc[va_idx].reset_index(drop=True))
            lr = LogisticRegression(max_iter=2500, C=0.25, random_state=SEED + seed_offset)
            lr.fit(X_tr, y_all[tr_idx])
            probs_va = lr.predict_proba(X_va)[:, 1]
            # Confidence = P(true label)
            run_conf[va_idx] = np.where(y_all[va_idx] == 1, probs_va, 1 - probs_va)
        confidence_runs.append(run_conf)

    conf_matrix = np.stack(confidence_runs, axis=1)  # (n, 5)
    mean_conf = conf_matrix.mean(axis=1)
    std_conf = conf_matrix.std(axis=1)

    # Hard-to-learn: mean confidence < 0.45
    HARD_THRESHOLD = 0.45
    issue_idx_dc = set(np.where(mean_conf < HARD_THRESHOLD)[0])

    easy = (mean_conf >= 0.55).sum()
    ambiguous = ((mean_conf >= 0.45) & (mean_conf < 0.55)).sum()
    hard = (mean_conf < 0.45).sum()
    print(f"  Easy-to-learn : {easy} ({easy/n*100:.1f}%)")
    print(f"  Ambiguous     : {ambiguous} ({ambiguous/n*100:.1f}%)")
    print(f"  Hard-to-learn : {hard} ({hard/n*100:.1f}%)  ← likely flipped")

    # ── Step 4: Consensus ────────────────────────────────────────────────────
    print("\n[Step 4] Consensus scoring...")

    vote_count = np.zeros(n, dtype=int)
    for idx_set in [issue_idx_cl, issue_idx_aum, issue_idx_dc]:
        for i in idx_set:
            vote_count[i] += 1

    tier1_idx = np.where(vote_count == 3)[0].tolist()  # all 3 agree → definite flip
    tier2_idx = np.where(vote_count == 2)[0].tolist()  # 2 agree → probable
    tier3_idx = np.where(vote_count <= 1)[0].tolist()  # 0 or 1 → clean

    print(f"\n  Tier 1 (definite flips, correct):   {len(tier1_idx)} ({len(tier1_idx)/n*100:.1f}%)")
    print(f"  Tier 2 (probable flips, downweight): {len(tier2_idx)} ({len(tier2_idx)/n*100:.1f}%)")
    print(f"  Tier 3 (clean, full weight):         {len(tier3_idx)} ({len(tier3_idx)/n*100:.1f}%)")

    # Overlap with cleanlab
    overlap = len(set(tier1_idx) & issue_idx_cl)
    print(f"\n  Overlap of Tier 1 with Cleanlab flags: {overlap}/{len(tier1_idx)}")

    # Corrected labels: flip Tier 1 to what model thinks they should be
    corrected_labels = y_all.copy()
    for i in tier1_idx:
        corrected_labels[i] = 1 - y_all[i]  # flip the label
    print(f"  Of {len(tier1_idx)} Tier 1 samples, {len(tier1_idx)} will actually have their label changed")

    # ── Step 5: 5-fold CV with tiered correction ─────────────────────────────
    print("\n[Step 5] 5-fold CV with tiered label correction...")
    tier1_set = set(tier1_idx)
    tier2_set = set(tier2_idx)
    tiers_arr = np.full(n, 3, dtype=int)
    for i in tier1_set:
        tiers_arr[i] = 1
    for i in tier2_set:
        tiers_arr[i] = 2

    oof_tiered_lr = np.zeros(n, dtype=float)
    oof_tiered_cb = np.zeros(n, dtype=float)
    oof_ref = np.zeros(n, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X_df, y_all), start=1):
        fold_tiers = tiers_arr[tr_idx]
        fold_corr = corrected_labels[tr_idx]
        y_tr = y_all[tr_idx].copy()
        y_tr_corr = y_tr.copy()
        t1_mask = fold_tiers == 1
        t2_mask = fold_tiers == 2
        y_tr_corr[t1_mask] = fold_corr[t1_mask]

        # Sample weights: T2 at 0.1 (found optimal in push_corrected_cv)
        sw = np.ones(len(y_tr), dtype=float)
        sw[t2_mask] = 0.1

        pre = build_preprocessor(X_df.iloc[tr_idx])
        X_tr_pre = pre.fit_transform(X_df.iloc[tr_idx].reset_index(drop=True))
        X_va_pre = pre.transform(X_df.iloc[va_idx].reset_index(drop=True))

        # LR corrected
        lr = LogisticRegression(max_iter=2500, C=0.25, random_state=SEED)
        lr.fit(X_tr_pre, y_tr_corr, sample_weight=sw)
        oof_tiered_lr[va_idx] = lr.predict_proba(X_va_pre)[:, 1]

        # CatBoost corrected (for comparison)
        X_tr_cb = X_df.iloc[tr_idx].copy()
        X_va_cb = X_df.iloc[va_idx].copy()
        for col in X_tr_cb.select_dtypes(include="object").columns:
            X_tr_cb[col] = X_tr_cb[col].astype("category").cat.codes
            X_va_cb[col] = X_va_cb[col].astype("category").cat.codes
        cb_fold = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
            thread_count=1, allow_writing_files=False,
        )
        cb_fold.fit(X_tr_cb.values, y_tr_corr, sample_weight=sw)
        oof_tiered_cb[va_idx] = cb_fold.predict_proba(X_va_cb.values)[:, 1]

        # Reference: LR with original labels (no correction)
        lr_ref = LogisticRegression(max_iter=2500, C=0.25, random_state=SEED)
        lr_ref.fit(X_tr_pre, y_tr)
        oof_ref[va_idx] = lr_ref.predict_proba(X_va_pre)[:, 1]

        print(f"  fold {fold}: tier1={t1_mask.sum()}  tier2={t2_mask.sum()}  tier3={(fold_tiers==3).sum()}")

    def report(name, probs):
        acc = accuracy_score(y_all, (probs >= 0.5).astype(int))
        auc = roc_auc_score(y_all, probs)
        print(f"\n{name}:")
        print(f"  accuracy @ 0.50 = {acc:.4f}")
        print(f"  AUC             = {auc:.4f}")
        return acc

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    report("logreg_tiered", oof_tiered_lr)
    report("catboost_tiered", oof_tiered_cb)
    report("logreg_ref (no correction)", oof_ref)

    # ── Save tier assignments ─────────────────────────────────────────────────
    output = {
        "tier1_indices": tier1_idx,
        "tier2_indices": tier2_idx,
        "tier3_indices": tier3_idx,
        "corrected_labels": corrected_labels.tolist(),
        "original_labels": y_all.tolist(),
        "description": (
            "Tier 1: all 3 methods agree — label flipped. "
            "Tier 2: 2 methods agree — downweight to 0.1. "
            "Tier 3: clean — full weight 1.0."
        ),
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "tiered_noise_labels.json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\n  Saved tier assignments → {out_path}")

    print("\n" + "=" * 60)
    print("ENSEMBLE NOISE DETECTION DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
