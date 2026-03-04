"""
detect suspected label issues using out-of-fold predicted probabilities.
uses logistic regression for OOF probs, then ranks by self-confidence.
also tries cleanlab if available.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from config import TRAIN_PATH, ID_COL, TARGET_COL, NUM_COLS, SEED


def main():
    train = pd.read_csv(TRAIN_PATH)
    train[TARGET_COL] = train[TARGET_COL].astype(int)
    train["comorbidity"] = train["comorbidity"].fillna("Unknown")

    y = train[TARGET_COL].values
    X = train.drop(columns=[ID_COL, TARGET_COL])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), NUM_COLS),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ], remainder="passthrough")

    model = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=600, random_state=SEED))])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)

    self_conf = np.where(y == 1, probs[:, 1], probs[:, 0])
    rank = np.argsort(self_conf)

    print("=" * 60)
    print("LABEL NOISE DETECTION (via OOF self-confidence)")
    print("=" * 60)

    for thresh in [0.3, 0.35, 0.4, 0.45]:
        n_sus = (self_conf < thresh).sum()
        print(f"  self_confidence < {thresh}: {n_sus} samples ({n_sus/len(y):.1%})")

    print(f"\ntop 20 most suspicious samples (lowest self-confidence):")
    print(f"{'idx':>5} {'id':>5} {'label':>5} {'P(0)':>6} {'P(1)':>6} {'self_conf':>9}")
    for i in rank[:20]:
        row = train.iloc[i]
        print(f"{i:5d} {int(row[ID_COL]):5d} {int(y[i]):5d} {probs[i,0]:6.3f} {probs[i,1]:6.3f} {self_conf[i]:9.3f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "suspected_label_issues.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        from cleanlab.filter import find_label_issues
        issues = find_label_issues(
            labels=y, pred_probs=probs,
            return_indices_ranked_by="self_confidence",
            n_jobs=1,
        )
        print(f"\ncleanlab found {len(issues)} label issues ({len(issues)/len(y):.1%} of train)")
        print(f"top 10 cleanlab suspects: {issues[:10].tolist()}")

        with open(out_path, "w") as f:
            json.dump({
                "method": "cleanlab_confident_learning",
                "n_issues": int(len(issues)),
                "fraction": round(len(issues) / len(y), 4),
                "issue_indices": issues.tolist(),
            }, f, indent=2)
        print(f"saved issue indices to {out_path}")
    except ImportError:
        print("\ncleanlab not installed -- skipping. install w/ `pip install cleanlab`")
        low_conf_idx = rank[:int(len(y) * 0.10)].tolist()
        with open(out_path, "w") as f:
            json.dump({
                "method": "oof_self_confidence_bottom_10pct",
                "n_issues": len(low_conf_idx),
                "fraction": round(len(low_conf_idx) / len(y), 4),
                "issue_indices": low_conf_idx,
            }, f, indent=2)
        print(f"saved bottom 10% low-confidence indices to {out_path}")

    print("\n" + "=" * 60)
    print("NOISE CHECK DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
