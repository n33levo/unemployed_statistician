"""
baseline cross-validation with CatBoost and logistic regression.
also does a quick threshold sweep to see if 0.5 is optimal.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from catboost import CatBoostClassifier
from config import TRAIN_PATH, ID_COL, TARGET_COL, NUM_COLS, SEED

train = pd.read_csv(TRAIN_PATH)
train[TARGET_COL] = train[TARGET_COL].astype(int)
train["comorbidity"] = train["comorbidity"].fillna("None")

y = train[TARGET_COL].values
X = train.drop(columns=[ID_COL, TARGET_COL])

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("=" * 60)
print("BASELINE CROSS-VALIDATION")
print("=" * 60)

# ---- logistic regression ----
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

lr = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=600, random_state=SEED))])
lr_probs = cross_val_predict(lr, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
lr_preds = (lr_probs >= 0.5).astype(int)

lr_acc = accuracy_score(y, lr_preds)
lr_auc = roc_auc_score(y, lr_probs)
lr_ll = log_loss(y, lr_probs)
print(f"\nLogistic Regression (OOF):")
print(f"  accuracy = {lr_acc:.4f}")
print(f"  AUC      = {lr_auc:.4f}")
print(f"  logloss  = {lr_ll:.4f}")

# ---- catboost ----
cb_probs = np.zeros(len(y))
for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
    cb = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.03,
        l2_leaf_reg=3.0, loss_function="Logloss",
        verbose=False, random_seed=SEED,
    )
    cb.fit(X.iloc[tr_idx], y[tr_idx], cat_features=cat_idx)
    cb_probs[va_idx] = cb.predict_proba(X.iloc[va_idx])[:, 1]

cb_preds = (cb_probs >= 0.5).astype(int)
cb_acc = accuracy_score(y, cb_preds)
cb_auc = roc_auc_score(y, cb_probs)
cb_ll = log_loss(y, cb_probs)
print(f"\nCatBoost (OOF, native categoricals):")
print(f"  accuracy = {cb_acc:.4f}")
print(f"  AUC      = {cb_auc:.4f}")
print(f"  logloss  = {cb_ll:.4f}")

# ---- threshold sweep ----
print("\n--- threshold sweep (CatBoost OOF probs) ---")
thresholds = np.arange(0.30, 0.71, 0.01)
accs = [accuracy_score(y, (cb_probs >= t).astype(int)) for t in thresholds]
best_idx = np.argmax(accs)
best_t = thresholds[best_idx]
best_a = accs[best_idx]

print(f"  acc @ 0.50 = {accuracy_score(y, (cb_probs >= 0.50).astype(int)):.4f}")
print(f"  best threshold = {best_t:.2f}, acc = {best_a:.4f}")
if abs(best_t - 0.50) > 0.02:
    print(f"  -> threshold tuning gives +{best_a - cb_acc:.4f} over default 0.5")
else:
    print(f"  -> 0.5 is close to optimal")

# feature importance from catboost (last fold model)
print("\n--- CatBoost feature importance (last fold) ---")
imp = cb.get_feature_importance()
feat_names = X.columns.tolist()
for name, score in sorted(zip(feat_names, imp), key=lambda x: -x[1]):
    if score > 1.0:
        print(f"  {name:30s} {score:6.2f}")

print("\n" + "=" * 60)
print("BASELINE DONE")
print("=" * 60)
