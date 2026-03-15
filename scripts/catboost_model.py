import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from config import TRAIN_PATH, ID_COL, TARGET_COL, SEED

train = pd.read_csv(TRAIN_PATH)
train[TARGET_COL] = train[TARGET_COL].astype(int)
train['comorbidity'] = train['comorbidity'].fillna('None')

X = train.drop(columns=[ID_COL, TARGET_COL])
y = train[TARGET_COL].values
cat_features = ['gender', 'comorbidity']

def objective(trial):
    params = {
        "iterations": 500,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "loss_function": "Logloss",
        "verbose": False,
        "allow_writing_files": False,
        "random_seed": SEED
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_va, y_va), early_stopping_rounds=50)
        
        preds = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, preds)
        cv_scores.append(auc)

    return np.mean(cv_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
best_params.update({"iterations": 1000, "verbose": 0, "allow_writing_files": False})

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cb_probs = np.zeros(len(y))

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
    model = CatBoostClassifier(**best_params)
    model.fit(X.iloc[tr_idx], y[tr_idx], cat_features=cat_features)
    cb_probs[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]

cb_preds = (cb_probs >= 0.5).astype(int)
cb_acc = accuracy_score(y, cb_preds)
cb_auc = roc_auc_score(y, cb_probs)
cb_ll = log_loss(y, cb_probs)

imp = model.get_feature_importance()
feat_names = X.columns.tolist()

output_file = "reports/catboost_output.txt"
with open(output_file, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("CATBOOST + OPTUNA RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"best parameters:\n")
    for key, value in best_params.items():
        f.write(f"- {key}: {value}\n")
    
    f.write("\n" + "-" * 30 + "\n\n")
    f.write(f"- accuracy: {cb_acc:.4f}\n")
    f.write(f"- AUC: {cb_auc:.4f}\n")
    f.write(f"- logloss: {cb_ll:.4f}\n")

    f.write("\n--- CatBoost feature importance (last fold) ---\n")
    for name, score in sorted(zip(feat_names, imp), key=lambda x: -x[1]):
        if score > 1.0:
            f.write(f"- {name:30s} {score:6.2f}\n")
