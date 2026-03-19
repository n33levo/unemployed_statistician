from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, \
    roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, cross_validate, StratifiedKFold
import pandas as pd
from config import (SEED, TRAIN_PATH, TEST_PATH, ID_COL, TARGET_COL, NUM_COLS, SYMPTOM_COLS,
                    CAT_COLS,
                    SAMPLE_SUB_PATH)

# Read Train and Test data
train = pd.read_csv(TRAIN_PATH)
train['comorbidity'] = train['comorbidity'].fillna('None')
test = pd.read_csv(TEST_PATH)
test['comorbidity'] = test['comorbidity'].fillna('None')


# Remove NA and set indicators
train = train.dropna()
train = pd.get_dummies(train, drop_first=True)
test = test.dropna()
test = pd.get_dummies(test, drop_first=True)

# Set X and Y
X_train = train.drop(columns=[TARGET_COL])
Y_train = train[TARGET_COL]
X_test = test
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=200))
])

# CV Summary
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = cross_validate(pipeline, X_train, Y_train, cv=cv,
                         scoring=["accuracy", "precision_weighted",
                                  "recall_weighted", "roc_auc"])

lr_probs = cross_val_predict(pipeline, X_train, Y_train, cv=cv, method="predict_proba", n_jobs=1)[
    :, 1]
lr_preds = (lr_probs >= 0.5).astype(int)

lr_acc = accuracy_score(Y_train, lr_preds)
lr_auc = roc_auc_score(Y_train, lr_probs)
lr_ll = log_loss(Y_train, lr_probs)

print("=" * 60)
print("CROSS VALIDATION SUMMARY")
print("=" * 60)
for i, acc in enumerate(results["test_accuracy"], 1):
    print(f"  Fold {i} Accuracy: {acc:.4f}")
print("-" * 60)
print(f"Mean Accuracy:  {lr_acc:.4f}")
print(f"Std Deviation:  {results['test_accuracy'].std():.4f}")
print(f"Precision:      {results['test_precision_weighted'].mean():.4f}")
print(f"Recall:         {results['test_recall_weighted'].mean():.4f}")
print(f"AUC:            {lr_auc:.4f}")
print(f"Log-loss:       {lr_ll:.4f}")

# Fit Model
pipeline.fit(X_train, Y_train)
model = pipeline.named_steps["model"]
scaler = pipeline.named_steps["scaler"]
Y_predict = pipeline.predict(X_test)
