import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from config import TRAIN_PATH, TEST_PATH, ID_COL, TARGET_COL, NUM_COLS, SYMPTOM_COLS, CAT_COLS
from tabpfn import TabPFNClassifier
from sklearn.model_selection import cross_val_score


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

def pre_process(train):
    """
    Given the train and test dataframes, perform one-hot encoding on the categorical columns and return the transformed dataframes.
    """
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(train[CAT_COLS])

    one_hot_df = pd.DataFrame(one_hot_encoded, 
                            columns=encoder.get_feature_names_out(CAT_COLS))
    
    new_train = pd.concat([train.drop(CAT_COLS, axis=1), one_hot_df], axis=1)

    # print(f"One-Hot Encoded Data using Scikit-Learn:\n{new_train}\n")
    # print(new_train.columns)

    new_train[TARGET_COL] = train[TARGET_COL].astype(int)

    return new_train


def tabpfn_classifier(train):
    y_train = train[TARGET_COL]
    X_train = train.drop(columns=[ID_COL, TARGET_COL])
    
    clf = TabPFNClassifier(random_state=42)
    
    cross_val = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"Cross-Validation Scores: {cross_val}")
    print(f"Mean Accuracy: {cross_val.mean():.3f} (+/- {cross_val.std() * 2:.3f})")


if __name__ == "__main__":
    new_train = pre_process(train)
    print(new_train.head())
    tabpfn_classifier(new_train)