import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from config import TRAIN_PATH, TEST_PATH, ID_COL, TARGET_COL, CAT_COLS
from tabpfn import TabPFNClassifier
from sklearn.model_selection import cross_val_score


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train["comorbidity"] = train["comorbidity"].fillna("None")  # Fill missing values in 'comorbidity' with "None"

def pre_process(train):
    """
    Given the train dataframe, perform one-hot encoding on the categorical columns and return the transformed dataframe.
    """
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(train[CAT_COLS])

    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(CAT_COLS))
    
    new_train = pd.concat([train.drop(CAT_COLS, axis=1), one_hot_df], axis=1)

    # print(f"One-Hot Encoded Data using Scikit-Learn:\n{new_train}\n")
    # print(new_train.columns)

    new_train[TARGET_COL] = train[TARGET_COL].astype(int)

    return new_train


def tabpfn_classifier(train):
    Y = train[TARGET_COL]
    X = train.drop(columns=[ID_COL, TARGET_COL])
    
    clf = TabPFNClassifier(random_state=42)
    
    cross_val = cross_val_score(clf, X, Y, cv=5, scoring="accuracy")
    
    print ("=" * 60)
    print("TabPFN CLASSIFIER")
    print ("=" * 60)
    for fold, score in enumerate(cross_val):
        print(f"Fold {fold + 1}: Accuracy = {score:.4f}")
    
    print(f"\nOverall TabPFN Cross-Validation Accuracy: {cross_val.mean():.4f}")
    print(f"Overall TabPFN Cross-Validation Std Dev: {cross_val.std():.4f}")


if __name__ == "__main__":
    new_train = pre_process(train)
    print(f"Columns after One-Hot Encoding: {new_train.columns.tolist()}")
    tabpfn_classifier(new_train)