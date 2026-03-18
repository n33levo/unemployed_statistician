import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from config import TRAIN_PATH, TEST_PATH, ID_COL, TARGET_COL, CAT_COLS
from tabpfn import TabPFNClassifier
from sklearn.model_selection import cross_val_score

# Note on using TabPFN.
# We are runnign TabPFN locally. For this, after installing it, when you run the script for the first time, you will receive an error related to authentication
# To fix this, you need to accept the license agreement and authenticate with Hugging Face. You can do this by running the command provided in the error message
# Since we are running locally, we can use the cpu version, to avoid the gpu memory issues. This may cause longer training and some slight slowing down, but it does work.

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train["comorbidity"] = train["comorbidity"].fillna("None")  # Fill missing values in 'comorbidity' with "None"
test["comorbidity"] = test["comorbidity"].fillna("None")  # Fill missing values in 'comorbidity' with "None"


def pre_process(train):
    """
    Given the train dataframe, perform one-hot encoding on the categorical columns and return the transformed dataframe.
    """
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(train[CAT_COLS])

    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(CAT_COLS), index=train.index)
    
    new_train = pd.concat([train.drop(CAT_COLS, axis=1), one_hot_df], axis=1)
    
    return new_train


def corrected_classifier(train):
    """
    Given the train dataframe, read the tiered noise labels. Correct tier 1 labels, and update the train df
    """
    in_path = os.path.join(os.path.dirname(__file__), "..", "reports", "tiered_noise_labels.json")
    with open(in_path, "r") as f:
        tiered_noise_labels = json.load(f)
    
    t1 = tiered_noise_labels["tier1_indices"]
    corrected = tiered_noise_labels["corrected_labels"]
    
    for tier1 in t1:
        train.at[tier1, TARGET_COL] = corrected[tier1]
    
    return train


def tabpfn_classifier(train):
    """
    Train the TabPFN Classifier on the training data. Evaulate model with 5-fold cv
    """
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


def tabpfn_classifier_test(train, test):
    """
    Train the TabPFNClassifier on the provided training data and make predictions on the test data. Save the predictions to a CSV file.
    """
    Y = train[TARGET_COL]
    X = train.drop(columns=[ID_COL, TARGET_COL])
    X_test = test.drop(columns=[ID_COL])
    
    clf = TabPFNClassifier(random_state=42, device="cpu", ignore_pretraining_limits=True)
    clf.fit(X, Y)
    
    
    # Batch prediction to avoid memory issues
    batch_size = 1000
    predictions = []
    for i in range(0, len(X_test), batch_size):
        print(f"Predicting batch {i // batch_size + 1} of {len(X_test) // batch_size + 1}")
        batch = clf.predict(X_test.iloc[i:i + batch_size])
        predictions.append(batch)

    predictions = np.concatenate(predictions)

    results = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET_COL: predictions
    })

    results.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    encoded_train = pre_process(train)
    corrected_train = corrected_classifier(encoded_train)
    encoded_test = pre_process(test)

    tabpfn_classifier_test(corrected_train, encoded_test)
