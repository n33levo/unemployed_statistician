from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "covid-19-patient-diagnosis-based-on-symtoms"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

ID_COL = "id"
TARGET_COL = "covid_result"

NUM_COLS = ["age", "oxygen_level", "body_temperature"]
SYMPTOM_COLS = [
    "fever", "dry_cough", "sore_throat", "fatigue",
    "headache", "shortness_of_breath", "loss_of_smell", "loss_of_taste",
]
BINARY_COLS = ["travel_history", "contact_with_patient", "chest_pain"]
CAT_COLS = ["gender", "comorbidity"]

SEED = 42
