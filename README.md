# STA314 Final Project -- COVID-19 Diagnosis Prediction

predicting whether a patient tested positive for covid from symptoms, demographics, and some clinical stuff. binary classification, ~1500 train rows, ~3500 test rows.

## quick summary

- **target**: `covid_result` (0 or 1)
- **metric**: accuracy (kaggle leaderboard)
- **train**: ~1500 rows, **test**: ~3500 rows
- **submission**: csv with `id` and `covid_result`

## what we know so far

Xin mentioned in the project doc that he intentionally flipped some labels and added data errors to the training set. so dealing w/ that is prob our biggest edge -- most teams will just ignore it.

Other observations from the EDA (see `scripts/eda.py`):
- train numerics have weirdly precise decimals (like `98.5433231...`), test is rounded. tree models dont care but worth noting
- `comorbidity` is missing for ~53% of train. we should treat that as its own category ("Unknown"), not drop it
- theres a small labeled sample in the course materials that overlaps w/ test ids 1-15 -- useful for sanity checking predictions

## approach (high level)

1. **EDA + data profiling** -- understand distributions, missingness, class balance, train/test shift
2. **label noise detection** -- use out-of-fold probabilities + confident learning (cleanlab) to flag suspected mislabels
3. **handle noisy labels** -- try removing, reweighting, or smoothing the suspicious ones and see what helps via CV
4. **model training** -- start w/ CatBoost (handles categoricals natively, has built-in noise robustness), then try a few others (logistic, HistGBT, etc.)
5. **ensemble** -- blend predictions from diverse models
6. **threshold tuning** -- since we care about accuracy, the best cutoff might not be 0.5

## repo structure

```
├── config.py               # paths + column constants
├── requirements.txt
├── data/
│   ├── README.md            # data schema and gotchas
│   └── covid-19-patient-diagnosis-based-on-symtoms/
│       ├── train.csv
│       ├── test.csv
│       └── sample_submission.csv
├── scripts/
│   ├── eda.py               # exploratory data analysis
│   ├── label_noise_check.py # confident learning noise detection
│   ├── baseline_cv.py       # CatBoost + LogReg baseline w/ CV
│   └── sanity_check.py      # check predictions against 15 known labels
├── reports/                 # output from scripts
└── figures/                 # plots
```

## setup

```bash
pip install -r requirements.txt
# on macOS, if you want xgboost/lightgbm later:
# brew install libomp
```

## team

Neel, Hussain, Cyrus, Kayin
