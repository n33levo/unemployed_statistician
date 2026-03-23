# STA314 Final Project -- COVID-19 Diagnosis Prediction

predicting whether a patient tested positive for covid from symptoms, demographics, and some clinical stuff. binary classification, ~1500 train rows, ~3500 test rows.

---

## update: final results

we finished **#2 on the private leaderboard** with a private score of **0.99130** (public: 0.98985). our best submission was S4 -- a 5-seed TabPFN ensemble trained on noise-corrected labels.

### scores by strategy

| strategy | description | private score |
|----------|-------------|---------------|
| S1 | TabPFN + T1 corrections only | 0.99015 |
| S2 | TabPFN + engineered features (worse) | 0.98261 |
| S3 | TabPFN + T1+T2 corrections | 0.99073 |
| S4 | **5-seed TabPFN ensemble + T1+T2** | **0.99130** |

### how to reproduce our best result

you need to run two things in order:

**step 1 — noise detection** (identifies mislabelled training samples):
```bash
python scripts/noise_ensemble_detect.py
```
this runs 3 independent methods (Cleanlab, AUM via CatBoost, Dataset Cartography), combines their votes into consensus tiers, and saves the corrected labels to `reports/tiered_noise_labels.json`. takes a few minutes.

**step 2 — S4 model** (trains the final ensemble and generates the submission):
```bash
python winning_strategy/scripts/s4_multiseed_ensemble.py
```
this trains TabPFN with 5 random seeds (42, 123, 314, 777, 2026) using the T1+T2 corrected labels from step 1, averages their probabilities, and writes the submission csv. the output goes to `winning_strategy/submissions/`.

### files involved

- [`config.py`](config.py) — paths and column name constants
- [`scripts/noise_ensemble_detect.py`](scripts/noise_ensemble_detect.py) — 3-method noise detection pipeline
- [`reports/tiered_noise_labels.json`](reports/tiered_noise_labels.json) — saved tier assignments + corrected labels (output of step 1)
- [`winning_strategy/scripts/s4_multiseed_ensemble.py`](winning_strategy/scripts/s4_multiseed_ensemble.py) — final model training + submission generation
- [`winning_strategy/submissions/s4_multiseed_t1t2_t50.csv`](winning_strategy/submissions/s4_multiseed_t1t2_t50.csv) — our best submission file

### notes

- TabPFN v2 needs a Hugging Face token to download the pretrained weights the first time. run `huggingface-cli login` or set `HF_TOKEN` before running step 2.
- the noise detection step is the key differentiator -- it flips 296 labels total (215 tier 1 + 81 tier 2) which is what pushed us from ~0.974 to 0.991.

---

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
│   ├── noise_ensemble_detect.py  # 3-method noise detection (run this first)
│   ├── label_noise_check.py # confident learning noise detection
│   ├── baseline_cv.py       # CatBoost + LogReg baseline w/ CV
│   └── sanity_check.py      # check predictions against 15 known labels
├── winning_strategy/
│   ├── scripts/
│   │   ├── s1_baseline_tabpfn.py      # T1 corrections only
│   │   ├── s3_tabpfn_tier2.py         # T1+T2 corrections
│   │   └── s4_multiseed_ensemble.py   # final 5-seed ensemble (run this second)
│   ├── submissions/
│   │   └── s4_multiseed_t1t2_t50.csv  # best submission
│   └── reports/
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
