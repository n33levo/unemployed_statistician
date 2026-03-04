# Data Notes

## files

| file | rows | columns | notes |
|------|------|---------|-------|
| `train.csv` | 1500 | 18 | has `covid_result` target |
| `test.csv` | 3500 | 17 | no target, this is what we predict |
| `sample_submission.csv` | 3500 | 2 | format: `id`, `covid_result` (must be "0" or "1") |

## schema

All files use `id` as the identifier column (**not** `patient_id` -- the course sample file uses `patient_id` but the actual competition data uses `id`).

| column | type | values / range |
|--------|------|----------------|
| id | int | unique row identifier |
| age | int | 1-90ish |
| gender | str | Male, Female |
| fever | int | 0/1 |
| dry_cough | int | 0/1 |
| sore_throat | int | 0/1 |
| fatigue | int | 0/1 |
| headache | int | 0/1 |
| shortness_of_breath | int | 0/1 |
| loss_of_smell | int | 0/1 |
| loss_of_taste | int | 0/1 |
| oxygen_level | float | ~80-100 (SpO2 %) |
| body_temperature | float | ~35-42 (Celsius) |
| comorbidity | str | Diabetes, Asthma, Heart Disease, None -- **53% missing in train** |
| travel_history | int | 0/1 |
| contact_with_patient | int | 0/1 |
| chest_pain | int | 0/1 |
| covid_result | str | "0" / "1" (target, only in train) |

## things to watch out for

1. **train numerics are weirdly precise**: oxygen_level in train looks like `98.5433231409651`, but in test its just `98`. tree models handle this fine but be aware if doing anything w/ distances or linear models

2. **comorbidity is 53% missing in train**: don't drop these rows. fill w/ "Unknown" and treat as a 5th category

3. **the column is `id`, not `patient_id`**: the sample csv from the course materials uses `patient_id` but the actual competition csvs use `id`. this has caused bugs before

4. **15 known test labels**: the course sample file has `patient_id` 111-125 which map to test `id` 1-15 (same features, just offset by 110). we can use these to sanity check any submission

5. **label noise is intentional**: the project description says he flipped some labels on purpose. so ~5-15% of training labels might be wrong
