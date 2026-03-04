# what we should do next

just putting my thoughts here after going thru the data and running some initial tests. we can discuss at the meeting but this is basically where i think we should focus.

## where we're at rn

i ran a basic EDA, a baseline model (CatBoost + logistic), and a label noise check. results are in `reports/`. tl;dr:

- baseline CatBoost gives ~73% accuracy, logistic gives ~74%
- the label noise check flagged ~283 samples (almost 19% of train) as likely mislabeled
- oxygen_level is by far the most important feature, then age and body_temp
- dry_cough, shortness_of_breath, and loss_of_smell have the strongest signal among symptoms
- comorbidity is missing for 53% of rows but we can just treat that as its own category
- class balance is pretty much 50/50 so thats not an issue

## the big thing: label noise

the project description literally says he flipped some labels on purpose. so i think the biggest win is going to come from dealing w/ that properly. most other teams prob wont bother.

what i think we should try:
1. **remove the worst offenders** -- drop the top ~5-10% most suspicious samples and retrain, see if accuracy goes up
2. **reweight instead of removing** -- give lower weight to suspicious samples instead of throwing them out. less aggressive
3. **soft labels** -- instead of hard 0/1, use the model's own predicted probability as the "true" label (like label smoothing)

we should test all 3 via cross-val and pick whatever works best. i already saved the list of suspicious indices in `reports/suspected_label_issues.json` so we can start from there.

## feature engineering ideas

some stuff we could add on top of the raw features:
- `symptom_count` -- just sum up all the binary symptom columns. overall sickness measure
- `low_oxygen` -- binary flag for oxygen < 95 (clinically thats hypoxemia)
- `fever_temp` -- body temp > 38 (actual clinical fever cutoff)
- `exposure_score` -- travel_history + contact_with_patient combined
- `respiratory_distress` -- shortness_of_breath AND low_oxygen together
- `smell_taste_loss` -- loss_of_smell + loss_of_taste (loss of smell/taste is super specific to covid)

tree models can learn interactions on their own but sometimes spelling them out explicitly helps, esp w/ noisy data.

## models to try

i think we should go w/ a mix of 3-4 different models rather than just one. something like:

1. **CatBoost** -- handles categoricals natively, has built-in regularization, should be our primary model. tune w/ optuna
2. **HistGradientBoosting** (sklearn) -- no extra deps, fast, good baseline
3. **ExtraTrees** -- random splits make it naturally robust to noise
4. **LogReg** -- different inductive bias, good for the blend

if we can get xgboost/lightgbm working (need `brew install libomp` on mac) those are worth adding too. theres also this thing called TabPFN which is a pretrained transformer for tabular data -- zero tuning needed and apparently really good on small datasets. worth a shot.

then we blend their predictions. either a weighted average of the probabilities or stacking (train another model on top of all their outputs).

## threshold tuning

since the metric is accuracy, the optimal threshold for converting probabilities to 0/1 might not be 0.5. we should sweep thresholds on the OOF predictions and use whatever maximizes accuracy. from the baseline run it looks like it doesn't matter much for catboost alone but it might matter more for the ensemble.

## sanity check

theres 15 test rows where we actually know the answer (from the course sample data). i built a script that checks our predictions against those. its not a lot but its something -- if we're getting less than ~80% on those 15 we should investigate.

## who does what (suggestion)

- label noise handling + catboost tuning
- feature engineering + other models (histgbt, extratrees, etc)
- ensemble + threshold tuning + submission pipeline
- EDA writeup + report draft

we can figure out the split at the meeting. lmk what you guys think
