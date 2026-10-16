# CYP Inhibition Leaderboard

Leaderboard for the DTU Deep Learning project **Predicting CYP Inhibition with Molecular Foundation Models**.

**Live leaderboard: https://khelverskovp.github.io/dtu-cyp-leaderboard/** · [Starter code](https://github.com/khelverskovp/dtu-cyp-project)

There are two leaderboards. Each shows the best submission per team.

| Task | Endpoints | Ranked by |
|---|---|---|
| Regression | pIC50 for CYP1A2, CYP2C9, CYP2D6, CYP3A4 | Macro ST-RAE (lower is better) |
| Classification | TDI for CYP2D6, CYP3A4 | Macro MCC at threshold 0.5 (higher is better) |

## How to submit

1. Train your model with the starter code and choose your final model on the **validation** set.
2. Take the `test_predictions.csv` that `finetune.py` wrote for that run (in `runs/<task>/<model>/<timestamp>/`).
3. [Open a submission](https://github.com/khelverskovp/dtu-cyp-leaderboard/issues/new?template=submission.yml): fill in your team name, the task and a short model description, and drag the CSV into the form.
4. Within a few minutes a bot comments your score on the issue and updates the leaderboard.

The file must contain `Molecule_Name` and one column per endpoint, with a prediction for each of the 921 test molecules from `prepare_data.py` (default seed 42). Classification predictions are probabilities between 0 and 1. If you build your own pipeline, write the same format.

## Rules

- **Tune on validation, not test.** Choose models, hyperparameters and epochs using the validation split only. The test split is for the final evaluation. The leaderboard shows test scores, so do not use it for model selection.
- **One submission per team per task per day** (Copenhagen time). Rejected submissions do not count.
- **Use the starter repo's splits.** Do not train on the test molecules or their labels. The test labels are in your `data/test.csv`, and the challenge data is public, so the leaderboard relies on your honesty.
- **Use one team name** for all your team's submissions.

Breaking these rules makes your results meaningless for your report, and the course staff may remove such entries.

## How it works

Submissions are GitHub issues. A GitHub Actions workflow (`.github/workflows/submission.yml`) checks the file, scores it with the same metrics as `finetune.py` (`scoring/score.py`), records it in `submissions.json` and redeploys the page in `docs/`. The test labels are rebuilt at scoring time from the public challenge data; `scoring/test_molecules.csv` lists the test molecules.
