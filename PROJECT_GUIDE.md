# Project Guide: Employee Performance Predictor

Sections follow your A to G format. Code lives in `src/`; this guide explains it.

---
# A. Explanation

## What is Employee Performance Prediction?
**Simple:** like a weather forecast for careers. Using what we know about an employee (training, workload, attendance, feedback), the computer guesses whether performance will be Low, Medium or High.

**Technical:** a supervised multi-class classification problem. Features X = employee attributes, target y = performance tier. The model learns a mapping X -> y from labelled history and outputs class probabilities for new employees.

## Why companies need it
- **HR:** plan training, spot burnout and retention risk early.
- **Managers:** know who needs coaching or stretch work before review season.
- **Business leaders:** spend training and promotion budget where it pays back.

## How companies use it
| Use | How the prediction helps |
|---|---|
| Identify high performers | High tier plus high confidence goes to a talent pool |
| Predict low performance | Low tier triggers coaching, workload checks, support |
| Promotion decisions | Supports (never replaces) manager judgement |
| Training planning | Low `training_hours` plus Low tier means target programmes |
| Retention | Low satisfaction, overtime, stalled promotion are early flags |

## Workflow
`data -> preprocessing -> analysis -> model -> prediction -> insights -> HR decisions`

---
# B. Tech stack options

| | Option A (Easy) | Option B (Intermediate) | Option C (Advanced) |
|---|---|---|---|
| Tools | Python, Pandas, Matplotlib | + scikit-learn pipelines, Seaborn, Flask | + XGBoost, SHAP, Streamlit/FastAPI, Docker, MLflow |
| Models | Logistic Regression | LogReg, Random Forest, Gradient Boosting | + XGBoost, neural nets, hyperparameter search |
| Difficulty | Low | Medium | High |
| Student fit | Too basic for a portfolio | **Best balance** | Time-heavy, hard to explain |

**Selected: Option B.** It shows the full professional workflow (pipelines, model comparison, API) while every part stays explainable in an interview.

---
# C. Architecture

```
+----------------------+   +---------------------+   +------------------------+
| data_generator.py    |-->| preprocess.py       |-->| eda.py                 |
| synthetic HR data    |   | clean + new features|   | charts -> images/      |
+----------------------+   +---------------------+   +------------------------+
                                     |
                                     v
                          +---------------------+     +---------------------+
                          | train.py            |---->| models/*.joblib     |
                          | 3 models, 5-fold CV |     | outputs/metrics.json|
                          +---------------------+     +---------------------+
                                     |                          |
                          +---------------------+               v
                          | insights.py         |     +---------------------+
                          | HR analytics JSON   |---->| api.py (Flask)      |
                          +---------------------+     | + dashboard (HTML)  |
                                                      +---------------------+
                                                                |
                                                          HR user in browser
```

**Data flow:** raw CSV (with deliberate mess) -> clean CSV (+5 engineered columns) -> train/test split -> pipeline (scale numbers, one-hot categories, model) -> saved bundle -> API loads it once and answers `/api/predict`.

---
# D. Code, setup and phases

## Folder structure
```
Employee-Performance-Predictor/
├── data/raw, data/processed   datasets (generated)
├── notebooks/                 optional experiments
├── src/
│   ├── config.py              paths, constants, theme colours
│   ├── data_generator.py      synthetic data
│   ├── preprocess.py          cleaning + feature engineering
│   ├── eda.py                 charts
│   ├── train.py               models, evaluation, saving
│   ├── insights.py            HR analytics
│   ├── predict.py             one-employee prediction + advice
│   ├── pipeline.py            runs all stages
│   └── api.py                 Flask API
├── templates/index.html       themed dashboard
├── models/  outputs/  images/ generated artifacts
├── main.py                    pipeline only
├── runall.py                  pipeline + API + dashboard
├── requirements.txt  Procfile  README.md
```

## Installation
1. Install Python 3.10+ from python.org (Windows: tick "Add Python to PATH").
2. Check: `python --version` (Mac may need `python3`).
3. Create the environment: `python -m venv venv`
4. Activate: Windows `venv\Scripts\activate`, Mac/Linux `source venv/bin/activate`
5. Install: `pip install -r requirements.txt`

## Phases
| Phase | What | Why | Expected output | Common mistake |
|---|---|---|---|---|
| 1 Setup | Folders, venv, requirements | Reproducible project | Empty structure | Forgetting to activate venv |
| 2 Data | `data_generator.py` | No real HR data available | `employees_raw.csv` (1,515 rows) | Not setting a random seed |
| 3 Cleaning | `preprocess.clean` | Models fail on NaN and typos | Duplicates removed, gaps filled | Filling before the split on real data (leakage) |
| 4 EDA | `eda.py` | Understand patterns first | 5 charts | Skipping it and trusting the model blindly |
| 5 Features | `engineer_features` | Better signals (engagement, overtime) | 5 new columns | Using the target to build features |
| 6 Model | `train.py` | Compare 3 models fairly | CV table | Judging on training data |
| 7 Evaluation | Test set, confusion matrix | Honest performance | 78% accuracy | Reporting accuracy only |
| 8 Insights | `insights.py` | Turn numbers into HR actions | `insights.json` | Insights without a recommendation |
| 9 Visualization | Dashboard + charts | Communicate results | Web UI | Hard-to-read charts |
| 10 GitHub | Push with screenshots | Proof of work | Public repo | Committing `venv/` |

**Full code:** every file is in the project folder (config, generator, preprocessing, EDA, training, prediction, API, dashboard, `runall.py`), commented for beginners.

## Run
```bash
python runall.py
```
Expected: five numbered stages print, test accuracy about 0.78, then the browser opens the dashboard at `http://127.0.0.1:5000`.

---
# E. Virtual simulation

## How the company is simulated
1. **Employees generated:** 1,500 people with department, level, education, age, experience, salary tied to level and department.
2. **Behaviour generated:** training hours, projects, weekly hours, attendance, peer review, satisfaction, promotion gap, absences, drawn from realistic distributions.
3. **Performance calculated:** a hidden formula rewards training, projects, attendance, peer feedback, satisfaction and experience; it penalises absences, heavy overtime and long promotion gaps; random noise (±5) mimics real-life unpredictability.
4. **Tiers assigned:** bottom 25% Low, middle 50% Medium, top 25% High.
5. **Mess added:** missing values, duplicates, salary typos.
6. **Cleaning and training:** the model never sees the exact score, only the inputs.
7. **HR usage:** HR enters a profile in the dashboard (or calls the API) and gets a tier, probabilities and actions.

**Flow:** Generate -> Mess -> Clean -> Engineer -> EDA -> Train -> Evaluate -> Predict -> HR action.

## Screenshots to take (1 to 13 list merged)
| # | Capture | File/where |
|---|---|---|
| 1 | Dataset preview in terminal or Excel | `data/raw/employees_raw.csv` |
| 2 | Cleaning log (duplicates, missing values) | terminal |
| 3 | Performance distribution | `images/01_...png` |
| 4 | Department boxplot | `images/02_...png` |
| 5 | Training vs score | `images/03_...png` |
| 6 | Correlation heatmap | `images/04_...png` |
| 7 | Model comparison + accuracy | terminal output |
| 8 | Confusion matrix | `images/06_...png` |
| 9 | Feature importance | `images/07_...png` |
| 10 | Dashboard: Strong performer result | browser, save as `images/dashboard.png` |
| 11 | Dashboard: Needs support result | browser |
| 12 | `curl` or browser at `/api/health` | terminal |
| 13 | GitHub repo page with commits | browser |

---
# F. GitHub

## Steps
```bash
git init
git add .
git commit -m "Initial project structure and synthetic data generator"
git branch -M main
git remote add origin https://github.com/<your-username>/Employee-Performance-Predictor.git
git push -u origin main
```
1. On github.com click **New repository**, name it below, leave README unchecked, create.
2. Run the commands above inside your project folder.
3. Put screenshots in `images/`, commit and push.

- **Repo name:** `Employee-Performance-Predictor`
- **Description:** Predict employee performance tiers with ML, a Flask API and an HR analytics dashboard (synthetic data).
- **Topics:** `machine-learning` `data-analytics` `hr-analytics` `python` `scikit-learn` `flask` `classification` `data-science` `portfolio-project`

## 5-day proof plan
| Day | Do | Commit message |
|---|---|---|
| 1 | Setup, folders, requirements, config | `chore: set up project structure and environment` |
| 2 | Data generator + raw CSV + dataset screenshot | `feat: add synthetic HR dataset generator` |
| 3 | Cleaning, features, EDA charts | `feat: add preprocessing, feature engineering and EDA` |
| 4 | Model training, comparison, evaluation | `feat: train and evaluate classification models` |
| 5 | API, dashboard, runall.py, README, screenshots | `feat: add themed API dashboard and docs` |

---
# G. Interview preparation

## 10 questions
1. **What problem does it solve?** Early identification of high and low performers so HR can act on training, retention and promotion before review season.
2. **Why synthetic data?** Real HR data is private. I encoded realistic relationships, added noise and mess, so the method is real even though the numbers are not.
3. **Why classification, not regression?** HR acts on tiers. I also kept the raw score for analysis and can switch to regression.
4. **How did you pick the model?** Compared Logistic Regression, Random Forest and Gradient Boosting with 5-fold CV on the training set, then evaluated once on a held-out test set. Logistic Regression won because the underlying signal is mostly linear.
5. **Why not accuracy alone?** Confusion matrix, per-class precision/recall and macro F1 show whether Low and High are caught, not only the large Medium group.
6. **What is data leakage and did you avoid it?** Leakage is training on information unavailable at prediction time. I dropped `performance_score` because it directly defines the label, and split before evaluating.
7. **What feature engineering did you do?** Engagement index, overtime flag, stalled promotion flag, training per project, salary per experience year.
8. **How do you handle missing values and outliers?** Median for numeric, mode for categorical, IQR cap for salary.
9. **How do you explain a prediction?** Probabilities per tier, global drivers via permutation importance, and rule-based HR advice.
10. **What are the risks?** Bias and misuse. Never use protected attributes, keep a human in the loop, audit fairness, and treat predictions as decision support.

## Explaining to HR
"It's an early-warning tool. You enter an employee's profile, it tells you which performance tier they're likely in and why, and suggests actions like training or a workload review. It supports your judgement, it doesn't replace it."

## Explaining to a technical interviewer
"A scikit-learn Pipeline with ColumnTransformer (scaling plus one-hot) feeding a classifier chosen by stratified 5-fold CV. Evaluated on a stratified 80/20 holdout with confusion matrix and macro F1, explained with permutation importance, served through a Flask JSON API with input validation, and orchestrated by a single `runall.py`."

---
# Future improvements
- Real or public HR dataset (e.g. IBM HR Analytics on Kaggle) with the same pipeline
- Employee **attrition** prediction as a second model
- Deep learning (MLP) or XGBoost with SHAP explanations
- Streamlit or React dashboard, login and role-based access
- Real-time scoring with a database, scheduled retraining, Docker and CI

# Troubleshooting
| Error | Fix |
|---|---|
| `python` not recognised | Reinstall with "Add to PATH", or use `python3` |
| `ModuleNotFoundError` | Activate venv, then `pip install -r requirements.txt` |
| PowerShell blocks activate | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Port 5000 busy (Mac AirPlay) | `python runall.py --port 8000` |
| Dashboard says "Run the pipeline first" | `python runall.py` without `--skip-pipeline` |
| `FileNotFoundError` model | Run the pipeline once before the API |
| Charts not saved | Make sure `images/` exists and you run from the project root |
| `git push` rejected | `git pull origin main --allow-unrelated-histories`, then push |
| Different accuracy each run | Keep `RANDOM_STATE = 42` |
