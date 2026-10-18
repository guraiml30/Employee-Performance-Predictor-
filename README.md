# Employee Performance Predictor using Data Analytics

Predict whether an employee will be a **Low, Medium or High** performer, and give HR concrete next steps. Built end to end on a **synthetic HR dataset**: data generation, cleaning, EDA, model comparison, evaluation, a REST API and a themed web dashboard.

![image alt](https://github.com/guraiml30/Employee-Performance-Predictor-/blob/e98e05a38d5bb7372abea35b8e4f5c64bc1a5bfc/Git1.png)

## Problem statement
Managers spot strong and struggling employees late, usually at the annual review. This project uses HR data (training, workload, attendance, peer feedback, promotion history) to flag performance tiers early so HR can plan training, retention and promotions.

## Business value
| HR question | How the project helps |
|---|---|
| Who are our high performers? | High-tier predictions feed promotion and succession lists |
| Who needs support? | Low-tier predictions trigger coaching and training |
| Where should training budget go? | Insights show score gains per training band |
| Who is at risk of burnout? | Overtime and stalled-promotion signals are flagged |

## Tech stack
Python, Pandas, NumPy, scikit-learn (Logistic Regression, Random Forest, Gradient Boosting), Matplotlib, Seaborn, Flask, HTML/CSS/JS.

## Architecture
```
Synthetic HR data -> Cleaning -> Feature engineering -> EDA charts
                                         |
                              Train 3 models (5-fold CV)
                                         |
                       Best model + metrics + insights (saved)
                                         |
                    Flask API  <->  Themed HR dashboard (browser)
```

## How to run
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python runall.py                # pipeline + API + dashboard at http://127.0.0.1:5000
```
Other options: `python runall.py --no-server`, `--skip-pipeline`, `--port 8000`, or `python main.py` (pipeline only).

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/api/health` | Service and model status |
| GET | `/api/metrics` | Accuracy, confusion matrix, top drivers |
| GET | `/api/insights` | Department, training and overtime analytics |
| POST | `/api/predict` | Employee JSON in, tier + probabilities + HR advice out |

```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d '{"department":"Engineering","job_level":"Senior","education":"Masters","age":34,"years_experience":10,"monthly_salary":90000,"training_hours":70,"projects_completed":9,"weekly_hours":44,"attendance_pct":98,"peer_review_score":4.6,"satisfaction_score":4.3,"years_since_promotion":1,"absences":2}'
```

## Results (1,500 synthetic employees, 80/20 split)
| Model | 5-fold CV accuracy |
|---|---|
| Logistic Regression (selected) | 0.740 |
| Gradient Boosting | 0.703 |
| Random Forest | 0.698 |

Held-out test accuracy: **78.3%**, macro F1: **0.78**.

| EDA | Evaluation |
|---|---|
| ![](images/03_training_vs_score.png) | ![](images/06_confusion_matrix.png) |
| ![](images/02_score_by_department.png) | ![](images/07_feature_importance.png) |

## Honest limitations
Data is synthetic, so results show the method works, not how a real company behaves. Real deployments need fairness audits (no gender, age or ethnicity-based decisions) and human review of every prediction.

## Deploy
`pip install -r requirements-deploy.txt`, build command `python runall.py --no-server`, start command in `Procfile` (works on Render, Railway, Heroku).

## Folder structure
`data/` datasets, `src/` all code, `models/` saved model, `outputs/` metrics and insights, `images/` charts, `templates/` dashboard, `docs/` full guide.
