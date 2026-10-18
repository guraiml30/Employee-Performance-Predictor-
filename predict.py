"""Use the trained model on ONE employee and explain what HR should do."""
import joblib
import pandas as pd

from src import config as C
from src.preprocess import engineer_features

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not C.MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run `python runall.py` first.")
        _bundle = joblib.load(C.MODEL_PATH)
    return _bundle


def recommendations(e: dict) -> list[str]:
    """Simple, explainable HR rules - easy to defend in an interview."""
    tips = []
    if e["training_hours"] < 25:
        tips.append("Enrol in a skills programme: training hours are well below the team norm.")
    if e["weekly_hours"] > 50:
        tips.append("Review workload: sustained overtime is linked with lower performance and burnout.")
    if e["years_since_promotion"] >= 4:
        tips.append("Hold a career-growth conversation: a long gap since the last promotion.")
    if e["satisfaction_score"] < 3:
        tips.append("Schedule a one-to-one check-in: satisfaction is low, so retention risk is higher.")
    if e["attendance_pct"] < 90 or e["absences"] > 10:
        tips.append("Talk about attendance early and offer support before it affects results.")
    if e["peer_review_score"] < 3:
        tips.append("Arrange mentoring or team-collaboration coaching to lift peer feedback.")
    if not tips:
        tips.append("Keep momentum: consider stretch projects, mentoring roles or a promotion review.")
    return tips


def predict_employee(employee: dict) -> dict:
    bundle = _load()
    row = engineer_features(pd.DataFrame([employee]))[bundle["features"]]
    pipe = bundle["pipeline"]
    probs = pipe.predict_proba(row)[0]
    prob_map = {cls: float(p) for cls, p in zip(pipe.classes_, probs)}
    label = max(prob_map, key=prob_map.get)
    return {
        "prediction": label,
        "confidence": round(prob_map[label], 3),
        "probabilities": {k: round(prob_map.get(k, 0.0), 3) for k in C.LEVELS},
        "recommendations": recommendations(employee),
    }


if __name__ == "__main__":
    demo = dict(department="Engineering", job_level="Mid", education="Masters", age=30,
                years_experience=6, monthly_salary=62000, training_hours=55, projects_completed=7,
                weekly_hours=44, attendance_pct=96, peer_review_score=4.2, satisfaction_score=4.0,
                years_since_promotion=1, absences=3)
    print(predict_employee(demo))
