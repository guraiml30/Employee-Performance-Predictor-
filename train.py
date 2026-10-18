"""Steps 5-7 - Train several models, pick the best, evaluate it, save it."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C
from src.preprocess import CATEGORICAL, NUMERIC_BASE

ENGINEERED = ["salary_per_exp_year", "engagement_index", "overtime_flag",
              "promotion_stalled", "training_per_project"]
NUMERIC = NUMERIC_BASE + ENGINEERED
FEATURES = NUMERIC + CATEGORICAL


def build_pipeline(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def run():
    df = pd.read_csv(C.CLEAN_PATH)
    X, y = df[FEATURES], df[C.TARGET]   # performance_score is NOT used (it would leak the answer)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=C.RANDOM_STATE)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=C.RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=C.RANDOM_STATE),
    }

    rows = []
    for name, model in candidates.items():
        scores = cross_val_score(build_pipeline(model), X_train, y_train, cv=5, scoring="accuracy")
        rows.append({"model": name, "cv_accuracy": round(scores.mean(), 4), "cv_std": round(scores.std(), 4)})
        print(f"   {name:<20} CV accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")
    comparison = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False)
    comparison.to_csv(C.COMPARISON_PATH, index=False)

    best_name = comparison.iloc[0]["model"]
    print(f"   best model: {best_name}")
    pipe = build_pipeline(candidates[best_name]).fit(X_train, y_train)

    # ---- final exam on data the model has never seen ----
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    report = classification_report(y_test, pred, labels=C.LEVELS, output_dict=True)
    cm = confusion_matrix(y_test, pred, labels=C.LEVELS)
    print(f"   TEST accuracy = {acc:.3f} | macro F1 = {f1:.3f}")
    print(classification_report(y_test, pred, labels=C.LEVELS))

    # ---- confusion matrix picture ----
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(C.THEME["brand"], as_cmap=True),
                xticklabels=C.LEVELS, yticklabels=C.LEVELS, cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - {best_name}")
    plt.tight_layout(); plt.savefig(C.IMAGES_DIR / "06_confusion_matrix.png", dpi=140); plt.close()

    # ---- what matters most? (permutation importance, works for any model) ----
    perm = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=C.RANDOM_STATE)
    imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    top = imp.head(10)[::-1]
    plt.figure(figsize=(7, 4.5))
    plt.barh(top.index, top.values, color=C.THEME["brand"])
    plt.title("Top 10 drivers of performance"); plt.xlabel("Importance (drop in accuracy when shuffled)")
    plt.tight_layout(); plt.savefig(C.IMAGES_DIR / "07_feature_importance.png", dpi=140); plt.close()

    joblib.dump({"pipeline": pipe, "features": FEATURES, "levels": C.LEVELS}, C.MODEL_PATH)

    metrics = {
        "best_model": best_name, "test_accuracy": round(acc, 4), "macro_f1": round(f1, 4),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "confusion_matrix": cm.tolist(), "labels": C.LEVELS,
        "per_class": {k: {m: round(v, 3) for m, v in report[k].items()} for k in C.LEVELS},
        "feature_importance": {k: round(float(v), 4) for k, v in imp.head(10).items()},
        "model_comparison": comparison.to_dict(orient="records"),
    }
    C.METRICS_PATH.parent.mkdir(exist_ok=True)
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"   saved model -> {C.MODEL_PATH.relative_to(C.ROOT)}")
    return metrics


if __name__ == "__main__":
    run()
