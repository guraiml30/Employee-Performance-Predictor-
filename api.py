"""Flask REST API + themed dashboard.

Endpoints
  GET  /                 dashboard (HTML)
  GET  /api/health       is the service alive + model loaded?
  GET  /api/options      allowed dropdown values and number ranges
  GET  /api/metrics      model accuracy, confusion matrix, drivers
  GET  /api/insights     HR analytics (departments, training, overtime)
  POST /api/predict      JSON employee in -> performance tier + advice out
"""
import json

from flask import Flask, jsonify, render_template, request, send_from_directory

from src import config as C
from src.predict import predict_employee

CATEGORY_FIELDS = {"department": C.DEPARTMENTS, "job_level": C.JOB_LEVELS, "education": C.EDUCATION}
NUMBER_FIELDS = {  # name: (min, max)
    "age": (18, 70), "years_experience": (0, 50), "monthly_salary": (1000, 1000000),
    "training_hours": (0, 200), "projects_completed": (0, 50), "weekly_hours": (10, 90),
    "attendance_pct": (0, 100), "peer_review_score": (1, 5), "satisfaction_score": (1, 5),
    "years_since_promotion": (0, 40), "absences": (0, 100),
}


def _read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def validate(payload):
    """Return (clean_dict, error_message)."""
    if not isinstance(payload, dict):
        return None, "Send a JSON object."
    clean = {}
    for name, allowed in CATEGORY_FIELDS.items():
        if payload.get(name) not in allowed:
            return None, f"'{name}' must be one of: {', '.join(allowed)}."
        clean[name] = payload[name]
    for name, (lo, hi) in NUMBER_FIELDS.items():
        try:
            value = float(payload.get(name))
        except (TypeError, ValueError):
            return None, f"'{name}' is required and must be a number."
        if not lo <= value <= hi:
            return None, f"'{name}' must be between {lo} and {hi}."
        clean[name] = value
    return clean, None


def create_app():
    app = Flask(__name__, template_folder=str(C.ROOT / "templates"))

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", model_ready=C.MODEL_PATH.exists())

    @app.get("/api/options")
    def options():
        return jsonify(categories=CATEGORY_FIELDS, numbers=NUMBER_FIELDS)

    @app.get("/api/metrics")
    def metrics():
        data = _read_json(C.METRICS_PATH)
        return (jsonify(data), 200) if data else (jsonify(error="Run the pipeline first."), 404)

    @app.get("/api/insights")
    def insights():
        data = _read_json(C.INSIGHTS_PATH)
        return (jsonify(data), 200) if data else (jsonify(error="Run the pipeline first."), 404)

    @app.post("/api/predict")
    def predict():
        clean, error = validate(request.get_json(silent=True))
        if error:
            return jsonify(error=error), 400
        return jsonify(predict_employee(clean))

    @app.get("/images/<path:name>")
    def images(name):
        return send_from_directory(C.IMAGES_DIR, name)

    return app


if __name__ == "__main__":
    create_app().run(port=5000)
