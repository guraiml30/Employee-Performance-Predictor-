"""Central settings. Change numbers here, not inside other files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT / "data" / "raw" / "employees_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "employees_clean.csv"
MODEL_PATH = ROOT / "models" / "performance_model.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
INSIGHTS_PATH = ROOT / "outputs" / "insights.json"
COMPARISON_PATH = ROOT / "outputs" / "model_comparison.csv"
IMAGES_DIR = ROOT / "images"

RANDOM_STATE = 42
N_EMPLOYEES = 1500

DEPARTMENTS = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations", "Support"]
EDUCATION = ["Diploma", "Bachelors", "Masters", "PhD"]
JOB_LEVELS = ["Junior", "Mid", "Senior", "Lead"]

TARGET = "performance_level"      # what we predict
LEAK_COLUMNS = ["performance_score"]  # exact score would give away the answer
ID_COLUMN = "employee_id"
LEVELS = ["Low", "Medium", "High"]

# Theme colours (used by charts AND the web dashboard so both match)
THEME = {
    "ink": "#11202E",
    "brand": "#1D3A5F",
    "mist": "#E9EEF2",
    "Low": "#CF4B3A",
    "Medium": "#D9A21B",
    "High": "#138A72",
}
