"""Run only the data + model pipeline (no web server)."""
from src.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
