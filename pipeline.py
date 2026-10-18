"""Runs every stage in order and times each one."""
import time

from src import data_generator, eda, insights, preprocess, train

STEPS = [
    ("1/5  Generate synthetic HR data", data_generator.run),
    ("2/5  Clean data + engineer features", preprocess.run),
    ("3/5  Exploratory analysis (charts)", eda.run),
    ("4/5  Train, compare and evaluate models", train.run),
    ("5/5  Build HR insights", insights.run),
]


def run_pipeline():
    start = time.time()
    for title, fn in STEPS:
        print(f"\n== {title}")
        t0 = time.time()
        fn()
        print(f"   done in {time.time() - t0:.1f}s")
    print(f"\nPipeline finished in {time.time() - start:.1f}s")
