"""Steps 2-3 - Clean the data and create new helpful columns (feature engineering)."""
import pandas as pd

from src import config as C

NUMERIC_BASE = ["age", "years_experience", "monthly_salary", "training_hours", "projects_completed",
                "weekly_hours", "attendance_pct", "peer_review_score", "satisfaction_score",
                "years_since_promotion", "absences"]
CATEGORICAL = ["department", "job_level", "education"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset=C.ID_COLUMN).reset_index(drop=True)
    print(f"   removed {before - len(df)} duplicate rows")

    # Fill missing values: median for numbers, most common value for text
    for col in NUMERIC_BASE:
        if df[col].isna().any():
            print(f"   filled {df[col].isna().sum()} missing in {col} (median)")
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Cap extreme salary typos using the IQR rule
    q1, q3 = df["monthly_salary"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    capped = (df["monthly_salary"] > upper).sum()
    df["monthly_salary"] = df["monthly_salary"].clip(upper=upper)
    print(f"   capped {capped} salary outliers at {upper:,.0f}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new columns from existing ones. Used in training AND prediction."""
    df = df.copy()
    df["salary_per_exp_year"] = df["monthly_salary"] / (df["years_experience"] + 1)
    df["engagement_index"] = (df["peer_review_score"] + df["satisfaction_score"]) / 2
    df["overtime_flag"] = (df["weekly_hours"] > 50).astype(int)
    df["promotion_stalled"] = (df["years_since_promotion"] >= 4).astype(int)
    df["training_per_project"] = df["training_hours"] / (df["projects_completed"] + 1)
    return df


def run():
    raw = pd.read_csv(C.RAW_PATH)
    df = engineer_features(clean(raw))
    C.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEAN_PATH, index=False)
    print(f"   saved {len(df)} clean rows, {df.shape[1]} columns -> {C.CLEAN_PATH.relative_to(C.ROOT)}")
    return df


if __name__ == "__main__":
    run()
