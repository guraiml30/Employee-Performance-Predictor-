"""Step 1 - Build a synthetic (fake but realistic) HR dataset.

Performance score is a weighted mix of real-world drivers + random noise,
so the model has genuine patterns to learn. We also inject missing values
and duplicates on purpose so the cleaning step has real work to do.
"""
import numpy as np
import pandas as pd

from src import config as C


def generate_employees(n=C.N_EMPLOYEES, seed=C.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    department = rng.choice(C.DEPARTMENTS, n, p=[0.25, 0.18, 0.07, 0.10, 0.12, 0.16, 0.12])
    age = rng.integers(22, 60, n)
    experience = np.clip(age - 22 - rng.integers(0, 6, n), 0, None)
    education = rng.choice(C.EDUCATION, n, p=[0.10, 0.50, 0.32, 0.08])

    # Job level grows with experience
    job_level = np.select(
        [experience < 3, experience < 8, experience < 14],
        ["Junior", "Mid", "Senior"], default="Lead")

    level_pay = {"Junior": 30000, "Mid": 50000, "Senior": 75000, "Lead": 100000}
    dept_mult = {"Engineering": 1.2, "Sales": 1.0, "HR": 0.9, "Finance": 1.1,
                 "Marketing": 0.95, "Operations": 0.9, "Support": 0.8}
    salary = np.array([level_pay[l] * dept_mult[d] for l, d in zip(job_level, department)])
    salary = (salary * rng.normal(1.0, 0.08, n)).round(-2)

    training_hours = np.clip(rng.gamma(4, 8, n), 0, 120).round(1)
    projects = np.clip(rng.poisson(4 + experience * 0.15), 0, 20)
    weekly_hours = np.clip(rng.normal(44, 5, n), 30, 65).round(1)
    attendance = np.clip(rng.normal(94, 4, n), 70, 100).round(1)
    peer_review = np.clip(rng.normal(3.5, 0.6, n), 1, 5).round(2)
    satisfaction = np.clip(rng.normal(3.4, 0.8, n), 1, 5).round(2)
    since_promo = np.minimum(rng.integers(0, 8, n), experience)
    absences = np.clip(rng.poisson(5, n), 0, 30)

    edu_bonus = pd.Series(education).map({"Diploma": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}).values

    # --- the "hidden rule" that decides real performance ---
    score = (
        20
        + 0.15 * training_hours
        + 2.2 * projects
        + 0.25 * (attendance - 70)
        + 6.0 * peer_review
        + 3.0 * satisfaction
        + 0.6 * np.minimum(experience, 15)
        + 1.0 * edu_bonus
        - 0.5 * absences
        - 0.8 * np.maximum(weekly_hours - 50, 0)
        - 0.6 * np.maximum(since_promo - 4, 0)
        + rng.normal(0, 5, n)
    )
    score = np.clip(score, 0, 100).round(1)

    low_cut, high_cut = np.percentile(score, [25, 75])
    level = np.where(score < low_cut, "Low", np.where(score >= high_cut, "High", "Medium"))

    df = pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(1, n + 1)],
        "department": department, "job_level": job_level, "education": education,
        "age": age, "years_experience": experience, "monthly_salary": salary,
        "training_hours": training_hours, "projects_completed": projects,
        "weekly_hours": weekly_hours, "attendance_pct": attendance,
        "peer_review_score": peer_review, "satisfaction_score": satisfaction,
        "years_since_promotion": since_promo, "absences": absences,
        "performance_score": score, "performance_level": level,
    })

    # --- make the data messy on purpose (real HR data always is) ---
    for col, frac in [("training_hours", 0.04), ("satisfaction_score", 0.03), ("peer_review_score", 0.02)]:
        df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "monthly_salary"] *= 8   # data-entry outliers
    df = pd.concat([df, df.sample(15, random_state=seed)], ignore_index=True)  # duplicates
    return df


def run():
    df = generate_employees()
    C.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RAW_PATH, index=False)
    print(f"   saved {len(df)} raw rows -> {C.RAW_PATH.relative_to(C.ROOT)}")
    return df


if __name__ == "__main__":
    run()
