"""Step 8 - Turn the data into numbers HR can act on."""
import json

import pandas as pd

from src import config as C


def run():
    df = pd.read_csv(C.CLEAN_PATH)
    tiers = df[C.TARGET].value_counts().reindex(C.LEVELS).to_dict()

    dept = (df.groupby("department")
              .agg(avg_score=("performance_score", "mean"),
                   headcount=("employee_id", "count"),
                   low_share=(C.TARGET, lambda s: (s == "Low").mean()))
              .round(3).sort_values("avg_score", ascending=False).reset_index())

    df["training_band"] = pd.cut(df["training_hours"], [-1, 20, 40, 60, 200],
                                 labels=["0-20h", "21-40h", "41-60h", "60h+"])
    training = (df.groupby("training_band", observed=True)["performance_score"].mean().round(1)
                  .rename_axis("band").reset_index().to_dict(orient="records"))

    overtime = (df.assign(overtime=df["weekly_hours"] > 50)
                  .groupby("overtime")["performance_score"].mean().round(1).to_dict())

    insights = {
        "headcount": len(df),
        "tiers": tiers,
        "avg_score": round(df["performance_score"].mean(), 1),
        "departments": dept.to_dict(orient="records"),
        "training_vs_score": training,
        "overtime_effect": {"over_50h": overtime.get(True), "up_to_50h": overtime.get(False)},
        "stalled_promotion_low_share": round(
            float((df[df["promotion_stalled"] == 1][C.TARGET] == "Low").mean()), 3),
    }
    C.INSIGHTS_PATH.write_text(json.dumps(insights, indent=2))
    print(f"   best department: {dept.iloc[0]['department']} | weakest: {dept.iloc[-1]['department']}")
    print(f"   saved insights -> {C.INSIGHTS_PATH.relative_to(C.ROOT)}")
    return insights


if __name__ == "__main__":
    run()
