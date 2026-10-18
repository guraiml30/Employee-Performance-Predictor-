"""Step 4 - Exploratory Data Analysis: make charts that tell the story."""
import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config as C

T = C.THEME
PALETTE = {lvl: T[lvl] for lvl in C.LEVELS}


def _style():
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": T["ink"], "text.color": T["ink"],
                                         "axes.labelcolor": T["ink"], "xtick.color": T["ink"],
                                         "ytick.color": T["ink"], "figure.facecolor": "white"})


def _save(name):
    C.IMAGES_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(C.IMAGES_DIR / name, dpi=140)
    plt.close()
    print(f"   saved images/{name}")


def run():
    _style()
    df = pd.read_csv(C.CLEAN_PATH)
    print("   dataset preview:\n", df.head(3).to_string(max_cols=8))
    order = C.LEVELS

    # 1. How many employees in each tier?
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=C.TARGET, order=order, hue=C.TARGET, palette=PALETTE, legend=False)
    for container in ax.containers:
        ax.bar_label(container)
    plt.title("Employees by performance level")
    _save("01_performance_distribution.png")

    # 2. Score by department
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(data=df, x="department", y="performance_score", color=T["brand"])
    plt.xticks(rotation=25)
    plt.title("Performance score by department")
    _save("02_score_by_department.png")

    # 3. Training vs score
    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(data=df, x="training_hours", y="performance_score", hue=C.TARGET,
                    hue_order=order, palette=PALETTE, alpha=0.7)
    plt.title("More training, higher scores?")
    _save("03_training_vs_score.png")

    # 4. Correlation heatmap
    plt.figure(figsize=(9, 7))
    corr = df.select_dtypes("number").corr()
    sns.heatmap(corr, cmap=sns.diverging_palette(15, 175, as_cmap=True), center=0, linewidths=0.3)
    plt.title("Correlation between numeric columns")
    _save("04_correlation_heatmap.png")

    # 5. Overtime vs tier
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=df, x=C.TARGET, y="weekly_hours", order=order, hue=C.TARGET,
                palette=PALETTE, legend=False)
    plt.title("Weekly hours by performance level")
    _save("05_weekly_hours_by_level.png")


if __name__ == "__main__":
    run()
