"""Generate questionnaire-based figures for the Project 2 extension.

This script creates publication-friendly charts from questionairre.csv and
saves them to reports/project2_figures/.

The figures are intended for the hypothesis/testing section of the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "questionairre.csv"
OUTPUT_DIR = ROOT / "reports" / "project2_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


def normalize_yes_no_maybe(series: pd.Series) -> pd.Series:
    mapping = {
        "yes": "Yes",
        "yess": "Yes",
        "yesss": "Yes",
        "yesss like anger": "Yes",
        "yess like anger": "Yes",
        "no": "No",
        "nope": "No",
        "maybe": "Maybe",
        "may be": "Maybe",
        "na": "NA",
        "nil": "Nil",
    }
    return series.astype(str).str.strip().str.lower().replace(mapping)


def save_barplot(data: pd.Series, title: str, xlabel: str, ylabel: str, filename: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    order = data.index.tolist()
    sns.barplot(x=data.index, y=data.values, ax=ax, color=color)
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.bar_label(ax.containers[0], padding=3, fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_likert_hist(series: pd.Series, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(series, bins=[1, 2, 3, 4, 5, 6], discrete=True, ax=ax, color="#4C72B0")
    ax.set_title(title, pad=12)
    ax.set_xlabel("Likert score")
    ax.set_ylabel("Count")
    ax.set_xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_grouped_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, filename: str, palette: str = "Set2") -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x=x_col, y=y_col, hue=y_col, estimator=len, errorbar=None, ax=ax, palette=palette)
    ax.set_title(title, pad=12)
    ax.set_xlabel(x_col)
    ax.set_ylabel("Count")
    ax.legend_.remove()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Questionnaire file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Basic descriptive charts.
    activity = df["Physical Activity Level"].astype(str).str.strip().value_counts()
    save_barplot(
        activity,
        "Physical Activity Level Distribution",
        "Physical Activity Level",
        "Participants",
        "01_physical_activity_level.png",
        "#55A868",
    )

    speed = df["How would you describe your usual walking speed?"].astype(str).str.strip().value_counts()
    save_barplot(
        speed,
        "Usual Walking Speed Distribution",
        "Walking speed",
        "Participants",
        "02_walking_speed.png",
        "#C44E52",
    )

    org_col = "I am highly organized and disciplined."
    save_likert_hist(
        pd.to_numeric(df[org_col], errors="coerce"),
        "Organization and Discipline Score Distribution",
        "03_organization_score.png",
    )

    anxiety_col = "Do you have anxiety related problem ?"
    stage_col = "Do you have phobia of walking in stage?"
    injury_col = "Have you had any injuries in the past year that altered your gait?"

    anxiety = normalize_yes_no_maybe(df[anxiety_col]).value_counts().reindex(["Yes", "No", "Maybe"], fill_value=0)
    save_barplot(
        anxiety,
        "Anxiety-Related Problem Responses",
        "Response",
        "Participants",
        "04_anxiety.png",
        "#4C72B0",
    )

    stage = normalize_yes_no_maybe(df[stage_col]).value_counts().reindex(["Yes", "No", "Maybe"], fill_value=0)
    save_barplot(
        stage,
        "Stage Phobia Responses",
        "Response",
        "Participants",
        "05_stage_phobia.png",
        "#8172B2",
    )

    injury = normalize_yes_no_maybe(df[injury_col]).value_counts().reindex(["Yes", "No"], fill_value=0)
    save_barplot(
        injury,
        "Recent Gait-Altering Injury Responses",
        "Response",
        "Participants",
        "06_injury_history.png",
        "#CCB974",
    )

    # Hypothesis-supporting grouping plots.
    df["anxiety_norm"] = normalize_yes_no_maybe(df[anxiety_col]).replace({"Maybe": "Maybe"})
    df["org_group"] = pd.to_numeric(df[org_col], errors="coerce").apply(lambda v: ">=3" if v >= 3 else "<=2")

    anxiety_org = pd.crosstab(df["anxiety_norm"], df["org_group"])
    fig, ax = plt.subplots(figsize=(8, 5))
    anxiety_org.plot(kind="bar", stacked=False, ax=ax, colormap="tab10")
    ax.set_title("Anxiety vs Organization Group", pad=12)
    ax.set_xlabel("Anxiety response")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_anxiety_vs_organization.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # A compact visual for the free-text walk trait groups is useful for the report.
    trait_notes = {
        "Positive/Stable": [
            "confident",
            "balanced",
            "steady",
            "athletic",
            "purposeful",
            "calm",
            "relaxed",
            "smooth",
            "normal",
            "chilled",
            "gentle",
            "friendly",
            "composed",
            "brisk",
            "graceful",
            "bold",
            "powerful",
            "energetic",
        ],
        "Tense/Unstable": [
            "nervous",
            "anxious",
            "rushed",
            "jittery",
            "frantic",
            "guarded",
            "aggressive",
            "fear",
            "panic",
            "unease",
            "restless",
            "apprehension",
            "hesitant",
            "stress",
            "worry",
            "tensed",
            "shy",
            "uncertain",
        ],
    }

    trait_col = "Which personality trait is most visible in your walk?"
    traits = df[trait_col].astype(str).str.strip().str.lower()

    def categorize_trait(text: str) -> str:
        if any(keyword in text for keyword in trait_notes["Positive/Stable"]):
            return "Positive/Stable"
        if any(keyword in text for keyword in trait_notes["Tense/Unstable"]):
            return "Tense/Unstable"
        return "Other/Unclear"

    trait_group = traits.apply(categorize_trait).value_counts().reindex(["Positive/Stable", "Tense/Unstable", "Other/Unclear"], fill_value=0)
    save_barplot(
        trait_group,
        "Visible Trait in Walk (Grouped)",
        "Trait group",
        "Participants",
        "08_visible_trait_group.png",
        "#64B5CD",
    )

    captions = OUTPUT_DIR / "figure_captions.txt"
    captions.write_text(
        "Figure 5.3: Physical activity, walking speed, organization score, anxiety, stage phobia, and injury history distributions.\n"
        "Figure 5.4: Hypothesis-support plots including anxiety vs organization grouping and visible trait grouping.\n",
        encoding="utf-8",
    )

    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
