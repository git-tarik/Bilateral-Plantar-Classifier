"""Questionnaire analysis for Bilateral-Plantar-Classifier.

This script reads `questionairre.csv`, cleans key fields, computes
summary statistics related to the psychological hypotheses, and writes
human-readable results to `reports/questionnaire_analysis.txt`.

It does **not** touch any human-subject raw gait data. It only uses the
questionnaire already stored in the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "questionairre.csv"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = REPORTS_DIR / "questionnaire_analysis.txt"


def norm_yesno(series: pd.Series) -> pd.Series:
    """Normalize yes/no/maybe style answers.

    Returns lower-cased strings like "yes", "no", "maybe", "na", "nil".
    """

    mapping = {
        "yes": "yes",
        "yess": "yes",
        "yesss like anger": "yes",
        "yesss": "yes",
        "yess like anger": "yes",
        "nope": "no",
        "no": "no",
        "maybe": "maybe",
        "may be": "maybe",
        "na": "na",
        "nil": "nil",
    }
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace(mapping)
    )


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find questionnaire CSV at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df


def compute_summary(df: pd.DataFrame) -> str:
    lines: list[str] = []

    lines.append("QUESTIONNAIRE OVERVIEW")
    lines.append("----------------------")
    lines.append(f"Total participants: {len(df)}")
    lines.append("")

    # Basic demographics / activity level / walking speed
    lines.append("Physical Activity Level (counts):")
    lines.append(df["Physical Activity Level"].value_counts().to_string())
    lines.append("")

    lines.append("Usual walking speed (counts):")
    speed_col = "How would you describe your usual walking speed?"
    lines.append(df[speed_col].value_counts().to_string())
    lines.append("")

    # Key Likert item: organisation / discipline
    org_col = "I am highly organized and disciplined."
    org = df[org_col]
    lines.append("Organisation / discipline scores (1–5):")
    lines.append(f"  Min:  {org.min()}")
    lines.append(f"  Max:  {org.max()}")
    lines.append(f"  Mean: {org.mean():.2f}")
    lines.append(f"  >= 3: {(org >= 3).sum()} participants")
    lines.append(f"  <= 2: {(org <= 2).sum()} participants")
    lines.append("")

    # Binary-style questions: anxiety, stage phobia, injury
    anxiety_col = "Do you have anxiety related problem ?"
    stage_col = "Do you have phobia of walking in stage?"
    injury_col = "Have you had any injuries in the past year that altered your gait?"

    for raw_col, label in [
        (anxiety_col, "Anxiety"),
        (stage_col, "Stage phobia"),
        (injury_col, "Recent gait-altering injury"),
    ]:
        if raw_col not in df.columns:
            continue
        norm_col = f"{raw_col} (normalized)"
        df[norm_col] = norm_yesno(df[raw_col])
        counts = df[norm_col].value_counts(dropna=False)
        lines.append(f"{label} responses (normalized):")
        lines.append(counts.to_string())
        lines.append("")

    # Simple cross-tab examples *within* questionnaire (no gait labels yet)
    # Example: anxiety vs organisation score category
    if anxiety_col in df.columns:
        df["anxiety_norm"] = norm_yesno(df[anxiety_col])
        df["org_high"] = (df[org_col] >= 3).map({True: ">=3", False: "<=2"})
        ct = pd.crosstab(df["anxiety_norm"], df["org_high"])
        lines.append("Anxiety (yes/no/maybe) vs organisation category (>=3 vs <=2):")
        lines.append(ct.to_string())
        lines.append("")

    lines.append(
        "NOTE: At present this analysis only uses the questionnaire. "
        "To formally test the hypotheses about Normal vs Abnormal gait, "
        "a future step would merge these participants with the gait labels "
        "produced by the existing Random Forest classifier."
    )

    return "\n".join(lines)


def main() -> None:
    df = load_data()
    report_text = compute_summary(df)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Wrote questionnaire analysis to: {OUT_PATH}")


if __name__ == "__main__":  # pragma: no cover
    main()
