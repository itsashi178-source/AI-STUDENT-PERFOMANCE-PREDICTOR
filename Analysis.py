"""Marks analysis using Pandas and NumPy."""
import numpy as np
import pandas as pd

WEAK_BELOW = 50    # a subject below 50% is "weak"
STRONG_FROM = 75   # a subject at or above 75% is "strong"


def build_dataframe(rows) -> pd.DataFrame:
    """Validate the subject table and add a Percentage column."""
    df = pd.DataFrame(rows)
    for col in ("Subject", "Obtained", "Total"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df[["Subject", "Obtained", "Total"]].copy()
    df["Subject"] = df["Subject"].fillna("").astype(str).str.strip()
    df["Obtained"] = pd.to_numeric(df["Obtained"], errors="coerce")
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
    # drop rows that are completely blank
    blank = (df["Subject"] == "") & df["Obtained"].isna() & df["Total"].isna()
    df = df[~blank].reset_index(drop=True)
    if df.empty:
        raise ValueError("Please add at least one subject with marks.")
    for i, r in df.iterrows():
        label = r["Subject"] or f"row {i + 1}"
        if r["Subject"] == "":
            raise ValueError(f"Row {i + 1} has no subject name.")
        if pd.isna(r["Obtained"]) or pd.isna(r["Total"]):
            raise ValueError(f"'{label}': please enter both obtained and total marks.")
        if r["Total"] <= 0:
            raise ValueError(f"'{label}': total marks must be greater than 0.")
        if r["Obtained"] < 0 or r["Obtained"] > r["Total"]:
            raise ValueError(f"'{label}': obtained marks must be between 0 and total marks.")
    df["Percentage"] = (df["Obtained"] / df["Total"] * 100).round(1)
    return df


def classify(pct: float) -> dict:
    """Map a percentage to a performance category."""
    if pct >= 85:
        return {"label": "Outstanding", "emoji": "🌟", "color": "#16a34a"}
    if pct >= 70:
        return {"label": "Very Good", "emoji": "🚀", "color": "#7c3aed"}
    if pct >= 55:
        return {"label": "Good", "emoji": "👍", "color": "#3b82f6"}
    if pct >= 40:
        return {"label": "Needs Improvement", "emoji": "⚠️", "color": "#f59e0b"}
    return {"label": "At Risk", "emoji": "🚨", "color": "#ef4444"}


def summarize(df: pd.DataFrame) -> dict:
    pct = df["Percentage"].to_numpy(dtype=float)
    avg = float(np.mean(pct))
    weak_mask = (pct < WEAK_BELOW) | (pct < avg - 10)
    strong_mask = ((pct >= STRONG_FROM) | (pct >= avg + 10)) & ~weak_mask
    order = np.argsort(pct)
    subjects = df["Subject"].to_numpy()
    weak_set = set(subjects[weak_mask])
    strong_set = set(subjects[strong_mask])
    return {
        "n_subjects": int(len(df)),
        "average_pct": avg,
        "overall_pct": float(df["Obtained"].sum() / df["Total"].sum() * 100),
        "avg_obtained": float(df["Obtained"].mean()),
        "total_obtained": float(df["Obtained"].sum()),
        "total_marks": float(df["Total"].sum()),
        "std": float(np.std(pct)),
        "best_subject": str(subjects[order[-1]]),
        "worst_subject": str(subjects[order[0]]),
        "weak": [str(s) for s in subjects[order] if s in weak_set],
        "strong": [str(s) for s in subjects[order[::-1]] if s in strong_set],
        "category": classify(avg),
    }


def build_recommendations(summary: dict, student: dict) -> list:
    """Simple rule-based advice (works without any API key)."""
    recs = []
    if summary["weak"]:
        recs.append("Give extra time to your weak subjects: " + ", ".join(summary["weak"])
                    + ". Start each study session with the hardest one while your energy is high.")
    if student["attendance"] < 75:
        recs.append("Your attendance is below 75%. Many institutions require a minimum attendance, "
                    "so try to attend every class from now on.")
    elif student["attendance"] < 90:
        recs.append("Your attendance is fair. Attending more classes usually helps with understanding topics early.")
    if student["study_hours"] < 2:
        recs.append("You study less than 2 hours a day. Try to build up to 3-4 focused hours with short breaks.")
    elif student["study_hours"] > 7:
        recs.append("You study many hours a day. Make sure you sleep well and take breaks so your study stays effective.")
    if summary["std"] > 15:
        recs.append("Your marks vary a lot between subjects. Balance your time so strong subjects stay strong and weak ones catch up.")
    recs.append("Practice past papers and test yourself instead of only re-reading notes (active recall).")
    recs.append("Review what you learned the same day, then again after 3 and 7 days (spaced repetition).")
    return recs
