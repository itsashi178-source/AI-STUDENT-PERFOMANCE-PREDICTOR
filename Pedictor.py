"""Machine-learning model that estimates next-semester performance.

IMPORTANT: there is no real student dataset in this project. The model is a Ridge
regression trained on SYNTHETIC data generated from a simple, documented formula
(see `_target`). It is a demo of the ML pipeline, so results are only rough estimates.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.analysis import classify


def _target(cur, prev, att, hrs):
    """Assumed relationship used to create the synthetic training data."""
    return 0.55 * cur + 0.20 * prev + 0.12 * att + 1.5 * np.minimum(hrs, 6) + 6.0


def make_training_data(n=800, seed=42):
    rng = np.random.default_rng(seed)
    cur = np.clip(rng.normal(65, 15, n), 20, 98)
    prev = np.clip(cur + rng.normal(0, 8, n), 20, 100)
    att = np.clip(rng.normal(80, 12, n), 40, 100)
    hrs = np.clip(rng.normal(3, 1.5, n), 0, 10)
    y = np.clip(_target(cur, prev, att, hrs) + rng.normal(0, 3, n), 0, 100)
    return np.column_stack([cur, prev, att, hrs]), y


def train_model(seed=42) -> dict:
    X, y = make_training_data(seed=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
    r2 = float(1 - np.sum((y_te - pred) ** 2) / np.sum((y_te - np.mean(y_te)) ** 2))
    return {"model": model, "rmse": rmse, "r2": r2, "n_train": int(len(X_tr))}


def predict(bundle, current_avg, prev_pct, attendance, study_hours) -> dict:
    """prev_pct may be None (then the current average is used instead)."""
    prev = current_avg if prev_pct is None else prev_pct
    x = np.array([[current_avg, prev, attendance, study_hours]], dtype=float)
    value = float(np.clip(bundle["model"].predict(x)[0], 0, 100))
    margin = 1.96 * bundle["rmse"]
    return {
        "predicted": value,
        "low": max(0.0, value - margin),
        "high": min(100.0, value + margin),
        "delta": value - current_avg,
        "category": classify(value),
    }


def study_hours_curve(bundle, current_avg, prev_pct, attendance):
    hours = np.linspace(0, 8, 17)
    prev = current_avg if prev_pct is None else prev_pct
    X = np.column_stack([np.full_like(hours, current_avg), np.full_like(hours, prev),
                         np.full_like(hours, attendance), hours])
    return hours, np.clip(bundle["model"].predict(X), 0, 100)


def explain(student, summary, pred) -> list:
    """Explain the prediction in simple English."""
    pts = []
    d = pred["delta"]
    if d > 2:
        pts.append(f"Your estimated next score is about {abs(d):.1f} points higher than your current average.")
    elif d < -2:
        pts.append(f"Your estimated next score is about {abs(d):.1f} points lower than your current average.")
    else:
        pts.append("Your estimated next score is close to your current average, so you are on a steady path.")
    if student["attendance"] >= 85:
        pts.append("Your good attendance helps your estimate.")
    elif student["attendance"] < 75:
        pts.append("Low attendance pulls your estimate down. Improving it is one of the easiest wins.")
    if student["study_hours"] >= 3:
        pts.append("Your daily study time supports your estimate.")
    else:
        pts.append("More daily study time (up to about 4-6 focused hours) would raise your estimate.")
    if summary["weak"]:
        pts.append("Weak subjects to focus on: " + ", ".join(summary["weak"]) + ".")
    return pts
