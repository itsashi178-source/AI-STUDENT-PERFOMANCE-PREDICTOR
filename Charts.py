"""Matplotlib charts. Every function returns a Figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PURPLE, BLUE, PINK, TEAL, AMBER, RED


def _band_color(p):
    return TEAL if p >= 75 else (BLUE if p >= 55 else (AMBER if p >= 40 else RED))


def _short(name, n=14):
    return name if len(name) <= n else name[: n - 1] + "…"


def _new(title, xlabel="", ylabel="", size=(7, 4)):
    fig, ax = plt.subplots(figsize=size, dpi=100)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#1e1b4b", pad=12)
    ax.set_xlabel(xlabel, color="#4c1d95")
    ax.set_ylabel(ylabel, color="#4c1d95")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    return fig, ax


def bar_subject_marks(df, avg):
    fig, ax = _new("Subject-wise marks", "Subject", "Percentage (%)")
    labels = [_short(s) for s in df["Subject"]]
    bars = ax.bar(labels, df["Percentage"], color=[_band_color(p) for p in df["Percentage"]],
                  edgecolor="white", width=0.65)
    ax.bar_label(bars, labels=[f"{v:.0f}%" for v in df["Percentage"]], padding=3, fontsize=9)
    ax.axhline(avg, ls="--", color=PURPLE, lw=1.5, label=f"Average {avg:.1f}%")
    ax.set_ylim(0, 112)
    ax.legend(frameon=False)
    if len(labels) > 4:
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    return fig


def progress_line(prev_pct, current, predicted):
    fig, ax = _new("Academic progress", "", "Percentage (%)")
    labels, values = [], []
    if prev_pct is not None:
        labels.append("Previous\nsemester"); values.append(prev_pct)
    labels += ["Current\nsemester", "Predicted\nnext"]
    values += [current, predicted]
    x = np.arange(len(values))
    ax.plot(x[:-1], values[:-1], color=PURPLE, lw=3, marker="o", ms=9)
    ax.plot(x[-2:], values[-2:], color=PINK, lw=3, ls="--", marker="o", ms=9)
    ax.fill_between(x, values, alpha=0.12, color=PURPLE)
    for xi, v in zip(x, values):
        ax.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontweight="bold", color="#1e1b4b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.4, len(values) - 0.6)
    ax.set_ylim(0, 112)
    fig.tight_layout()
    return fig


def attendance_donut(attendance):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    present = float(np.clip(attendance, 0, 100))
    ax.pie([present, 100 - present], colors=[_band_color(present), "#e5e7eb"], startangle=90,
           counterclock=False, wedgeprops={"width": 0.32, "edgecolor": "white"})
    ax.text(0, 0.05, f"{present:.0f}%", ha="center", va="center", fontsize=26,
            fontweight="bold", color="#1e1b4b")
    ax.text(0, -0.22, "attendance", ha="center", va="center", color="#6b7280")
    ax.set_title("Attendance", fontsize=13, fontweight="bold", color="#1e1b4b")
    fig.tight_layout()
    return fig


def study_hours_chart(hours, preds, current_hours, current_pred):
    fig, ax = _new("Study hours vs expected marks", "Daily study hours", "Estimated next score (%)")
    ax.plot(hours, preds, color=BLUE, lw=3)
    ax.fill_between(hours, preds, alpha=0.12, color=BLUE)
    ax.scatter([current_hours], [current_pred], s=140, color=PINK, zorder=5, label="You (now)")
    ax.legend(frameon=False)
    ax.set_ylim(max(0, preds.min() - 10), min(105, preds.max() + 10))
    fig.tight_layout()
    return fig


def comparison_chart(df, target):
    fig, ax = _new("Your score vs target", "Subject", "Percentage (%)")
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["Percentage"], w, color=PURPLE, label="Your score")
    ax.bar(x + w / 2, [target] * len(df), w, color=PINK, alpha=0.6, label=f"Target ({target:.0f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([_short(s) for s in df["Subject"]], rotation=25 if len(df) > 4 else 0,
                       ha="right" if len(df) > 4 else "center")
    ax.set_ylim(0, 112)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
