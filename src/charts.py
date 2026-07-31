"""Reproduce the benchmark charts from data/results.json.

Two figures, matching the ones in the write-up:

  1. accuracy_by_approach.png  - grouped bars, keyframes vs video, per judge.
  2. cost_vs_accuracy_keyframes.png - the keyframes cost/accuracy scatter with the
     efficient frontier (the points nothing beats on both price and accuracy).

Run:  python src/charts.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "results.json")
OUT = os.path.join(ROOT, "charts")
GREEN = "#10b981"

# Per-model keyframes cost is read straight from results.json; these are the
# x-values of the scatter. Accuracy is the y-value.


def load():
    with open(DATA) as f:
        return json.load(f)


def cell(res, model, approach):
    return next(c for c in res["cells"] if c["model"] == model and c["approach"] == approach)


def accuracy_chart(res):
    models = res["models"]
    labels = [res["model_labels"][m] for m in models]
    kf = [cell(res, m, "keyframes")["mean_accuracy"] for m in models]
    vid = [cell(res, m, "video")["mean_accuracy"] for m in models]

    x = range(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar([i - w / 2 for i in x], kf, w, label="keyframes", color="#111111")
    ax.bar([i + w / 2 for i in x], vid, w, label="video", color="#9ca3af")
    for i, (a, b) in enumerate(zip(kf, vid)):
        ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, b + 0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Task-success judging accuracy: keyframes vs video")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "accuracy_by_approach.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def _frontier(points):
    """Points (cost, acc, label) that nothing beats on both lower cost and higher accuracy."""
    out = []
    for p in points:
        if not any(q is not p and q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]) for q in points):
            out.append(p)
    return sorted(out, key=lambda t: t[0])


def cost_scatter(res):
    pts = []
    for m in res["models"]:
        c = cell(res, m, "keyframes")
        pts.append((c["cost_per_episode_usd"], c["mean_accuracy"], res["model_labels"][m]))
    front = _frontier(pts)

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(front) > 1:
        ax.plot([p[0] for p in front], [p[1] for p in front], "--", color=GREEN, lw=1.4, alpha=0.7, zorder=1)
    for cost, acc, label in pts:
        on_front = any(f[2] == label for f in front)
        ax.scatter(cost, acc, s=70, color=GREEN if on_front else "#111111", zorder=2)
        ax.annotate(label, (cost, acc), textcoords="offset points", xytext=(9, -3), fontsize=9)
    ax.set_xlabel("cost per episode (USD)")
    ax.set_ylabel("accuracy")
    ax.set_title("Keyframes: accuracy vs cost per episode  (green = efficient frontier)")
    ax.set_xlim(left=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "cost_vs_accuracy_keyframes.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = load()
    accuracy_chart(res)
    cost_scatter(res)
