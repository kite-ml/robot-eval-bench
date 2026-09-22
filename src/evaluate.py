"""Score judges against the ground truth, and the entry point for adding a new one.

Two things live here:

  1. `score()` - turn per-episode predictions into accuracy / precision / recall / f1
     against the ground-truth labels. Model-agnostic.

  2. A runnable check - re-score every benchmarked model straight from
     `data/episodes.jsonl` (the per-episode predictions we shipped) and print the
     table. This needs no API keys and no dataset downloads: it verifies the
     headline numbers from the raw data.

To benchmark a NEW model:

    from judges import Judge, Verdict            # implement Judge.judge(...) for your model
    from selection import build_labeled_set      # reconstruct the exact labeled set
    # 1. Load a dataset's demos as selection.Episode objects (LeRobot loader of your choice).
    # 2. labeled = build_labeled_set(demos, negatives=cfg["negatives"], seed=cfg["seed"])
    #    using the (negatives, seed) from data/ground_truth.json[dataset]["config"].
    # 3. For each item, encode per approach (item.episode.sample_frames(4) for keyframes;
    #    native MP4 or 16 frames for video) and call your judge.
    # 4. score({eid: verdict.success}, {eid: item.label}) and compare.

Because the labeled set is fully determined by (dataset, seed, negatives), your run
grades against exactly the same episodes and labels every other model saw.

Run:  python src/evaluate.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from statistics import mean
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def score(pred: Dict[str, Optional[bool]], truth: Dict[str, Optional[bool]]) -> dict:
    """Accuracy et al. for one set of predictions vs the ground truth.

    Matches the harness definition exactly: a `None` prediction (the judge abstained
    or returned unparseable JSON) counts as INCORRECT for accuracy - the judge was
    asked and did not answer - so it stays in the denominator. Precision and recall
    exclude abstentions from their numerators. Episodes whose ground-truth label is
    `None` (human-marked "exclude") are dropped entirely.
    """
    tp = fp = tn = fn = n_invalid = 0
    for eid, gt in truth.items():
        if gt is None:  # excluded from scoring
            continue
        p = pred.get(eid)
        if p is None:
            n_invalid += 1
            continue
        if gt and p:
            tp += 1
        elif gt and not p:
            fn += 1
        elif (not gt) and p:
            fp += 1
        else:
            tn += 1
    n = tp + fp + tn + fn + n_invalid  # total episodes asked, abstentions included
    correct = tp + tn
    acc = correct / n if n else None
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {"n": n, "n_invalid": n_invalid, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def rescore_from_data() -> None:
    """Recompute each model's per-approach accuracy from the shipped predictions."""
    per = defaultdict(lambda: defaultdict(lambda: {"pred": {}, "truth": {}}))  # (model,approach) -> dataset -> ...
    with open(os.path.join(ROOT, "data", "episodes.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            if e.get("excluded"):
                continue
            b = per[(e["model_label"], e["approach"])][e["dataset"]]
            b["pred"][e["episode_id"]] = e["predicted"]
            b["truth"][e["episode_id"]] = e["ground_truth"]

    # The original five lead the table; any other judge with per-episode data
    # (later additions backfilled into episodes.jsonl) follows in discovery order.
    order = ["Gemini 3.1 pro", "Gemini 3.6 flash", "Claude Opus 5", "GPT 5.6 Sol", "Kimi 3"]
    order += [lbl for lbl in dict.fromkeys(lbl for lbl, _ in per) if lbl not in order]
    print(f"{'judge':<19}{'keyframes':>11}{'video':>9}   (mean accuracy across datasets)")
    for label in order:
        row = [label]
        for ap in ("keyframes", "video"):
            accs = [score(d["pred"], d["truth"])["accuracy"] for d in per[(label, ap)].values()]
            accs = [a for a in accs if a is not None]
            row.append(f"{mean(accs):.2f}" if accs else "  - ")
        print(f"{row[0]:<19}{row[1]:>11}{row[2]:>9}")


if __name__ == "__main__":
    rescore_from_data()
