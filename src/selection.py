"""Episode selection and the labeled-set construction used by robot-eval-bench.

Every dataset in the benchmark is a set of open-source LeRobot demonstrations,
which are almost entirely *successes*. A judge is only useful if it can tell a
failure from a success, so we build a class-balanced labeled set: each demo is a
success, and one matched *negative* is synthesized per demo by corrupting the
episode in a well-defined way. The negative's failure mode is recorded so the
report can show which approach catches which kind of failure.

This mirrors the production harness. The exact labeled set for a dataset is fully
determined by (dataset, seed, negatives, n_positive_requested), all of which are
stored per dataset in `data/ground_truth.json` so anyone can reconstruct it.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

# Negative-synthesis operators. Each maps a success episode to a failed one.
NEGATIVE_OPS = ("truncate", "mismatch", "shuffle", "reverse")


@dataclass
class Episode:
    """A robot episode: an ordered list of RGB frames plus its task instruction."""

    episode_id: str
    task: str
    frames: List[np.ndarray] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.frames)

    def sample_frames(self, k: int) -> List[np.ndarray]:
        """Return ``k`` frames evenly spaced across the episode, start and end inclusive.

        This is exactly how the KEYFRAMES approach picks its stills. The benchmark
        uses k=4, so a judge sees frames at 0, 1/3, 2/3, and the final frame, and
        has to infer the motion in between.
        """
        n = self.length
        if n == 0:
            return []
        if k >= n:
            return list(self.frames)
        if k <= 1:
            return [self.frames[-1]]
        idx = [round(i * (n - 1) / (k - 1)) for i in range(k)]
        seen, out = set(), []
        for i in idx:  # de-dup while preserving order (short episodes collapse indices)
            if i not in seen:
                seen.add(i)
                out.append(self.frames[i])
        return out


@dataclass
class LabeledItem:
    episode: Episode
    label: bool                        # True = task accomplished (success)
    source: str                        # "demo" (success) | "synthetic" (negative)
    failure_mode: Optional[str] = None  # set on negatives: truncate|mismatch|shuffle|reverse


# --- negative-synthesis operators -------------------------------------------------

def _truncate(ep: Episode, rng: random.Random) -> Episode:
    """Stop the episode early so the objective is never completed."""
    cut = max(1, int(ep.length * rng.uniform(0.3, 0.6)))
    return dataclasses.replace(ep, episode_id=f"{ep.episode_id}#truncate", frames=ep.frames[:cut])


def _mismatch(ep: Episode, other_task: str) -> Episode:
    """Pair a successful video with a different task it does not satisfy."""
    return dataclasses.replace(ep, episode_id=f"{ep.episode_id}#mismatch", task=other_task)


def _shuffle(ep: Episode, rng: random.Random) -> Episode:
    """Temporally scramble frames so the execution is incoherent."""
    frames = list(ep.frames)
    rng.shuffle(frames)
    return dataclasses.replace(ep, episode_id=f"{ep.episode_id}#shuffle", frames=frames)


def _reverse(ep: Episode) -> Episode:
    """Play the episode backwards: the manipulation is undone, not done."""
    return dataclasses.replace(ep, episode_id=f"{ep.episode_id}#reverse", frames=list(reversed(ep.frames)))


def _distinct_task(episodes: Sequence[Episode], current: str, rng: random.Random) -> Optional[str]:
    tasks = list({e.task for e in episodes if e.task and e.task != current})
    return rng.choice(tasks) if tasks else None


def build_labeled_set(
    episodes: Sequence[Episode],
    *,
    negatives: Sequence[str] = NEGATIVE_OPS,
    seed: int = 0,
) -> List[LabeledItem]:
    """Turn raw demo episodes into a class-balanced labeled set.

    Each demo becomes a success; one synthetic negative is generated per demo by
    cycling through ``negatives``. Deterministic given ``seed``.
    """
    rng = random.Random(seed)
    episodes = [e for e in episodes if e.length > 0]
    ops = [op for op in negatives if op in NEGATIVE_OPS]
    if not ops:
        raise ValueError(f"no valid negative ops in {negatives!r}; choose from {NEGATIVE_OPS}")

    items: List[LabeledItem] = []
    for i, ep in enumerate(episodes):
        items.append(LabeledItem(episode=ep, label=True, source="demo"))
        op = ops[i % len(ops)]
        neg = _make_negative(op, ep, episodes, rng)
        if neg is not None:
            items.append(LabeledItem(episode=neg, label=False, source="synthetic", failure_mode=op))
    return items


def _make_negative(op: str, ep: Episode, episodes: Sequence[Episode], rng: random.Random) -> Optional[Episode]:
    if op == "truncate":
        return _truncate(ep, rng)
    if op == "shuffle":
        return _shuffle(ep, rng)
    if op == "reverse":
        return _reverse(ep)
    if op == "mismatch":
        other = _distinct_task(episodes, ep.task, rng)
        # Skip rather than emit a mislabeled "negative" when every episode shares one task.
        return _mismatch(ep, other) if other else None
    raise ValueError(f"unknown negative op: {op}")
