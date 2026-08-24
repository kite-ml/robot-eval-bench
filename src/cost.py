"""How per-episode cost is calculated in robot-eval-bench.

Cost is decoupled from the run: every judge call records its input/output token
counts, and dollar cost is computed after the fact from an editable price table.
That way you can re-price the whole benchmark by editing one dict, and compare
models on a like-for-like basis.

The figure reported per (model, approach) is a macro-average: for each dataset we
compute total_cost / episodes_judged, then average across datasets so every
dataset counts equally regardless of how many episodes it has.

Two details that matter for honest numbers:

  * The denominator is every episode the judge was *called on*, including the ones
    where it abstained or returned an unparseable answer (``n_invalid``). You paid
    for those calls, so they belong in the cost.
  * Gemini ingests video natively, and those early runs predate token logging, so
    for them we fall back to the run's recorded ``cost_usd``. Every other model is
    priced from tokens.
"""

from __future__ import annotations

from statistics import mean
from typing import Dict, List, Optional

# USD per 1,000,000 tokens (input, output). Edit to re-price the benchmark.
# These are best-estimate list prices at the time of the run.
PRICE_PER_MILLION: Dict[str, Dict[str, float]] = {
    "gemini-pro":      {"input": 1.25, "output": 10.0},
    "gemini-flash":    {"input": 0.30, "output": 2.5},
    "gemini-flash-37": {"input": 0.30, "output": 2.5},   # flash-tier, best estimate
    "opus-5":          {"input": 5.00, "output": 25.0},
    "gpt-sol":         {"input": 5.00, "output": 20.0},
    "kimi":            {"input": 0.60, "output": 2.5},
    "muse":            {"input": 1.25, "output": 4.25},  # Meta Model API list price
}


def cell_cost_per_episode(
    model: str,
    *,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    n: int,
    n_invalid: int = 0,
    cost_usd: Optional[float] = None,
) -> Optional[float]:
    """Per-episode cost for one (dataset, model, approach) cell.

    Prefers tokens x price; falls back to the run's baked ``cost_usd`` when tokens
    were not recorded (the native-video Gemini runs). Divides by every episode the
    judge was called on, abstentions included.
    """
    episodes = (n or 0) + (n_invalid or 0)
    if episodes <= 0:
        return None
    if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
        p = PRICE_PER_MILLION[model]
        total = (input_tokens * p["input"] + output_tokens * p["output"]) / 1e6
    elif isinstance(cost_usd, (int, float)):
        total = cost_usd
    else:
        return None
    return total / episodes


def model_cost_per_episode(per_dataset_costs: List[float]) -> Optional[float]:
    """Macro-average the per-dataset per-episode costs (each dataset weighted equally)."""
    costs = [c for c in per_dataset_costs if c is not None]
    return mean(costs) if costs else None
