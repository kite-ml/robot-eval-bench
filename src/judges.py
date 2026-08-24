"""How each model is called as a task-success judge, and how to add your own.

Every judge answers the same question about an episode: *was the task
accomplished, yes or no.* They differ only in how the episode is encoded (see
`selection.Episode.sample_frames` for keyframes; native MP4 or ~16 dense frames
for video) and in the API used to reach the model.

The prompt is identical across models. The verdict is a strict JSON object, parsed
tolerantly so a malformed answer counts as an abstention (`success=None`) rather
than a wrong guess.

To benchmark a NEW model, implement `Judge.judge(...)` for it and register it in
`MODELS`, then run `evaluate.py`. Nothing else changes: the ground truth, episode
selection, and scoring are all model-agnostic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

import numpy as np

# --- the shared prompt ------------------------------------------------------------

# Strict-evaluator system prompt. The judge scores ACCOMPLISHMENT, not whether the
# motion looked plausible. That distinction is what separates an eval from a vibe check.
JUDGE_SYSTEM = (
    "You are a meticulous robotics evaluation judge. You are given a robot manipulation "
    "episode (as frames, a video, or state) and a task instruction. Decide ONLY whether the "
    "task was actually ACCOMPLISHED by the end of the episode - not whether the motion looked "
    "reasonable. Be strict: partial progress, wrong object, or an unfinished/incoherent "
    "sequence is a FAILURE. Respond with ONLY a JSON object: "
    '{"success": true|false, "confidence": 0.0-1.0, "rationale": "one sentence"}.'
)


def task_prompt(task: str, *, n_frames: Optional[int] = None) -> str:
    parts = [f'Task instruction: "{task}"']
    if n_frames is not None:
        parts.append(f"You are shown {n_frames} frame(s) in time order.")
    parts.append("Was the task accomplished? Answer with the JSON object only.")
    return "\n".join(parts)


def parse_verdict(text: str) -> Tuple[Optional[bool], float, str]:
    """Extract (success, confidence, rationale). Returns (None, 0.0, ...) if unparseable."""
    if not text:
        return None, 0.0, "empty response"
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, 0.0, f"no JSON in response: {text[:120]}"
    try:
        data = json.loads(match.group(0))
    except (ValueError, json.JSONDecodeError):
        return None, 0.0, f"unparseable JSON: {match.group(0)[:120]}"
    success = data.get("success")
    if isinstance(success, str):
        success = success.strip().lower() in ("true", "yes", "success", "1")
    elif success is not None:
        success = bool(success)
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    return success, confidence, str(data.get("rationale", "")).strip()


@dataclass
class Verdict:
    success: Optional[bool]  # None = abstained / unparseable (counts as n_invalid)
    confidence: float = 0.0
    rationale: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class Judge(Protocol):
    """Implement this to add a model. `frames` are RGB uint8 arrays (H, W, 3)."""

    def judge(self, task: str, frames: List[np.ndarray], *, video_path: Optional[str] = None) -> Verdict:
        ...


# --- how each benchmarked model is reached ---------------------------------------

@dataclass
class ModelSpec:
    key: str
    label: str
    model_id: str
    provider: str            # gemini | anthropic | openai-compatible
    api: str                 # the SDK / endpoint used
    supports_native_video: bool
    token_param: str         # the max-output-tokens field name for this API
    send_temperature: bool   # reasoning models reject a custom temperature
    notes: str


# The five judges in the benchmark, and exactly how each was called.
MODELS: List[ModelSpec] = [
    ModelSpec("gemini-pro", "Gemini 3.1 pro", "gemini-3.1-pro-preview", "gemini",
              "google-genai generate_content", supports_native_video=True,
              token_param="max_output_tokens", send_temperature=True,
              notes="Video approach sends a real MP4 as an inline part; keyframes send stills as image parts."),
    ModelSpec("gemini-flash", "Gemini 3.6 flash", "gemini-3.6-flash", "gemini",
              "google-genai generate_content", supports_native_video=True,
              token_param="max_output_tokens", send_temperature=True,
              notes="Same path as Gemini 3.1 pro; cheaper and, here, more accurate."),
    ModelSpec("opus-5", "Claude Opus 5", "claude-opus-5", "anthropic",
              "Anthropic messages API", supports_native_video=False,
              token_param="max_tokens", send_temperature=True,
              notes="No native video: the video approach sends ~16 evenly-spaced frames as image blocks."),
    ModelSpec("gpt-sol", "GPT 5.6 Sol", "gpt-5.6-sol", "openai-compatible",
              "OpenAI chat.completions", supports_native_video=False,
              token_param="max_completion_tokens", send_temperature=False,
              notes="Reasoning model: rejects a custom temperature; frames sent as image_url data URIs."),
    ModelSpec("kimi", "Kimi 3", "kimi-k3", "openai-compatible",
              "Moonshot chat.completions (base_url=https://api.moonshot.ai/v1)", supports_native_video=False,
              token_param="max_tokens", send_temperature=False,
              notes="OpenAI-compatible; reasoning model, so temperature is left at the server default."),
    ModelSpec("gemini-flash-37", "Gemini 3.7 flash", "gemini-3.7-flash", "gemini",
              "google-genai generate_content", supports_native_video=True,
              token_param="max_output_tokens", send_temperature=True,
              notes="Successor to 3.6 flash; unlike its predecessor it holds up on video (0.89 vs 0.63)."),
    ModelSpec("muse", "Muse Spark 1.1", "muse-spark-1.1", "openai-compatible",
              "Meta Model API chat.completions (base_url=https://api.meta.ai/v1)", supports_native_video=False,
              token_param="max_tokens", send_temperature=True,
              notes="Meta Superintelligence Labs' multimodal reasoning model; frames sent as image_url data URIs. "
                    "It spends ~400 tokens reasoning per call, so give it a generous output budget."),
]

# Frame budgets used by the benchmark.
KEYFRAMES_K = 4        # evenly-spaced stills, first and last frame included
VIDEO_DENSE_FRAMES = 16  # frames sent when a model has no native video path
