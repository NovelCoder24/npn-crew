"""Task graders for OpenEnv manifest entrypoints.

These functions expose stable entrypoints used in openenv.yaml and are tolerant
to slightly different validator call signatures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


MAX_STEPS = 84


def _clip_0_1(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _extract_inputs(*args: Any, **kwargs: Any) -> Tuple[Any, List[float], Dict[str, Any]]:
    """Extract observation, rewards, and metadata from flexible grader signatures."""
    obs = kwargs.get("observation") or kwargs.get("obs")
    rewards = kwargs.get("rewards") or []
    metadata = kwargs.get("metadata") or {}

    for arg in args:
        if isinstance(arg, dict):
            if obs is None and ("observation" in arg or "obs" in arg):
                obs = arg.get("observation") or arg.get("obs")
            if not rewards and "rewards" in arg and isinstance(arg["rewards"], list):
                rewards = arg["rewards"]
            if not metadata and "metadata" in arg and isinstance(arg["metadata"], dict):
                metadata = arg["metadata"]
        elif obs is None and hasattr(arg, "muscle_size"):
            obs = arg
        elif not rewards and isinstance(arg, list):
            rewards = arg

    if obs is not None and not metadata:
        maybe_metadata = getattr(obs, "metadata", None)
        if isinstance(maybe_metadata, dict):
            metadata = maybe_metadata

    return obs, rewards, metadata


def _muscle_score(obs: Any) -> float:
    if obs is None:
        return 0.0
    return _clip_0_1((float(getattr(obs, "muscle_size", 50.0)) - 50.0) / 50.0)


def grade_easy(*args: Any, **kwargs: Any) -> float:
    """Easy task: maximize final muscle size."""
    obs, _rewards, _metadata = _extract_inputs(*args, **kwargs)
    return _muscle_score(obs)


def grade_medium(*args: Any, **kwargs: Any) -> float:
    """Medium task: muscle gain with lower average fatigue."""
    obs, _rewards, metadata = _extract_inputs(*args, **kwargs)
    muscle = _muscle_score(obs)
    avg_fatigue = float(metadata.get("avg_fatigue", getattr(obs, "fatigue", 0.5) if obs is not None else 0.5))
    return _clip_0_1((1.0 - avg_fatigue) * muscle)


def grade_hard(*args: Any, **kwargs: Any) -> float:
    """Hard task: muscle gain with overtraining control."""
    obs, _rewards, metadata = _extract_inputs(*args, **kwargs)
    muscle = _muscle_score(obs)
    overtrain_days = float(metadata.get("overtrain_days", 0.0))
    overtrain_ratio = overtrain_days / float(MAX_STEPS)
    no_overtrain = 1.0 - overtrain_ratio
    return _clip_0_1(muscle * 0.6 + no_overtrain * 0.4)
