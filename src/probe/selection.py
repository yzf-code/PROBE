"""Initial-ligand-referenced balanced hypervolume selection."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def validated_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Read canonical metrics without converting missing scores to zero."""
    if not isinstance(value, Mapping):
        raise ValueError("Metrics must be an object with vina_score, qed and sa")
    result = dict(value)
    for key in ("vina_score", "qed", "sa"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"Missing or non-numeric metric: {key}")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite metric: {key}")
        if key in ("qed", "sa") and not 0.0 <= number <= 1.0:
            raise ValueError(f"{key} must be in [0, 1] (SA must be normalized)")
        result[key] = number
    return result


def balanced_score(metrics: Mapping[str, Any], original_metrics: Mapping[str, Any]) -> float:
    """Strict normalized hypervolume against the *original* input ligand."""
    original = validated_metrics(original_metrics)
    current = validated_metrics(metrics)
    a0 = -original["vina_score"]
    d0 = original["qed"] + original["sa"]
    if a0 == 0 or d0 <= 0:
        raise ValueError("Balanced selection requires nonzero initial Vina and positive initial QED+SA")
    a_gain = -current["vina_score"] - a0
    d_gain = current["qed"] + current["sa"] - d0
    if a_gain <= 0 or d_gain <= 0:
        return 0.0
    return (a_gain / abs(a0)) * (d_gain / d0)


def select_winner(candidates: Iterable[dict[str, Any]], original_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return the highest scoring successful candidate, stably breaking ties."""
    balanced_score(original_metrics, original_metrics)
    winner = None
    best_score = -1.0
    for candidate in candidates:
        if candidate.get("status", "ok") != "ok" or not candidate.get("smiles"):
            continue
        try:
            score = balanced_score(candidate.get("metrics", candidate), original_metrics)
        except ValueError:
            continue
        if score > best_score:
            winner, best_score = candidate, score
    if winner is None:
        raise ValueError("No successfully evaluated candidate is available for selection")
    return winner
