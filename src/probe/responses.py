"""Canonical probe records and deterministic response summaries."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Mapping

from probe.selection import validated_metrics

STRATEGIES = ("A", "B", "C")
TIERS = ("High", "Medium", "Low", "Counterfactual")
FORWARD_SHAPES = ("monotone", "activity_cliff", "saturation", "flat_negative")
COUNTERFACTUAL_SIGNALS = ("load_bearing", "actively_harmful", "non_causal")


@dataclass(frozen=True)
class ResponseThresholds:
    """Thresholds for probe-response classification."""

    vina_epsilon: float = 0.1
    counterfactual_positive: float = 0.2
    sa_liability_drop: float = 0.15
    qed_drop: float = 0.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")


def normalize_probe(raw: Mapping[str, Any], original_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize input fields and retain failed probes explicitly."""
    if not isinstance(raw, Mapping):
        raise ValueError("Each probe must be an object")
    original = validated_metrics(original_metrics)
    probe_id = str(raw.get("probe_id", ""))
    strategy = raw.get("strategy_id", raw.get("hypothesis_id"))
    if strategy is None:
        match = re.search(r"(?:^|_)([ABC])(?:_|$)", probe_id)
        strategy = match.group(1) if match else None
    if isinstance(strategy, str):
        strategy = strategy.upper()
    aliases = {
        "high": "High", "high intensity": "High", "maximized": "High",
        "medium": "Medium", "medium intensity": "Medium", "moderate": "Medium",
        "low": "Low", "low intensity": "Low", "conservative": "Low",
        "counterfactual": "Counterfactual", "negativecontrol": "Counterfactual",
        "negative control": "Counterfactual", "negative_control": "Counterfactual",
    }
    tier = raw.get("intensity_tier")
    if tier is None:
        tier = next((part for part in probe_id.split("_") if part.lower() in aliases), None)
    tier = aliases.get(str(tier).strip().lower(), tier)
    record = {
        "probe_id": probe_id,
        "strategy_id": strategy,
        "intensity_tier": tier,
        "target_site": raw.get("target_site", raw.get("target_vector")),
        "site_id": raw.get("site_id"),
        "expected_signal": raw.get("expected_signal", raw.get("purpose", raw.get("theoretical_purpose", ""))),
        "modification_logic": deepcopy(raw.get("modification_logic", raw.get("mod_logic", {}))),
        "smiles": raw.get("smiles", raw.get("generated_smiles")),
        "status": raw.get("status", "ok"),
    }
    errors = []
    if raw.get("error"):
        errors.append(str(raw["error"]))
    if not probe_id:
        errors.append("Missing probe_id")
    if strategy not in STRATEGIES:
        errors.append(f"Unknown strategy_id: {strategy!r}")
    if tier not in TIERS:
        errors.append(f"Unknown intensity_tier: {tier!r}")
    if not record["target_site"]:
        errors.append("Missing target_site")
    if not isinstance(record["modification_logic"], dict) or not record["modification_logic"].get("action"):
        errors.append("Missing modification_logic.action")
    metric_input = raw.get("metrics")
    if metric_input is None:
        metric_input = {key: raw.get(key) for key in ("vina_score", "qed", "sa")}
        if raw.get("pose_path"):
            metric_input["pose_path"] = raw["pose_path"]
    try:
        record["metrics"] = validated_metrics(metric_input)
    except ValueError as exc:
        record["metrics"] = None
        errors.append(str(exc))
    if record["status"] != "ok":
        errors.append(f"Probe execution status: {record['status']}")
    if errors:
        record["status"] = "failed"
        record["error"] = "; ".join(dict.fromkeys(errors))
    else:
        metrics = record["metrics"]
        record.update(d_vina=metrics["vina_score"] - original["vina_score"],
                      d_qed=metrics["qed"] - original["qed"],
                      d_sa=metrics["sa"] - original["sa"])
    if raw.get("attempt_errors"):
        record["attempt_errors"] = deepcopy(raw["attempt_errors"])
    return record


def classify_forward(probes: Mapping[str, Mapping[str, Any]], thresholds: ResponseThresholds) -> str | None:
    """Classify only a complete, successfully evaluated forward triplet."""
    if any(tier not in probes or probes[tier].get("status", "ok") != "ok" for tier in TIERS[:3]):
        return None
    high, medium, low = (probes[tier]["d_vina"] for tier in TIERS[:3])
    if high > 0 and medium < -thresholds.vina_epsilon:
        return "activity_cliff"
    forward = (high, medium, low)
    if all(delta < -thresholds.vina_epsilon for delta in forward):
        gains = [-delta for delta in forward]
        return "saturation" if gains == sorted(gains) else "monotone"
    if any(delta < -thresholds.vina_epsilon for delta in forward):
        return "monotone"
    return "flat_negative"


def classify_counterfactual(probe: Mapping[str, Any] | None, thresholds: ResponseThresholds) -> str | None:
    if probe is None or probe.get("status", "ok") != "ok":
        return None
    if probe["d_vina"] > thresholds.counterfactual_positive:
        return "load_bearing"
    if probe["d_vina"] < -thresholds.vina_epsilon:
        return "actively_harmful"
    return "non_causal"


def summarize_responses(original_metrics: Mapping[str, Any], probe_results: list[dict[str, Any]],
                        thresholds: ResponseThresholds | None = None) -> dict[str, Any]:
    """Return auditable canonical records, coverage and text for induction."""
    thresholds = thresholds or ResponseThresholds()
    normalized = [normalize_probe(probe, original_metrics) for probe in probe_results]
    seen_ids, seen_slots = set(), set()
    for probe in normalized:
        identity = probe["probe_id"]
        slot = (probe["strategy_id"], probe["intensity_tier"])
        if identity and identity in seen_ids:
            raise ValueError(f"Duplicate probe_id: {identity}")
        if slot[0] in STRATEGIES and slot[1] in TIERS:
            if slot in seen_slots:
                raise ValueError(f"Duplicate probe tier: {slot[0]}/{slot[1]}")
            seen_slots.add(slot)
        seen_ids.add(identity)
    valid = [probe for probe in normalized if probe["status"] == "ok"]
    coverage = {"expected": 12, "received": len(normalized), "valid": len(valid), "strategies": {}}
    summary_lines = [f"Probe coverage: {len(valid)}/12 valid; {len(normalized)} records received.",
                     "Unavailable observations are unknown, not negative or non-causal evidence."]
    strategy_summaries = {}
    for strategy in STRATEGIES:
        records = [probe for probe in normalized if probe["strategy_id"] == strategy]
        by_tier = {probe["intensity_tier"]: probe for probe in records if probe["intensity_tier"] in TIERS}
        valid_tiers = [tier for tier in TIERS if tier in by_tier and by_tier[tier]["status"] == "ok"]
        missing = [tier for tier in TIERS if tier not in by_tier]
        failed = [tier for tier in TIERS if tier in by_tier and by_tier[tier]["status"] != "ok"]
        item = {"valid": len(valid_tiers), "expected": 4, "valid_tiers": valid_tiers,
                "missing_tiers": missing, "failed_tiers": failed,
                "forward_shape": classify_forward(by_tier, thresholds),
                "counterfactual_signal": classify_counterfactual(by_tier.get("Counterfactual"), thresholds)}
        coverage["strategies"][strategy] = deepcopy(item)
        strategy_summaries[strategy] = item
        summary_lines.extend([f"--- Strategy {strategy} response summary ---",
                              f"  valid={len(valid_tiers)}/4; missing={missing}; failed={failed}",
                              f"  forward_shape: {item['forward_shape'] or 'unavailable'}",
                              f"  counterfactual_signal: {item['counterfactual_signal'] or 'unavailable'}"])
        for probe in records:
            prefix = f"  - {probe['probe_id']} [tier={probe['intensity_tier']}, site={probe['target_site']}]"
            if probe["status"] != "ok":
                summary_lines.append(f"{prefix} FAILED: {probe.get('error', 'unknown error')}")
                continue
            logic = probe["modification_logic"]
            tags = []
            if probe["d_sa"] < -thresholds.sa_liability_drop:
                tags.append("[SA-LIABILITY]")
            if probe["d_qed"] < -thresholds.qed_drop:
                tags.append("[QED-DROP]")
            probe["response_tags"] = tags
            summary_lines.extend([prefix,
                                  f"      action: {logic['action']} | directive: {logic.get('modification_directive', '-')}",
                                  "      modification_logic: " + json.dumps(logic, ensure_ascii=False, sort_keys=True),
                                  f"      expected: {probe['expected_signal']}",
                                  f"      delta_Vina={probe['d_vina']:+.3f} | delta_QED={probe['d_qed']:+.3f} | delta_SA={probe['d_sa']:+.3f} {' '.join(tags)}"])
    unknown = [probe for probe in normalized if probe["strategy_id"] not in STRATEGIES]
    summary_lines.extend(f"Unassigned record {probe['probe_id']}: {probe.get('error', 'invalid strategy')}" for probe in unknown)
    highlights = ["=== Pre-computed highlights ==="]
    if valid:
        best = min(valid, key=lambda probe: probe["d_vina"])
        if best["d_vina"] < -thresholds.vina_epsilon:
            highlights.append(f"Best-affinity probe: {best['probe_id']} (delta_Vina={best['d_vina']:+.3f}).")
    wins = [probe["probe_id"] for probe in valid if probe["d_vina"] < 0 and probe["d_qed"] > 0 and probe["d_sa"] > 0]
    if wins:
        highlights.append("Joint-win probes (delta_Vina<0, delta_QED>0, delta_SA>0): " + ", ".join(wins) + ".")
    return {"normalized_probes": normalized, "coverage": coverage, "strategies": strategy_summaries,
            "response_summary": "\n".join(summary_lines), "global_highlights": "\n".join(highlights),
            "thresholds": vars(thresholds).copy()}
