"""Intent/outcome diagnostics over *all* attempted optimization designs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
from typing import Any

INTENTS = ("affinity", "druggability", "joint")
OUTCOMES = ("joint", "affinity_only", "druggability_only", "neither")


def _intent_label(value):
    if not isinstance(value, str):
        return None
    return {"affinity": "affinity", "druggability": "druggability", "joint": "joint", "both": "joint",
            "coopt": "joint", "co-optimization": "joint", "co_optimization": "joint"}.get(value.strip().lower())


def _candidate_intent(candidate):
    # An explicit unknown annotation must not silently fall back to a role.
    for key in ("diagnostic_intent", "declared_intent"):
        if key in candidate:
            return _intent_label(candidate[key]), "explicit_annotation"
    intent = candidate.get("intent")
    if isinstance(intent, str):
        return _intent_label(intent), "explicit_annotation"
    if isinstance(intent, Mapping):
        for key in ("label", "declared_intent"):
            if key in intent:
                return _intent_label(intent[key]), "explicit_annotation"
    role = candidate.get("role")
    if role is None and isinstance(intent, Mapping):
        role = intent.get("declared_by")
    if role is not None:
        return _intent_label(role), "role_mapping"
    return None, "unavailable"


def evaluate_diagnostics(results: list[dict]) -> dict[str, Any]:
    """Evaluate complete PROBE ``result.json`` objects, not winner records."""
    if not isinstance(results, list):
        raise ValueError("results must be a list of complete PROBE result objects")
    steps, rounds, missing_slots, extra_slots = [], 0, 0, 0
    for result_index, result in enumerate(results):
        if not isinstance(result, Mapping) or not isinstance(result.get("history"), list):
            raise ValueError("Diagnostics requires result.history; winner-only records are insufficient")
        result_id = result.get("case_id", result.get("run_id", f"result_{result_index + 1:04d}"))
        for round_index, round_record in enumerate(result["history"]):
            if not isinstance(round_record, Mapping) or not isinstance(round_record.get("candidates"), list):
                raise ValueError("Every history entry must contain a candidates list")
            candidates = round_record["candidates"]
            rounds += 1
            missing_slots += max(0, 3 - len(candidates))
            extra_slots += max(0, len(candidates) - 3)
            seen = set()
            for candidate_index, candidate in enumerate(candidates):
                if not isinstance(candidate, Mapping):
                    raise ValueError("Every history candidate must be an object")
                identity = candidate.get("candidate_id", f"candidate_{candidate_index + 1:02d}")
                if not isinstance(identity, str) or identity in seen:
                    raise ValueError("Candidate IDs must be strings and unique within each round")
                seen.add(identity)
                label, source = _candidate_intent(candidate)
                steps.append({"result_id": result_id, "round": round_record.get("round", round_index + 1),
                              "candidate_id": identity, "intent": label, "intent_source": source,
                              "status": candidate.get("status", "ok"), "error": candidate.get("error"),
                              "parent_metrics": round_record.get("parent_metrics"), "metrics": candidate.get("metrics")})
    report = evaluate_steps(steps)
    report["counts"].update(results=len(results), rounds=rounds)
    report["coverage"].update(expected_design_slots=3 * rounds, missing_design_slots=missing_slots,
                              extra_design_records=extra_slots)
    return report


def _metric(metrics, name, owner, reasons):
    value = metrics.get(name) if isinstance(metrics, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        reasons.append(f"{owner}.{name} is missing or not finite numeric data")
        return None
    if name in ("qed", "sa") and not 0 <= value <= 1:
        reasons.append(f"{owner}.{name} is outside [0, 1]; SA must be normalized and higher-is-better")
        return None
    return float(value)


def _observation(step, index):
    label = _intent_label(step.get("intent"))
    status = step.get("status", "ok")
    source = step.get("intent_source", "provided" if label is not None else "unavailable")
    result = {"result_id": step.get("result_id"), "round": step.get("round"),
              "candidate_id": step.get("candidate_id", f"step_{index + 1:04d}"),
              "intent": label, "intent_source": source, "status": status,
              "delta_a": None, "delta_d": None, "affinity_improved": None,
              "druggability_improved": None, "outcome": None, "unavailable_reasons": []}
    reasons = result["unavailable_reasons"]
    if status != "ok":
        reasons.append(f"Attempt status is {status!r}; no successful observed outcome is asserted")
        if step.get("error"):
            reasons.append(str(step["error"]))
        return result
    parent, child = step.get("parent_metrics"), step.get("metrics")
    pv = _metric(parent, "vina_score", "parent", reasons)
    cv = _metric(child, "vina_score", "candidate", reasons)
    pq = _metric(parent, "qed", "parent", reasons)
    ps = _metric(parent, "sa", "parent", reasons)
    cq = _metric(child, "qed", "candidate", reasons)
    cs = _metric(child, "sa", "candidate", reasons)
    if pv is not None and cv is not None:
        delta_a = pv - cv
        if math.isfinite(delta_a):
            result["delta_a"] = delta_a
            result["affinity_improved"] = delta_a > 0
        else:
            reasons.append("Affinity delta overflowed; no finite affinity response is available")
    if all(value is not None for value in (pq, ps, cq, cs)):
        result["delta_d"] = math.fsum((cq, cs)) - math.fsum((pq, ps))
        result["druggability_improved"] = result["delta_d"] > 0
    if result["delta_a"] is not None and result["delta_d"] is not None:
        a, d = result["affinity_improved"], result["druggability_improved"]
        result["outcome"] = "joint" if a and d else "affinity_only" if a else "druggability_only" if d else "neither"
    return result


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate_steps(steps: list[dict]) -> dict[str, Any]:
    """Evaluate normalized trajectories with explicit intent annotations."""
    if not isinstance(steps, list) or any(not isinstance(step, Mapping) for step in steps):
        raise ValueError("steps must be a list of normalized trajectory objects")
    observations = [_observation(step, index) for index, step in enumerate(steps)]
    declared = {intent: sum(row["intent"] == intent for row in observations) for intent in INTENTS}
    known = sum(declared.values())
    sources = Counter(str(row["intent_source"]) for row in observations)
    outcomes = {outcome: sum(row["outcome"] == outcome for row in observations) for outcome in OUTCOMES}
    both_observed = sum(outcomes.values())
    a_observed = sum(row["delta_a"] is not None for row in observations)
    d_observed = sum(row["delta_d"] is not None for row in observations)
    ioc_num, ioc_den = {}, {}
    oi_num, oi_den = {}, {}
    by_intent = {}
    for intent in INTENTS:
        rows = [row for row in observations if row["intent"] == intent]
        available_a = [row for row in rows if row["delta_a"] is not None]
        available_d = [row for row in rows if row["delta_d"] is not None]
        available_both = [row for row in rows if row["outcome"] is not None]
        available = available_a if intent == "affinity" else available_d if intent == "druggability" else available_both
        ioc_den[intent] = len(available)
        if intent == "affinity":
            ioc_num[intent] = sum(row["delta_a"] > 0 for row in available)
        elif intent == "druggability":
            ioc_num[intent] = sum(row["delta_d"] > 0 for row in available)
        else:
            ioc_num[intent] = sum(row["delta_a"] > 0 and row["delta_d"] > 0 for row in available)
        selected_improved = []
        if intent != "joint":
            chosen, other = ("delta_a", "delta_d") if intent == "affinity" else ("delta_d", "delta_a")
            selected_improved = [row for row in available if row[chosen] > 0]
            oi_available = [row for row in selected_improved if row[other] is not None]
            oi_den[intent] = len(oi_available)
            oi_num[intent] = sum(row[other] < 0 for row in oi_available)
        else:
            oi_num[intent], oi_den[intent] = None, None
        by_intent[intent] = {"declared": len(rows), "affinity_observed": len(available_a),
                             "druggability_observed": len(available_d), "both_observed": len(available_both),
                             "ioc_observed": len(available), "ioc_unobserved": len(rows) - len(available),
                             "oi_selected_improvements": len(selected_improved) if intent != "joint" else None,
                             "oi_unobserved_other_axis": len(selected_improved) - oi_den[intent] if intent != "joint" else None}
    count = len(observations)
    return {
        "schema_version": 1, "rate_unit": "fraction",
        "counts": {"results": None, "rounds": None, "attempts": count, "declared_intents": declared,
                   "unknown_intent": count - known, "intent_sources": dict(sources),
                   "failed_attempts": sum(row["status"] != "ok" for row in observations),
                   "affinity_observed": a_observed, "druggability_observed": d_observed,
                   "both_observed": both_observed,
                   "no_observed_outcome": sum(row["delta_a"] is None and row["delta_d"] is None for row in observations),
                   "outcomes": outcomes},
        "intent_ratios": {intent: _ratio(declared[intent], known) for intent in INTENTS},
        "ioc": {intent: _ratio(ioc_num[intent], ioc_den[intent]) for intent in INTENTS},
        "oi": {intent: _ratio(oi_num[intent], oi_den[intent]) if intent != "joint" else None for intent in INTENTS},
        "outcome_ratios": {outcome: _ratio(outcomes[outcome], both_observed) for outcome in OUTCOMES},
        "numerators": {"ioc": ioc_num, "oi": oi_num},
        "denominators": {"intent_ratios": known, "ioc": ioc_den, "oi": oi_den, "outcome_ratios": both_observed},
        "coverage": {"known_intent_fraction": _ratio(known, count), "affinity_observed_fraction": _ratio(a_observed, count),
                     "druggability_observed_fraction": _ratio(d_observed, count), "both_observed_fraction": _ratio(both_observed, count),
                     "by_intent": by_intent, "expected_design_slots": None, "missing_design_slots": None, "extra_design_records": None},
        "definitions": {"affinity": "A = -Vina", "druggability": "D = QED + normalized SA",
                        "reference": "Each candidate's actual round parent, not the initial ligand or final winner",
                        "improvement": "Strict delta > 0; equality is not improvement",
                        "interference": "Strict other-axis delta < 0, conditioned on selected-axis improvement",
                        "missingness": "Rates condition on the required observed axes; inspect coverage for unavailable outcomes",
                        "intent_note": "Role mapping is a declared-agent-role proxy, not the paper's reasoning-text intent extraction. Explicit annotations override roles."},
        "observations": observations,
    }
