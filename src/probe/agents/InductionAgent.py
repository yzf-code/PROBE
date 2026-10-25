"""Induce a site-indexed EditManual from measured probe responses."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
from typing import Any

from probe.prompts import render
from probe.responses import ResponseThresholds, summarize_responses

logger = logging.getLogger(__name__)


class InductionAgent:
    def __init__(self, client, thresholds: ResponseThresholds | None = None):
        self.client = client
        self.thresholds = thresholds or ResponseThresholds()
        self.last_prompt = ""

    def execute(self, original_metrics, diagnosis_report, strategies_block, probe_results) -> dict[str, Any]:
        """Return EditManual, summaries, coverage and canonical probe records."""
        if not isinstance(diagnosis_report, dict) or not isinstance(diagnosis_report.get("sites"), list):
            raise ValueError("Induction requires a structured diagnosis with a sites list")
        if not isinstance(strategies_block, dict) or set(strategies_block) != {"A", "B", "C"}:
            raise ValueError("Induction requires strategies A, B and C")
        sites = diagnosis_report["sites"]
        site_ids = [site.get("site_id") for site in sites]
        if not site_ids or any(not identity for identity in site_ids) or len(set(site_ids)) != len(site_ids):
            raise ValueError("Diagnosis site_id values must be nonempty and unique")
        fragments = [site.get("fragment_name") for site in sites]
        if any(not name for name in fragments) or len(set(fragments)) != len(fragments):
            raise ValueError("Diagnosis fragment_name values must be nonempty and unique")
        fragment_to_site = {site["fragment_name"]: site["site_id"] for site in sites}
        summary = summarize_responses(original_metrics, probe_results, self.thresholds)
        for probe in summary["normalized_probes"]:
            mapped_site = fragment_to_site.get(probe["target_site"])
            if probe["status"] == "ok" and mapped_site is None:
                raise ValueError(f"Probe {probe['probe_id']} targets an unknown fragment: {probe['target_site']}")
            if probe.get("site_id") is not None and mapped_site is not None and probe["site_id"] != mapped_site:
                raise ValueError(f"Probe {probe['probe_id']} has inconsistent site_id and target_site")
            if mapped_site is not None:
                probe["site_id"] = mapped_site
        self.last_prompt = render(
            "edit_manual",
            diagnosis_report=json.dumps(diagnosis_report, ensure_ascii=False, indent=2),
            strategies_block=json.dumps(strategies_block, ensure_ascii=False, indent=2),
            global_highlights=summary["global_highlights"],
            response_summary=summary["response_summary"],
        ) + "\n\n[Structured identity and incomplete evidence]\n" + (
            "Use exactly the site_id values in the diagnosis, with one Per_Site_Record per site. "
            "target_site in the probe records denotes fragment_name; the identity map is "
            + json.dumps(fragment_to_site, ensure_ascii=False)
            + ". Include exactly one Strategy_Verdict for A, B and C. "
            "Copy the analyzer's observed labels; unavailable labels must be JSON null. "
            "Do not interpret failed or missing probes as adverse, flat or non-causal evidence. "
            "Cite only successfully evaluated probe IDs belonging to that site. "
            "A strategy with no valid evidence must have verdict null; incomplete evidence "
            "cannot establish a fully validated verdict. Keep uncertain prior-based advice explicit."
        )
        if summary["coverage"]["valid"] != 12:
            logger.warning("Induction has %s/12 valid probes; incomplete coverage will be retained", summary["coverage"]["valid"])
        response = self.client.json(self.last_prompt)
        if not isinstance(response, dict) or not isinstance(response.get("EditManual"), dict):
            raise ValueError("Induction response must contain an EditManual object")
        manual = deepcopy(response["EditManual"])
        self._validate_manual(manual, sites, strategies_block, summary)
        return {"EditManual": manual, **summary}

    @staticmethod
    def _validate_manual(manual, sites, strategies, summary):
        for field, expected_type in (("Global_Summary", str), ("Strategy_Verdicts", list),
                                     ("Per_Site_Records", list), ("Combinatorial_Guidance", dict)):
            if not isinstance(manual.get(field), expected_type):
                raise ValueError(f"EditManual.{field} must be {expected_type.__name__}")
        verdicts = manual["Strategy_Verdicts"]
        if len(verdicts) != 3 or any(not isinstance(item, dict) for item in verdicts) or {item.get("strategy_id") for item in verdicts} != {"A", "B", "C"}:
            raise ValueError("EditManual requires exactly one verdict for each of A, B and C")
        for verdict in verdicts:
            strategy_id = verdict["strategy_id"]
            evidence = summary["strategies"][strategy_id]
            verdict["forward_shape"] = evidence["forward_shape"]
            verdict["counterfactual_signal"] = evidence["counterfactual_signal"]
            verdict["claimed_tradeoff"] = strategies[strategy_id]["trade_off"]
            verdict["coverage"] = deepcopy(evidence)
            if not isinstance(verdict.get("reasoning"), str):
                raise ValueError(f"Strategy {strategy_id} needs a textual reasoning")
            if evidence["valid"] == 0:
                verdict["verdict"] = None
                verdict["reasoning"] += " No successfully evaluated probes are available; no empirical verdict is established."
            else:
                if verdict.get("verdict") not in {"validated", "partially_validated", "invalidated"}:
                    raise ValueError(f"Invalid strategy verdict for {strategy_id}")
                if evidence["valid"] < 4:
                    if verdict["verdict"] == "validated":
                        verdict["verdict"] = "partially_validated"
                    verdict["reasoning"] += " Probe coverage is incomplete; unavailable responses remain unknown."
        records = manual["Per_Site_Records"]
        expected_ids = {site["site_id"] for site in sites}
        if len(records) != len(sites) or any(not isinstance(item, dict) for item in records) or {item.get("site_id") for item in records} != expected_ids:
            raise ValueError("EditManual must contain exactly one record for every diagnosed site")
        probes = {probe["probe_id"]: probe for probe in summary["normalized_probes"] if probe["status"] == "ok"}
        axes = {"size", "polarity", "flexibility", "shape", "charge"}
        for record in records:
            for field in ("originating_motif", "pharmacophore_role", "diagnostic_insight"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(f"Site {record['site_id']} requires {field}")
            evidence_ids = record.get("evidence_probes")
            if not isinstance(evidence_ids, list) or any(not isinstance(identity, str) for identity in evidence_ids):
                raise ValueError("evidence_probes must be a list of probe IDs")
            for identity in evidence_ids:
                if identity not in probes or probes[identity]["site_id"] != record["site_id"]:
                    raise ValueError(f"Site {record['site_id']} cites unavailable or unrelated evidence {identity}")
            guards = record.get("action_guardrails")
            if not isinstance(guards, dict) or any(not isinstance(guards.get(field), list) for field in ("allowed_actions", "forbidden_actions")):
                raise ValueError("Each site's action_guardrails needs allowed_actions and forbidden_actions lists")
            if any(not isinstance(entry, str) or not entry.strip() for field in ("allowed_actions", "forbidden_actions") for entry in guards[field]):
                raise ValueError("Action guardrails must contain nonempty text entries")
            envelopes = record.get("semantic_envelopes")
            if not isinstance(envelopes, dict) or not axes.issubset(envelopes):
                raise ValueError("Each site needs all five semantic_envelopes axes")
            for axis in axes:
                value = envelopes[axis]
                entries = value if isinstance(value, list) else [value]
                if not entries or any(not isinstance(entry, str) or not entry.strip() for entry in entries):
                    raise ValueError(f"Semantic envelope {axis} must contain nonempty text")
            if not isinstance(record.get("blacklist"), list) or any(not isinstance(entry, str) for entry in record["blacklist"]):
                raise ValueError("Each site's blacklist must be a list of text rules")
            # Do not let a model omit measured SA/QED hazards from its advice.
            for probe in probes.values():
                if probe["site_id"] != record["site_id"] or not probe.get("response_tags"):
                    continue
                directive = probe["modification_logic"].get("modification_directive", "")
                feature = directive or json.dumps(probe["modification_logic"], sort_keys=True)
                entry = f"Do not repeat the tested chemical edit: {feature} -- {probe['probe_id']} {' '.join(probe['response_tags'])}."
                if entry not in record["blacklist"]:
                    record["blacklist"].append(entry)
        guidance = manual["Combinatorial_Guidance"]
        if any(not isinstance(guidance.get(key), list) for key in ("orthogonal_sites", "mutually_exclusive_edits")):
            raise ValueError("Combinatorial_Guidance requires orthogonal_sites and mutually_exclusive_edits lists")
        for pair in guidance["orthogonal_sites"]:
            if not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(site, str) or site not in expected_ids for site in pair) or pair[0] == pair[1]:
                raise ValueError("Each orthogonal_sites entry must name two distinct known sites")
        for exclusion in guidance["mutually_exclusive_edits"]:
            if (not isinstance(exclusion, dict) or not isinstance(exclusion.get("sites"), list)
                    or not exclusion["sites"] or any(not isinstance(site, str) or site not in expected_ids for site in exclusion["sites"])
                    or not isinstance(exclusion.get("reason"), str) or not exclusion["reason"].strip()):
                raise ValueError("Each mutually_exclusive_edits entry needs known sites and a reason")
