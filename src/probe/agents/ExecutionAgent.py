"""Bilateral draft/review/refine DMTA with a separate Co-optimization agent."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import re
from typing import Any

from probe.chemistry.molecules import canonical_smiles, map_smiles
from probe.prompts import render
from probe.responses import ResponseThresholds, normalize_probe
from probe.selection import balanced_score, select_winner, validated_metrics

logger = logging.getLogger(__name__)

_ACTIONS = {"Grow", "Replace_Terminal_Group", "Replace_Sidechain_or_Ring", "Delete_Group"}
_AXES = {
    "size": {"Small", "Medium", "Large", "Any"},
    "polarity": {"Hydrophilic", "Neutral", "Lipophilic", "Any"},
    "flexibility": {"Rigid", "Flexible", "Any"},
    "shape": {"Disc-like", "Rod-like", "Sphere-like", "Any"},
    "charge": {"0", "+1", "-1", "Any"},
}
_FOCUS = {"Affinity": "lower Vina without degrading QED or SA",
          "Druggability": "raise QED+SA without sacrificing affinity"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


class ExecutionAgent:
    def __init__(self, client, assembler, evaluator, rounds=5, thresholds: ResponseThresholds | None = None):
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
            raise ValueError("rounds must be a nonnegative integer")
        self.client = client
        self.assembler = assembler
        self.evaluator = evaluator
        self.rounds = rounds
        self.thresholds = thresholds or ResponseThresholds()

    def execute(self, original_smiles, original_metrics, edit_manual, probe_results) -> dict[str, Any]:
        """Optimize a molecule using the site manual and evaluated probes."""
        original_smiles = canonical_smiles(original_smiles)
        original_metrics = validated_metrics(original_metrics)
        balanced_score(original_metrics, original_metrics)  # Validate denominators before any model call.
        manual = deepcopy(edit_manual.get("EditManual", edit_manual))
        records = manual.get("Per_Site_Records")
        if not isinstance(records, list) or not records:
            raise ValueError("Execution requires a nonempty EditManual.Per_Site_Records list")
        site_ids = [record.get("site_id") for record in records]
        if any(not isinstance(identity, str) or not identity for identity in site_ids) or len(set(site_ids)) != len(site_ids):
            raise ValueError("Execution requires unique nonempty site_id values")
        initial = {"candidate_id": "initial", "source": "initial", "round": 0,
                   "status": "ok", "smiles": original_smiles,
                   "metrics": original_metrics, "balanced_score": 0.0}
        all_candidates = [initial]
        used_smiles = {original_smiles}
        for raw in probe_results:
            probe = normalize_probe(raw, original_metrics)
            probe.update(candidate_id=probe["probe_id"], source="probe", round=0)
            if probe["status"] == "ok":
                try:
                    probe["smiles"] = canonical_smiles(probe["smiles"])
                    probe["balanced_score"] = balanced_score(probe["metrics"], original_metrics)
                    used_smiles.add(probe["smiles"])
                except (ValueError, TypeError) as exc:
                    probe.update(status="failed", error=f"Invalid probe candidate: {exc}")
            all_candidates.append(probe)
        history = []
        current = initial
        directives = {"Affinity": "Choose a manual-supported site to improve affinity while preserving QED and SA.",
                      "Druggability": "Choose a manual-supported liability site to improve QED+SA while preserving affinity."}
        for round_number in range(1, self.rounds + 1):
            mapped, legend = map_smiles(current["smiles"])
            round_record = {
                "round": round_number, "parent_smiles": current["smiles"],
                "parent_metrics": deepcopy(current["metrics"]),
                "drafts": {}, "reviews": {}, "designs": {}, "errors": [],
            }
            shared_context = render(
                "shared_context", mapped_smi=mapped, atom_legend=legend,
                site_resolution_note=render("site_resolution_note"),
                edit_manual_json=_json(manual), global_heuristics=render("global_heuristics"),
                site_history=_json({"prior_rounds": history, "agent_directives": directives}),
            )
            identity_note = (
                "\n[Machine-readable site identity]\nInclude top-level site_id with exactly one value from "
                + _json(site_ids)
                + ". Keep the explanation in analysis.target_site_mapping. Atom indices are the CURRENT "
                "positive atom-map labels in the supplied dictionary; [0] in the example is only a "
                "placeholder and is invalid. Do not reuse initial-ligand atom indices."
            )
            designs = self._generate_designs(shared_context, identity_note, mapped, manual, round_record)
            round_candidates = []
            for role in ("Affinity", "Druggability", "CoOpt"):
                candidate = self._execute_design(
                    designs.get(role), role, round_number, mapped, used_smiles,
                    current["metrics"], original_metrics, round_record,
                )
                round_candidates.append(candidate)
                all_candidates.append(candidate)
                self._update_manual(manual, candidate)
            round_record["candidates"] = deepcopy(round_candidates)
            successful = [candidate for candidate in round_candidates if candidate["status"] == "ok"]
            if successful:
                current = select_winner(successful, original_metrics)
                round_record["selected_candidate_id"] = current["candidate_id"]
            else:
                round_record["selected_candidate_id"] = current["candidate_id"]
                round_record["selection_note"] = "All three attempts failed; retain the current parent."
            directives = self._next_directives(round_candidates)
            round_record["agent_directives"] = directives.copy()
            history.append(round_record)
        winner = select_winner(all_candidates, original_metrics)
        return {"all_candidates": all_candidates, "winner": winner, "history": history, "edit_manual": manual}

    def _generate_designs(self, shared, identity_note, mapped, manual, audit):
        drafts, reviews, refined = {}, {}, {}
        for role in ("Affinity", "Druggability"):
            prompt = render(f"prompt_draft_{role.lower()}", shared_context=shared) + identity_note
            drafts[role] = self._request_design(prompt, role, f"Draft_{role}", mapped, manual, audit, "draft")
        audit["drafts"] = deepcopy(drafts)
        for reviewer, opposing in (("Affinity", "Druggability"), ("Druggability", "Affinity")):
            if drafts[opposing] is None:
                self._record_error(audit, f"review_by_{reviewer}", f"Skipped because {opposing} draft is unavailable")
                reviews[reviewer] = None
                continue
            prompt = render("base_cross_review_prompt", reviewer_role=f"{reviewer} agent",
                            opposing_role=f"{opposing} agent", opposing_draft=_json(drafts[opposing]),
                            shared_context=shared, reviewer_focus=_FOCUS[reviewer])
            try:
                review = self.client.complete(prompt)
                if not isinstance(review, str) or not review.strip():
                    raise ValueError("Cross-review returned no text")
                reviews[reviewer] = review
            except Exception as exc:
                reviews[reviewer] = None
                self._record_error(audit, f"review_by_{reviewer}", str(exc))
        audit["reviews"] = deepcopy(reviews)
        for role, opposing in (("Affinity", "Druggability"), ("Druggability", "Affinity")):
            if drafts[role] is None or reviews.get(opposing) is None:
                refined[role] = None
                self._record_error(audit, f"refine_{role}", "Skipped because a draft or its opposing cross-review is unavailable")
                continue
            prompt = render("base_refine_prompt", self_role=f"{role} agent", self_tag=role,
                            shared_context=shared, your_draft=_json(drafts[role]),
                            opposing_role=f"{opposing} agent", opposing_review=reviews[opposing],
                            self_focus=_FOCUS[role]) + identity_note
            refined[role] = self._request_design(prompt, role, f"Refined_{role}", mapped, manual, audit, "refine")
        if all(refined.get(role) is not None for role in ("Affinity", "Druggability")):
            affinity, druggability = _json(refined["Affinity"]), _json(refined["Druggability"])
            prompt = render("prompt_co_optimization", shared_context=shared,
                            refined_affinity=affinity, refined_druggability=druggability,
                            **{'refined_affinity if refined_affinity else "N/A"': affinity,
                               'refined_druggability if refined_druggability else "N/A"': druggability}) + identity_note
            refined["CoOpt"] = self._request_design(prompt, "CoOpt", "FinalDesign_CoOpt", mapped, manual, audit, "co_optimization")
        else:
            refined["CoOpt"] = None
            self._record_error(audit, "co_optimization", "Skipped because both refined drafts are required")
        audit["designs"] = deepcopy(refined)
        return refined

    def _request_design(self, prompt, role, design_id, mapped, manual, audit, phase):
        try:
            raw = self.client.json(prompt)
            design = self._validate_design(raw, mapped, manual, role)
            design["design_id" if phase != "draft" else "draft_id"] = design_id
            return design
        except Exception as exc:
            self._record_error(audit, f"{phase}_{role}", str(exc))
            return None

    @staticmethod
    def _validate_design(raw, mapped, manual, role):
        if not isinstance(raw, dict):
            raise ValueError("Design must be a JSON object")
        design = deepcopy(raw)
        logic = design.get("modification_logic")
        if not isinstance(logic, dict) or logic.get("action") not in _ACTIONS:
            raise ValueError("Design requires a recognized modification_logic.action")
        anchors = logic.get("target_atom_indices")
        current_labels = {int(label) for label in re.findall(r":(\d+)\]", mapped)}
        if not isinstance(anchors, list) or len(anchors) != 1 or type(anchors[0]) is not int or anchors[0] not in current_labels:
            raise ValueError("Design must target exactly one current, positive atom-map label")
        constraints = logic.get("constraints")
        if not isinstance(constraints, dict):
            raise ValueError("Design requires five semantic constraints")
        for axis, allowed in _AXES.items():
            value = constraints.get(axis)
            if axis == "charge" and type(value) is int:
                value = "+1" if value == 1 else str(value)
                constraints[axis] = value
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"Invalid semantic constraint {axis}: {value!r}")
        if role == "Druggability" and constraints["size"] == "Large" and constraints["flexibility"] == "Flexible":
            raise ValueError("Druggability edits cannot combine Large and Flexible")
        if not isinstance(logic.get("modification_directive"), str) or not logic["modification_directive"].strip():
            raise ValueError("Design requires a chemically explicit modification_directive")
        if not isinstance(logic.get("pharmacophore_smarts", ""), str):
            raise ValueError("pharmacophore_smarts must be a string")
        logic.setdefault("pharmacophore_smarts", "")
        analysis = design.get("analysis")
        if not isinstance(analysis, dict) or not isinstance(analysis.get("target_site_mapping"), str):
            raise ValueError("Design requires analysis.target_site_mapping")
        records = {record["site_id"]: record for record in manual["Per_Site_Records"]}
        identity = design.get("site_id")
        if identity is None:
            text = analysis["target_site_mapping"]
            matches = [site for site in records if re.search(r"(?<!\w)" + re.escape(site) + r"(?!\w)", text)]
            if len(matches) == 1:
                identity = matches[0]
        if identity not in records:
            raise ValueError("Design site_id must resolve to one known abstract site")
        design["site_id"] = identity
        record = records[identity]
        guards = record.get("action_guardrails", {})
        action = logic["action"]
        def listed_action(entry):
            return next((name for name in _ACTIONS if isinstance(entry, str) and re.match(r"^" + re.escape(name) + r"(?:\b|:)", entry.strip())), None)
        forbidden = {listed_action(entry) for entry in guards.get("forbidden_actions", [])}
        allowed = {listed_action(entry) for entry in guards.get("allowed_actions", [])} - {None}
        if action in forbidden or (allowed and action not in allowed):
            raise ValueError(f"{action} violates action guardrails at {identity}")
        for axis, envelope in record.get("semantic_envelopes", {}).items():
            if axis not in _AXES:
                continue
            values = envelope if isinstance(envelope, list) else [envelope]
            if values and all(isinstance(value, str) and value in _AXES[axis] for value in values):
                if "Any" not in values and constraints[axis] not in values:
                    raise ValueError(f"{axis} violates the semantic envelope at {identity}")
        return design

    def _execute_design(self, design, role, round_number, mapped, used_smiles, parent_metrics, original_metrics, audit):
        record = {"candidate_id": f"round_{round_number:02d}_{role.lower()}", "source": "optimization",
                  "round": round_number, "role": role, "status": "failed", "smiles": None, "metrics": None}
        if design is None:
            record["error"] = "No validated refined design was available"
            return record
        record.update(design_id=design["design_id"], site_id=design["site_id"],
                      modification_logic=deepcopy(design["modification_logic"]), analysis=deepcopy(design["analysis"]))
        try:
            # Give assembly a copy: unsuccessful assembly must not mutate the
            # global set and prevent subsequent independent candidates.
            smiles, message = self.assembler.execute_editing(mapped, design["modification_logic"], set(used_smiles))
            record["assembly_message"] = str(message)
            if not smiles:
                raise ValueError(f"Fragment assembly failed: {message}")
            smiles = canonical_smiles(smiles)
            if smiles in used_smiles:
                raise ValueError("Fragment assembly returned an already evaluated molecule")
            record["smiles"] = smiles
            # A failed evaluation is not silently repeated in a later round.
            used_smiles.add(smiles)
            metrics = validated_metrics(self.evaluator.evaluate(smiles))
            record.update(metrics=metrics, status="ok", balanced_score=balanced_score(metrics, original_metrics))
            delta = {"d_vina": metrics["vina_score"] - parent_metrics["vina_score"],
                     "d_qed": metrics["qed"] - parent_metrics["qed"],
                     "d_sa": metrics["sa"] - parent_metrics["sa"]}
            delta["d_a"] = -delta["d_vina"]
            delta["d_d"] = delta["d_qed"] + delta["d_sa"]
            record["delta_from_parent"] = delta
            record["intent"] = {"affinity_improved": delta["d_a"] > 0,
                                "druggability_improved": delta["d_d"] > 0,
                                "declared_by": role}
            if role == "Affinity":
                success = delta["d_a"] > 0 and delta["d_qed"] >= 0 and delta["d_sa"] >= 0
            elif role == "Druggability":
                success = delta["d_d"] > 0 and delta["d_a"] >= 0
            else:
                success = delta["d_a"] > 0 and delta["d_d"] > 0
            record["intent"]["success"] = success
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            self._record_error(audit, f"make_test_{role}", str(exc))
        return record

    def _update_manual(self, manual, candidate):
        identity = candidate.get("site_id")
        if identity is None:
            return
        record = next(record for record in manual["Per_Site_Records"] if record["site_id"] == identity)
        observation = {key: deepcopy(candidate[key]) for key in ("candidate_id", "role", "modification_logic", "status", "delta_from_parent", "intent", "error") if key in candidate}
        record.setdefault("observed_edits", []).append(observation)
        delta = candidate.get("delta_from_parent", {})
        reasons = []
        if candidate["status"] != "ok":
            # Infrastructure failures are not evidence about chemistry.
            observation["interpretation"] = "No measured response; do not infer a chemical prohibition from this failure."
        else:
            if delta["d_sa"] < -self.thresholds.sa_liability_drop:
                reasons.append("SA-LIABILITY")
            if delta["d_qed"] < -self.thresholds.qed_drop:
                reasons.append("QED-DROP")
            if not candidate["intent"]["success"]:
                observation["interpretation"] = "Declared intent failed under these tested constraints; reconsider this exact edit, not every action at the site."
                reasons.append("failed declared intent")
        if reasons:
            logic = candidate["modification_logic"]
            ban = (f"Do not repeat exact tested edit {logic['action']} with {json.dumps(logic['constraints'], sort_keys=True)}: "
                   f"{logic['modification_directive']} -- {candidate['candidate_id']} ({', '.join(reasons)}).")
            blacklist = record.setdefault("blacklist", [])
            if ban not in blacklist:
                blacklist.append(ban)

    @staticmethod
    def _next_directives(candidates):
        result = {}
        for role in ("Affinity", "Druggability"):
            candidate = next(candidate for candidate in candidates if candidate["role"] == role)
            if candidate["status"] != "ok":
                result[role] = (f"Previous {role} attempt {candidate['candidate_id']} failed: {candidate.get('error')}. "
                                "Recheck site-to-current-atom mapping and choose a feasible, untested edit within the manual; no score was observed.")
            elif candidate["intent"]["success"]:
                result[role] = (f"Extend the successful direction at {candidate['site_id']} using a new localized edit within the manual; "
                                f"{candidate['candidate_id']} measured {_json(candidate['delta_from_parent'])}. Resolve its descendant atoms anew.")
            else:
                result[role] = (f"Repair the failed intent at {candidate['site_id']}: {candidate['candidate_id']} measured "
                                f"{_json(candidate['delta_from_parent'])}. Avoid the exact tested edit; tighten the relevant "
                                "size/polarity/flexibility constraints or choose another supported site, while protecting the other objective.")
        return result

    @staticmethod
    def _record_error(audit, phase, error):
        audit["errors"].append({"phase": phase, "error": error})
        logger.warning("DMTA round %s, %s: %s", audit["round"], phase, error)
