"""Realize the three-strategy, four-tier probing matrix."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import re
from typing import Any

from probe.prompts import render


TIERS = ("High", "Medium", "Low", "Counterfactual")
STRATEGIES = ("A", "B", "C")
ACTIONS = {"Grow", "Replace_Terminal_Group", "Replace_Sidechain_or_Ring", "Delete_Group"}
CONSTRAINTS = {
    "size": {"Small", "Medium", "Large", "Any"},
    "polarity": {"Hydrophilic", "Neutral", "Lipophilic", "Any"},
    "flexibility": {"Rigid", "Flexible", "Any"},
    "shape": {"Disc-like", "Rod-like", "Sphere-like", "Any"},
    "charge": {0, 1, -1, "Any"},
}
HEAVY_ATOM_BOUNDS = {
    "A": {"High": (4, None), "Medium": (1, 3), "Low": (0, 1), "Counterfactual": (None, -1)},
    "B": {"High": (None, -4), "Medium": (-3, -1), "Low": (-1, -1), "Counterfactual": (3, None)},
    "C": {"High": (-2, 2), "Medium": (-1, 1), "Low": (0, 0), "Counterfactual": (None, None)},
}


def _canonical_and_heavy(smiles: str) -> tuple[str, int]:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Assembler returned invalid SMILES")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True), molecule.GetNumHeavyAtoms()


def _validate_heavy_delta(strategy: str, tier: str, delta: int) -> None:
    lower, upper = HEAVY_ATOM_BOUNDS[strategy][tier]
    if (lower is not None and delta < lower) or (upper is not None and delta > upper):
        raise ValueError(
            f"Strategy {strategy} {tier}: delta_heavy_atoms={delta} violates the paper's "
            f"explicit interval [{lower if lower is not None else '-inf'}, "
            f"{upper if upper is not None else '+inf'}]"
        )


class DesignAgent:
    """Keep all 12 slots and their metadata, including explicitly failed probes."""

    def __init__(self, client: Any, assembler: Any, evaluator: Any, retries: int = 3):
        if type(retries) is not int or retries < 1:
            raise ValueError("retries must be a positive integer (total attempts per strategy)")
        self.client = client
        self.assembler = assembler
        self.evaluator = evaluator
        self.retries = retries
        self.last_prompt = ""
        self.prompts: list[dict[str, str]] = []

    @staticmethod
    def _sites_for_strategy(strategy: dict, site_map: dict, atom_maps: set[int]) -> dict:
        sites = {site["fragment_name"]: site for site in site_map["sites"]}
        if len(sites) != len(site_map["sites"]):
            raise ValueError("The site map contains ambiguous fragment names")
        targets = strategy.get("targeted_sites")
        if not isinstance(targets, list) or not targets:
            raise ValueError("A strategy must declare its targeted_sites")
        allowed = {}
        for target in targets:
            if not isinstance(target, dict) or target.get("fragment_name") not in sites:
                raise ValueError("Strategy references an undeclared target site")
            name = target["fragment_name"]
            site = sites[name]
            indices = target.get("atom_indices")
            if (target.get("site_id") != site["site_id"]
                    or target.get("type") != site["type"]
                    or not isinstance(indices, list)
                    or any(type(index) is not int for index in indices)
                    or sorted(indices) != sorted(site["atom_indices"])
                    or not set(indices) <= atom_maps):
                raise ValueError("Strategy target does not match its declared site or input atom maps")
            if name in allowed:
                raise ValueError("A strategy contains duplicate targeted sites")
            allowed[name] = site
        return allowed

    @staticmethod
    def _validate_probe(probe: Any, key: str, tier: str, sites: dict) -> dict:
        if not isinstance(probe, dict):
            raise ValueError("Each probe must be an object")
        fields = {"probe_id", "intensity_tier", "target_site", "expected_signal", "modification_logic"}
        if not fields <= set(probe) or set(probe) - fields - {"strategy_id"}:
            raise ValueError("Probe fields must follow the paper's design schema")
        number = TIERS.index(tier) + 1
        if probe.get("probe_id") != f"Probe_{key}_{number:02d}":
            raise ValueError(f"The {tier} slot must use probe_id Probe_{key}_{number:02d}")
        if probe.get("intensity_tier") != tier:
            raise ValueError("Probe tier does not match its matrix slot")
        if probe.get("strategy_id", key) != key:
            raise ValueError("Probe strategy_id does not match its matrix row")
        site_name = probe.get("target_site")
        if not isinstance(site_name, str) or site_name not in sites:
            raise ValueError("target_site must name a site declared in this strategy's V_k")
        if not isinstance(probe.get("expected_signal"), str) or not probe["expected_signal"].strip():
            raise ValueError("expected_signal is required")
        logic = probe.get("modification_logic")
        if not isinstance(logic, dict) or set(logic) != {"action", "target_atom_indices", "constraints"}:
            raise ValueError("modification_logic requires exactly action, target_atom_indices, constraints")
        if logic["action"] not in ACTIONS:
            raise ValueError("Unknown editing action")
        anchors = logic["target_atom_indices"]
        if (not isinstance(anchors, list) or len(anchors) != 1
                or type(anchors[0]) is not int or anchors[0] < 1):
            raise ValueError("Each probe must declare exactly one positive integer anchor")
        if anchors[0] not in sites[site_name]["atom_indices"]:
            raise ValueError("Probe anchor lies outside the declared target site")
        constraints = logic["constraints"]
        if not isinstance(constraints, dict) or set(constraints) != set(CONSTRAINTS):
            raise ValueError("All five semantic fragment constraints are required")
        for name, values in CONSTRAINTS.items():
            value = constraints[name]
            if type(value) not in (str, int) or value not in values:
                raise ValueError(f"Invalid semantic constraint {name}: {value!r}")
        normalized = deepcopy(probe)
        normalized["strategy_id"] = key
        normalized["site_id"] = sites[site_name]["site_id"]
        return normalized

    @staticmethod
    def _fail(record: dict, error: Exception | str, attempt: int) -> None:
        record["status"] = "failed"
        record["error"] = str(error)
        record["attempts"] = attempt
        record["attempt_errors"].append({"attempt": attempt, "error": str(error)})

    def execute(self, smiles: str, mapped_smiles: str, atom_legend: str,
                site_map: dict, strategies: dict, reports: dict) -> list[dict]:
        self.prompts = []
        atom_maps = {int(value) for value in re.findall(r":(\d+)\]", mapped_smiles)}
        if not atom_maps or 0 in atom_maps:
            raise ValueError("Design requires one-based atom-mapped SMILES")
        shared = render("shared_physics_context", **{
            "interaction_text": reports["interaction_text"],
            "steric_overlap_text": reports["clash_text"],
            "void_text": reports["void_text"],
            "mismatch_text": reports["mismatch_text"],
            "frag_inter_match": reports["frag_inter_match"],
            "medchem_text": reports["medchem_text"],
        })
        canonical_original, original_heavy = _canonical_and_heavy(smiles)
        used_smiles = {canonical_original}
        matrix = []
        for key in STRATEGIES:
            records = {
                tier: {
                    "probe_id": f"Probe_{key}_{number:02d}",
                    "strategy_id": key,
                    "intensity_tier": tier,
                    "target_site": None,
                    "site_id": None,
                    "expected_signal": None,
                    "modification_logic": None,
                    "smiles": None,
                    "metrics": None,
                    "status": "failed",
                    "error": "No valid proposal received",
                    "attempts": 0,
                    "attempt_errors": [],
                    "evaluation_attempted": False,
                } for number, tier in enumerate(TIERS, 1)
            }
            matrix.extend(records.values())
            try:
                strategy = strategies[key]
                allowed = self._sites_for_strategy(strategy, site_map, atom_maps)
            except (KeyError, TypeError, ValueError) as error:
                for record in records.values():
                    self._fail(record, f"Invalid strategy {key}: {error}", 0)
                continue
            base_prompt = render(
                "design_prompt", mapped_smi=mapped_smiles, atom_legend=atom_legend,
                shared_physics_context=shared, stg_key=key,
                stg_content=json.dumps(strategy), tactic_instructions=render(f"tactic_{key}"),
            )
            base_prompt += (
                "\n\n[Stable matrix identifiers]\n"
                "Use intensity_tier exactly High, Medium, Low, Counterfactual; respectively use "
                f"probe_id Probe_{key}_01, Probe_{key}_02, Probe_{key}_03, Probe_{key}_04. "
                "Retain expected_signal and modification_logic in every probe. "
                "Use only the target sites and atom maps declared above."
            )
            for attempt in range(1, self.retries + 1):
                pending = [tier for tier, record in records.items()
                           if record["status"] != "ok" and not record["evaluation_attempted"]]
                if not pending:
                    break
                feedback = ""
                if attempt > 1:
                    feedback = (
                        "\n\n[Repair failed matrix slots]\n"
                        "Return the full four-probe JSON row. Successful slots are already fixed "
                        "and will not be regenerated. Slots with evaluation_attempted=true are "
                        "also final, including failed evaluations, to retain the 12-probe budget. "
                        "Repair only the remaining failed slots, preserving "
                        "their strategy, tier, target_site, and scientific expected_signal; adjust "
                        "only the edit implementation or fragment constraints.\n"
                        + json.dumps(list(records.values()), default=str)
                    )
                self.last_prompt = base_prompt + feedback
                self.prompts.append({"strategy_id": key, "prompt": self.last_prompt})
                try:
                    response = self.client.json(self.last_prompt)
                    if not isinstance(response, dict) or response.get("strategy_id") != key:
                        raise ValueError(f"Response must declare strategy_id={key}")
                    proposals = response.get("probes")
                    if not isinstance(proposals, list) or len(proposals) > 4:
                        raise ValueError("probes must be an array containing at most four matrix slots")
                except Exception as error:
                    for tier in pending:
                        self._fail(records[tier], error, attempt)
                    continue
                for tier in pending:
                    record = records[tier]
                    candidates = [proposal for proposal in proposals
                                  if isinstance(proposal, dict) and proposal.get("intensity_tier") == tier]
                    try:
                        if len(candidates) != 1:
                            raise ValueError(f"Expected exactly one {tier} probe; received {len(candidates)}")
                        proposal = self._validate_probe(candidates[0], key, tier, allowed)
                        if record["target_site"] is not None:
                            if proposal["target_site"] != record["target_site"]:
                                raise ValueError("A repair cannot change the original target_site")
                            if proposal["expected_signal"] != record["expected_signal"]:
                                raise ValueError("A repair cannot change the original expected_signal")
                        # Copy every validated scientific field before assembly/evaluation.
                        record.update(proposal)
                        record["attempts"] = attempt
                        result_smiles, message = self.assembler.execute_editing(
                            mapped_smiles, deepcopy(record["modification_logic"]), set(used_smiles)
                        )
                        if not result_smiles:
                            raise ValueError(f"Fragment assembly failed: {message}")
                        result_smiles, heavy_atoms = _canonical_and_heavy(result_smiles)
                        if result_smiles in used_smiles:
                            raise ValueError("Assembler returned an unchanged or duplicate molecule")
                        record["smiles"] = result_smiles
                        record["delta_heavy_atoms"] = heavy_atoms - original_heavy
                        _validate_heavy_delta(key, tier, record["delta_heavy_atoms"])
                        # A failed docking still consumes this slot's one evaluation.
                        record["evaluation_attempted"] = True
                        metrics = self.evaluator.evaluate(result_smiles)
                        if not isinstance(metrics, dict) or any(
                            name not in metrics or not math.isfinite(float(metrics[name]))
                            for name in ("vina_score", "qed", "sa")
                        ):
                            raise ValueError("Probe evaluation must provide finite vina_score, qed, and sa")
                        record.update(metrics=deepcopy(metrics), status="ok", error=None)
                        used_smiles.add(result_smiles)
                    except Exception as error:
                        self._fail(record, error, attempt)
        return matrix
