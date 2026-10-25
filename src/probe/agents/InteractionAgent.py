"""Pocket diagnosis and three-strategy planning."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, Callable

from probe.chemistry.molecules import map_smiles
from probe.chemistry.signals import extract_signals
from probe.prompts import render


SITE_TYPES = {"SYNERGY", "TENSION", "LIABILITY"}
STRATEGY_IDS = ("A", "B", "C")


def _nonempty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _brics_fragments(mapped_smiles: str) -> list[set[int]]:
    """Keep the input's one-based map numbers through BRICS decomposition."""
    from rdkit import Chem
    from rdkit.Chem import BRICS

    molecule = Chem.MolFromSmiles(mapped_smiles)
    if molecule is None:
        raise ValueError("Invalid mapped SMILES")
    bonds = [molecule.GetBondBetweenAtoms(a, b).GetIdx()
             for (a, b), _ in BRICS.FindBRICSBonds(molecule)]
    fragmented = Chem.FragmentOnBonds(molecule, bonds, addDummies=False) if bonds else molecule
    return [
        {fragmented.GetAtomWithIdx(index).GetAtomMapNum() for index in component}
        for component in Chem.GetMolFrags(fragmented)
    ]


def _validate_site_map(data: Any, fragments: list[set[int]]) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Site map must be a JSON object")
    _nonempty(data.get("holistic_profile"), "holistic_profile")
    sites = data.get("sites")
    minimum = min(3, len(fragments))
    if not isinstance(sites, list) or not minimum <= len(sites) <= min(4, len(fragments)):
        raise ValueError(f"Select {minimum} to {min(4, len(fragments))} distinct BRICS fragments")
    seen_names: set[str] = set()
    seen_fragments: set[frozenset[int]] = set()
    for position, site in enumerate(sites, 1):
        if not isinstance(site, dict):
            raise ValueError("Each site must be an object")
        if site.get("site_id") != f"S{position}":
            raise ValueError("Site IDs must be S1, S2, ... in their listed order")
        name = site.get("fragment_name")
        _nonempty(name, "fragment_name")
        if name in seen_names:
            raise ValueError("fragment_name must uniquely identify a site")
        seen_names.add(name)
        indices = site.get("atom_indices")
        if (not isinstance(indices, list) or not indices
                or any(type(index) is not int or index < 1 for index in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("atom_indices must be distinct positive integer map numbers")
        atom_set = frozenset(indices)
        if atom_set not in fragments or atom_set in seen_fragments:
            raise ValueError("Each site must be exactly one distinct contiguous BRICS fragment")
        seen_fragments.add(atom_set)
        if site.get("type") not in SITE_TYPES:
            raise ValueError("A site must have exactly one type: SYNERGY, TENSION, or LIABILITY")
        for field in ("local_symptoms", "justification", "strategic_priority"):
            _nonempty(site.get(field), field)
    return deepcopy(data)


def _validate_strategies(data: Any, site_map: dict) -> dict:
    if not isinstance(data, dict) or set(data) != set(STRATEGY_IDS):
        raise ValueError("Planner output must have exactly the strategy keys A, B, C")
    sites = {site["site_id"]: site for site in site_map["sites"]}
    liabilities = {key for key, site in sites.items() if site["type"] == "LIABILITY"}
    for key in STRATEGY_IDS:
        strategy = data[key]
        if not isinstance(strategy, dict):
            raise ValueError(f"Strategy {key} must be an object")
        for field in ("edit_prescription", "trade_off"):
            _nonempty(strategy.get(field), f"Strategy {key}.{field}")
        targeted = strategy.get("targeted_sites")
        if not isinstance(targeted, list) or not targeted:
            raise ValueError(f"Strategy {key} must target at least one declared site")
        selected: set[str] = set()
        for target in targeted:
            if not isinstance(target, dict):
                raise ValueError("Each targeted site must contain its full site reference")
            site_id = target.get("site_id")
            if site_id not in sites or site_id in selected:
                raise ValueError(f"Invalid or duplicate targeted site: {site_id!r}")
            selected.add(site_id)
            expected = sites[site_id]
            if any(target.get(field) != expected[field] for field in ("fragment_name", "type")):
                raise ValueError("Targeted site name and type must match the site map")
            indices = target.get("atom_indices")
            if (not isinstance(indices, list)
                    or any(type(index) is not int for index in indices)
                    or sorted(indices) != sorted(expected["atom_indices"])):
                raise ValueError("Targeted site atom indices must match the site map")
        if key == "B" and not liabilities <= selected:
            raise ValueError("Strategy B must target every LIABILITY site")
    return deepcopy(data)


class InteractionAgent:
    """Evaluate the input once, then diagnose sites and plan A/B/C strategies."""

    def __init__(self, client: Any, evaluator: Any):
        self.client = client
        self.evaluator = evaluator
        self.last_prompt = ""
        self.prompts: list[dict[str, str]] = []

    def _request(self, stage: str, prompt: str, validate: Callable[[Any], dict]) -> dict:
        feedback = ""
        for attempt in range(3):
            self.last_prompt = prompt + feedback
            self.prompts.append({"stage": stage, "prompt": self.last_prompt})
            try:
                return validate(self.client.json(self.last_prompt))
            except (ValueError, TypeError, KeyError) as error:
                feedback = "\n\n[Output validation error; correct the JSON]\n" + str(error)
                if attempt == 2:
                    raise ValueError(f"{stage} failed validation after 3 attempts: {error}") from error
        raise AssertionError("Unreachable")

    def execute(self, smiles: str) -> dict:
        self.prompts = []
        mapped_smiles, atom_legend = map_smiles(smiles)
        metrics = self.evaluator.evaluate(smiles)
        for name in ("vina_score", "qed", "sa"):
            if name not in metrics or not math.isfinite(float(metrics[name])):
                raise ValueError(f"Input evaluation did not return a finite {name}")
        if not metrics.get("pose_path"):
            raise ValueError("Input evaluation did not return a docked pose")
        reports = extract_signals(
            self.evaluator.receptor_pdb, metrics["pose_path"], smiles, metrics["vina_score"]
        )
        mc = reports["mc_data"]
        values = {
            "smi": smiles,
            "mapped_smi": mapped_smiles,
            "score": metrics["vina_score"],
            "mc_data['le']:.3f": f"{mc['le']:.3f}",
            "mc_data['mw']:.1f": f"{mc['mw']:.1f}",
            "mc_data['logp']:.2f": f"{mc['logp']:.2f}",
            "mc_data['tpsa']:.1f": f"{mc['tpsa']:.1f}",
            "mc_data['chiral_centers']": mc["chiral_centers"],
            "mc_data['sa_proxy']:.1f": f"{mc['sa_proxy']:.1f}",
            "', '.join(mc_data['alerts']) if mc_data['alerts'] else 'Clean'":
                ", ".join(mc["alerts"]) if mc["alerts"] else "Clean",
            **{name: reports[name] for name in (
                "interaction_text", "clash_text", "void_text", "mismatch_text",
                "frag_inter_match", "medchem_text"
            )},
        }
        fragments = _brics_fragments(mapped_smiles)
        site_prompt = render("site_map", **values) + "\n\n" + (
            "[Machine-readable serialization of the paper's Site Map]\n"
            "Return only JSON with holistic_profile (string) and sites (array). Each site has "
            "site_id (S1, S2, ... in order), fragment_name (unique), atom_indices (integer array), "
            "local_symptoms, type, justification, and strategic_priority. Preserve all diagnostic "
            "content requested above. Atom indices are one-based map numbers. Each site must "
            "equal one complete fragment from the list below. If fewer than three fragments "
            "exist, report all available fragments instead of inventing sites.\n"
            "Available BRICS fragments: " + json.dumps([sorted(fragment) for fragment in fragments])
        )
        site_map = self._request(
            "site_map", site_prompt, lambda response: _validate_site_map(response, fragments)
        )
        planner_prompt = render("probing_planner") + "\n\n[Site Map]\n" + json.dumps(site_map) + (
            "\n\n[Machine-readable serialization of the three Strategies]\n"
            "Return only a JSON object with exactly A, B, C as keys. Each value has targeted_sites "
            "(array of objects with site_id, fragment_name, atom_indices, type, copied exactly from "
            "the Site Map), edit_prescription (string), and trade_off (string). These fields "
            "serialize V_k, pi_k, and tau_k. Follow the A/B/C objectives above."
        )
        strategies = self._request(
            "probing_planner", planner_prompt, lambda response: _validate_strategies(response, site_map)
        )
        return {
            "original_metrics": metrics,
            "mapped_smiles": mapped_smiles,
            "atom_legend": atom_legend,
            "site_map": site_map,
            "strategies": strategies,
            "reports": reports,
        }
