"""One complete PROBE run, with persistent stage boundaries for inspection."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import MoleculeInput, RunConfig


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_case(case: MoleculeInput, config: RunConfig, *, client=None,
             assembler=None, evaluator=None) -> dict:
    from .agents.InteractionAgent import InteractionAgent
    from .agents.DesignAgent import DesignAgent
    from .agents.InductionAgent import InductionAgent
    from .agents.ExecutionAgent import ExecutionAgent
    from .agents.MoleculeGenerationAgent import MoleculeGenerationAgent
    from .chemistry.docking import VinaEvaluator
    from .llm import LLMClient

    case_dir = config.output_dir / case.id
    case_dir.mkdir(parents=True, exist_ok=False)
    save_json(case_dir / "input.json", asdict(case))
    state = {"id": case.id, "status": "running", "stage": "initialize"}
    try:
        client = client or LLMClient(
            config.model, base_url=config.base_url, temperature=config.temperature,
            timeout=config.timeout, trace_path=case_dir / "llm_trace.jsonl",
        )
        assembler = assembler or MoleculeGenerationAgent(config.fragment_db)
        if hasattr(assembler, "library"):
            library = assembler.library
            save_json(case_dir / "library_manifest.json", {
                "sha256": hashlib.sha256(config.fragment_db.read_bytes()).hexdigest(),
                "total_count": library.total_count,
                "usable_count": library.usable_count,
                "rejected_entries": library.rejected_entries,
            })
        evaluator = evaluator or VinaEvaluator(
            case.receptor, case.reference_ligand, case_dir / "docking",
            exhaustiveness=config.exhaustiveness, seed=config.seed, cpu=config.cpu,
            receptor_pdbqt=case.receptor_pdbqt,
        )
        state["stage"] = "diagnosis"
        diagnosis = InteractionAgent(client, evaluator).execute(case.smiles)
        save_json(case_dir / "diagnosis.json", diagnosis)
        from .selection import balanced_score
        balanced_score(diagnosis["original_metrics"], diagnosis["original_metrics"])

        state["stage"] = "probing"
        probes = DesignAgent(client, assembler, evaluator, retries=config.retries).execute(
            case.smiles, diagnosis["mapped_smiles"], diagnosis["atom_legend"],
            diagnosis["site_map"], diagnosis["strategies"], diagnosis["reports"],
        )
        save_json(case_dir / "probes.json", probes)

        state["stage"] = "induction"
        induction = InductionAgent(client).execute(
            diagnosis["original_metrics"], diagnosis["site_map"],
            diagnosis["strategies"], probes,
        )
        save_json(case_dir / "induction.json", induction)

        state["stage"] = "optimization"
        result = ExecutionAgent(client, assembler, evaluator, rounds=config.rounds).execute(
            case.smiles, diagnosis["original_metrics"], induction["EditManual"], probes,
        )
        save_json(case_dir / "result.json", result)
        state.update(status="complete", stage="complete")
        return result
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json(case_dir / "status.json", state)
