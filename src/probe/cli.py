"""Command line interface for validation and method execution."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import sys

from . import __version__
from .config import RunConfig, load_inputs


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PROBE: probing-guided molecular optimization")
    p.add_argument("--version", action="version", version=f"PROBE {__version__}")
    commands = p.add_subparsers(dest="command", required=True)
    for command in ("check", "run"):
        sub = commands.add_parser(command, help=("Check inputs and dependencies without API calls"
                                                if command == "check" else "Run the PROBE method"))
        sub.add_argument("--input", type=Path, required=True, help="CSV with explicit receptor/ligand paths")
        sub.add_argument("--data-root", type=Path, default=Path("."), help="Root for paths inside the CSV")
        sub.add_argument("--fragment-db", type=Path, required=True, help="Annotated fragment JSON library")
        if command == "run":
            sub.add_argument("--output-dir", type=Path, required=True, help="A new directory for this run")
            sub.add_argument("--model", default=os.environ.get("OPENAI_MODEL"),
                             help="Model/deployment ID (or set OPENAI_MODEL)")
            sub.add_argument("--base-url", default=None, help="Optional OpenAI-compatible API endpoint")
            sub.add_argument("--rounds", type=int, default=5)
            sub.add_argument("--retries", type=int, default=3, help="Maximum probe design attempts")
            sub.add_argument("--exhaustiveness", type=int, default=16)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--cpu", type=int, default=1, help="Vina CPUs per molecule")
            sub.add_argument("--temperature", type=float, default=0.7)
            sub.add_argument("--timeout", type=float, default=300, help="LLM request timeout in seconds")
    from .evaluation.cli import add_commands
    add_commands(commands)
    return p


def _preflight(cases, library: Path) -> None:
    from .chemistry.molecules import molecule
    from .agents.MoleculeGenerationAgent import MoleculeGenerationAgent

    for case in cases:
        molecule(case.smiles)
    MoleculeGenerationAgent(library)
    for module in ("openai", "vina", "meeko", "openbabel", "plip", "gemmi"):
        if importlib.util.find_spec(module) is None:
            raise ValueError(f"Missing dependency {module}; install the documented environment")
    if any(case.receptor_pdbqt is None for case in cases):
        if shutil.which("mk_prepare_receptor.py") is None:
            raise ValueError("Install Meeko's mk_prepare_receptor.py or provide receptor_pdbqt in the CSV")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.command in ("evaluate", "rank"):
            from .evaluation.cli import dispatch
            return dispatch(args)
        cases = load_inputs(args.input, args.data_root)
        args.fragment_db = args.fragment_db.expanduser().resolve()
        _preflight(cases, args.fragment_db)
        if args.command == "check":
            print(f"Validated {len(cases)} input molecule(s), fragment library, and dependencies. "
                  "No LLM calls or docking jobs were run.")
            return 0
        if not args.model:
            raise ValueError("Specify --model or set OPENAI_MODEL")
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY")
        config = RunConfig(
            args.fragment_db.expanduser().resolve(), args.output_dir.expanduser().resolve(),
            args.model, args.rounds, args.retries, args.exhaustiveness,
            args.seed, args.cpu, args.temperature, args.base_url, args.timeout,
        )
        # Refuse to mix a new run with an existing run or overwrite its records.
        config.output_dir.mkdir(parents=True, exist_ok=False)
        from .pipeline import run_case, save_json
        from .records import case_record, append_record, input_record
        versions = {}
        for package in ("rdkit", "numpy", "openai", "vina", "meeko", "plip"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unknown"
        input_snapshots = {case.id: input_record(case) for case in cases}
        save_json(config.output_dir / "manifest.json", {
            "version": __version__, "created_at": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config), "dependencies": versions,
            "input_ids": [case.id for case in cases],
            "inputs": list(input_snapshots.values()),
            "record_schema_version": 1,
            "results_file": "results.jsonl",
        })
        (config.output_dir / "results.jsonl").touch()
        # Also protect custom output-directory names from accidental commits.
        (config.output_dir / ".gitignore").write_text("*\n", encoding="utf-8")
        failed = []
        for case in cases:
            try:
                result = run_case(case, config)
                record = case_record(case, result, input_metadata=input_snapshots[case.id])
                if record["status"] != "ok":
                    failed.append(case.id)
            except Exception as exc:
                logging.error("%s failed: %s", case.id, exc)
                failed.append(case.id)
                record = case_record(case, error=f"{type(exc).__name__}: {exc}", input_metadata=input_snapshots[case.id])
            append_record(config.output_dir / "results.jsonl", record)
        print(f"Finished: {len(cases) - len(failed)}/{len(cases)} complete. "
              f"Run records: {config.output_dir}")
        if failed:
            print("Failed IDs: " + ", ".join(failed), file=sys.stderr)
        return 1 if failed else 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"PROBE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
