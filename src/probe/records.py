"""Versioned run records shared by generation and independent evaluation."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def input_record(case) -> dict:
    """Snapshot input identity while the original files are still available."""
    inputs = asdict(case)
    inputs["sha256"] = {}
    for field in ("receptor", "reference_ligand", "receptor_pdbqt"):
        path = inputs.get(field)
        if path and Path(path).is_file():
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            inputs["sha256"][field] = digest.hexdigest()
    return inputs


def case_record(case, result: dict | None = None, *, error: str | None = None,
                input_metadata: dict | None = None) -> dict:
    winner = result.get("winner") if result else None
    return {
        "schema_version": 1,
        "id": case.id,
        "status": "ok" if winner and winner.get("status") == "ok" and not error else "failed",
        "input": input_metadata if input_metadata is not None else input_record(case),
        "selected": winner,
        "result_file": f"{case.id}/result.json" if result else None,
        "error": error,
    }


def append_record(path: Path, record: dict) -> None:
    """Flush one record per completed case so interrupted batches keep progress."""
    line = json.dumps(record, ensure_ascii=False, allow_nan=False, default=str)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_record_file(run_dir: Path, relative: str) -> Path:
    root = run_dir.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Recorded result_file must stay within its run directory")
    return path


def _load_case(root, identity, inputs=None):
    case_dir = resolve_record_file(root, identity)
    input_path = case_dir / "input.json"
    inputs = inputs or (_read_json(input_path) if input_path.is_file() else {"id": identity})
    result_path = case_dir / "result.json"
    status_path = case_dir / "status.json"
    status = _read_json(status_path) if status_path.is_file() else {}
    result = _read_json(result_path) if result_path.is_file() else {}
    winner = result.get("winner")
    succeeded = winner and winner.get("status") == "ok" and status.get("status") != "failed"
    return {"schema_version": 1, "id": identity,
            "status": "ok" if succeeded else "incomplete",
            "input": inputs, "selected": winner,
            "result_file": f"{identity}/result.json" if result else None,
            "error": None if succeeded else status.get("error", "No completed selected molecule")}


def load_run_records(run_dir: Path) -> tuple[list[dict], dict]:
    """Read the consolidated format or earlier per-case PROBE output folders."""
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Run directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest must be a JSON object")
    expected = manifest.get("input_ids", [])
    if (not isinstance(expected, list) or any(not isinstance(x, str) for x in expected)
            or len(set(expected)) != len(expected)):
        raise ValueError("Run manifest input_ids must be unique strings")
    manifest_inputs = manifest.get("inputs", [])
    if not isinstance(manifest_inputs, list) or any(not isinstance(item, dict) for item in manifest_inputs):
        raise ValueError("Run manifest inputs must be a list of input records")
    inputs_by_id = {item.get("id"): item for item in manifest_inputs}
    records_file = root / "results.jsonl"
    records = []
    if records_file.is_file():
        for line_number, line in enumerate(records_file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed results.jsonl line {line_number}") from exc
            if not isinstance(record, dict) or record.get("schema_version") != 1:
                raise ValueError(f"Unsupported run record at line {line_number}")
            records.append(record)
    else:
        identities = expected or sorted(p.name for p in root.iterdir()
                                        if p.is_dir() and (p / "input.json").is_file())
        for identity in identities:
            records.append(_load_case(root, identity, inputs_by_id.get(identity)))
    seen = set()
    for record in records:
        identity = record.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError(f"Missing or duplicate run-record id: {identity!r}")
        if expected and identity not in expected:
            raise ValueError(f"Run record {identity} is not listed in the manifest")
        if not isinstance(record.get("input"), dict):
            raise ValueError(f"Run record {identity} requires its input metadata")
        seen.add(identity)
    for identity in expected:
        if identity not in seen:
            # Recover the narrow crash window after result.json was saved but
            # before its consolidated JSONL row was appended.
            records.append(_load_case(root, identity, inputs_by_id.get(identity)))
    if not records:
        raise ValueError("No PROBE run records found; expected results.jsonl or per-case input/result files")
    return records, manifest
