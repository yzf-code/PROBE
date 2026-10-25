"""Validate run configuration and input paths."""

import csv
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class MoleculeInput:
    id: str
    smiles: str
    receptor: Path
    reference_ligand: Path
    receptor_pdbqt: Path | None = None


@dataclass(frozen=True)
class RunConfig:
    fragment_db: Path
    output_dir: Path
    model: str
    rounds: int = 5
    retries: int = 3
    exhaustiveness: int = 16
    seed: int = 42
    cpu: int = 1
    temperature: float = 0.7
    base_url: str | None = None
    timeout: float = 300

    def __post_init__(self):
        for name in ("rounds", "retries", "exhaustiveness", "cpu"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 1 <= self.seed < 2**31:
            raise ValueError("seed must be between 1 and 2**31 - 1")
        if self.timeout <= 0 or not self.model.strip():
            raise ValueError("A model name and positive timeout are required")
        if not self.fragment_db.is_file():
            raise FileNotFoundError(f"Fragment library not found: {self.fragment_db}")


def load_inputs(path: Path, data_root: Path) -> list[MoleculeInput]:
    """CSV paths are relative to data_root (absolute paths also work)."""
    data_root = data_root.expanduser().resolve()
    rows = []
    seen = set()

    def resolve_file(value: str, field: str, line: int) -> Path:
        p = Path(value).expanduser()
        p = (data_root / p).resolve() if not p.is_absolute() else p.resolve()
        if not p.is_file():
            raise ValueError(f"CSV line {line}: missing {field}: {p}")
        return p

    with path.expanduser().open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "smiles", "receptor", "reference_ligand"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Input CSV requires columns: {', '.join(sorted(required))}")
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"CSV line {line}: extra columns; quote fields containing commas")
            row = {key: (value or "").strip() for key, value in row.items()}
            if any(not row[key] for key in required):
                raise ValueError(f"CSV line {line}: required field is empty")
            identity = row["id"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", identity):
                raise ValueError(f"CSV line {line}: id must be 1–80 letters, digits, '_' or '-'")
            if identity in seen:
                raise ValueError(f"Duplicate molecule id: {identity}")
            seen.add(identity)
            rows.append(MoleculeInput(
                identity, row["smiles"],
                resolve_file(row["receptor"], "receptor", line),
                resolve_file(row["reference_ligand"], "reference_ligand", line),
                resolve_file(row["receptor_pdbqt"], "receptor_pdbqt", line)
                if row.get("receptor_pdbqt") else None,
            ))
    if not rows:
        raise ValueError("Input CSV has no molecules")
    return rows
