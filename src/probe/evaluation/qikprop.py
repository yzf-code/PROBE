"""Optional licensed QikProp evaluation, or strict import of an existing CSV."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


QIKPROP_REGION = {
    "#stars": (0, 5),
    "#amine": (0, 1),
    "#amidine": (0, 0),
    "#acid": (0, 1),
    "#amide": (0, 1),
    "#rotor": (0, 15),
    "#rtvFG": (0, 2),
    "mol_MW": (130.0, 725.0),
    "dipole": (1.0, 12.5),
    "SASA": (300.0, 1000.0),
    "FOSA": (0.0, 750.0),
    "FISA": (7.0, 330.0),
    "PISA": (0.0, 450.0),
    "WPSA": (0.0, 175.0),
    "volume": (500.0, 2000.0),
    "donorHB": (0.0, 6.0),
    "accptHB": (2.0, 20.0),
    "dip^2/V": (0.0, 0.13),
    "ACxDN^.5/SA": (0.0, 0.05),
    "glob": (0.75, 0.95),
    "QPpolrz": (13.0, 70.0),
    "QPlogPC16": (4.0, 18.0),
    "QPlogPoct": (8.0, 35.0),
    "QPlogPw": (4.0, 45.0),
    "QPlogPo/w": (-2.0, 6.5),
    "QPlogS": (-6.5, 0.5),
    "CIQPlogS": (-6.5, 0.5),
    "QPPCaco": (25, math.inf),
    "QPlogBB": (-3.0, 1.2),
    "QPPMDCK": (25, math.inf),
    "QPlogKp": (-8.0, -1.0),
    "IP(eV)": (7.9, 10.5),
    "EA(eV)": (-0.9, 1.7),
    "#metab": (1, 8),
    "QPlogKhsa": (-1.5, 1.5),
    "PercentHumanOralAbsorption": (25, 100),
    "SAFluorine": (0.0, 100.0),
    "SAamideO": (0.0, 35.0),
    "PSA": (7.0, 200.0),
    "#NandO": (2, 15),
    "RuleOfThree": (0, 3),
}

# These are alternate export spellings, not a change to the pass criterion.
_PROPERTY_NAMES = {name.casefold(): name for name in QIKPROP_REGION}
_PROPERTY_NAMES.update({
    "qplogpo_w": "QPlogPo/w",
    "dip^2_v": "dip^2/V",
    "acxdn^.5_sa": "ACxDN^.5/SA",
})
_TITLE_NAMES = {"molecule", "title", "s_m_title"}


def _ids(values: Iterable[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError("expected_ids must be a collection of molecule IDs")
    identities = list(values)
    if any(not isinstance(value, str) or not value.strip() for value in identities):
        raise ValueError("Every molecule ID must be a nonempty string")
    if len(set(identities)) != len(identities):
        raise ValueError("Molecule IDs must be unique")
    return identities


def _unavailable(status: str, error: str) -> dict[str, Any]:
    return {"status": status, "passed": None, "properties": {}, "error": error,
            "complete": False, "variants": [], "variants_observed": 0,
            "variants_complete": 0, "warnings": []}


def _read_csv(path: Path) -> tuple[list[str], list[tuple[int, dict[str, str]]]]:
    with path.expanduser().open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        if not reader.fieldnames:
            raise ValueError("QikProp CSV has no header")
        raw_headers = reader.fieldnames
        headers = [name.strip() for name in raw_headers]
        if len(set(headers)) != len(headers):
            raise ValueError("QikProp CSV contains duplicate column names")
        titles = [header for header in headers if header.casefold() in _TITLE_NAMES]
        if not titles:
            raise ValueError("QikProp CSV needs a molecule, title, or s_m_title column")
        property_headers = [_PROPERTY_NAMES[header.casefold()] for header in headers
                            if header.casefold() in _PROPERTY_NAMES]
        if len(set(property_headers)) != len(property_headers):
            raise ValueError("QikProp CSV contains ambiguous duplicate property aliases")
        rows = []
        for line, raw in enumerate(reader, 2):
            if None in raw and any(str(value).strip() for value in raw[None]):
                raise ValueError(f"QikProp CSV line {line} has more values than headers")
            row = {header: (raw.get(original) or "").strip()
                   for original, header in zip(raw_headers, headers)}
            if not any(row.values()):
                continue
            rows.append((line, row))
    return titles, rows


def _variant(row: dict[str, str], line: int, raw_titles: list[str]) -> dict[str, Any]:
    available = {_PROPERTY_NAMES[name.casefold()]: value for name, value in row.items()
                 if name.casefold() in _PROPERTY_NAMES}
    properties, missing, invalid, outside = {}, [], [], []
    for name, (lower, upper) in QIKPROP_REGION.items():
        value = available.get(name, "")
        if not value:
            missing.append(name)
            continue
        try:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("Non-finite value")
        except (ValueError, TypeError):
            invalid.append(name)
            continue
        properties[name] = numeric
        if not lower <= numeric <= upper:
            outside.append(name)
    complete = not missing and not invalid
    return {
        "csv_line": line, "titles": raw_titles, "properties": properties,
        "complete": complete, "passed": not outside if complete else None,
        "missing_properties": missing, "invalid_properties": invalid,
        "out_of_range_properties": outside,
    }


def _aggregate(variants: list[dict[str, Any]]) -> dict[str, Any]:
    if not variants:
        return _unavailable("missing", "No QikProp row matched this molecule ID")
    complete = [variant for variant in variants if variant["complete"]]
    passing = [variant for variant in complete if variant["passed"]]
    all_complete = len(complete) == len(variants)
    if passing:
        # One verified passing variant is sufficient, even if others failed.
        passed, status, error, chosen = True, "ok", None, passing[0]
    elif all_complete:
        passed, status, error, chosen = False, "ok", None, complete[0]
    else:
        passed, status = None, "incomplete"
        error = "Required properties are missing or invalid; no fully measured variant is known to pass"
        chosen = complete[0] if complete else variants[0]
    return {
        "status": status, "passed": passed, "properties": dict(chosen["properties"]),
        "error": error, "complete": all_complete, "variants": variants,
        "variants_observed": len(variants), "variants_complete": len(complete),
        "selected_csv_line": chosen["csv_line"],
        "warnings": [] if all_complete else [
            f"{len(variants) - len(complete)} of {len(variants)} observed variants have incomplete properties"
        ],
    }


def parse_qikprop_csv(path: str | Path, expected_ids: Iterable[str], *,
                      title_map: Mapping[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Parse an existing export without invoking or importing commercial tools."""
    identities = _ids(expected_ids)
    expected = set(identities)
    aliases = dict(title_map or {})
    for title, identity in aliases.items():
        if (not isinstance(title, str) or not title.strip()
                or not isinstance(identity, str) or identity not in expected):
            raise ValueError("title_map must map nonempty CSV titles to expected molecule IDs")
        if title in expected and title != identity:
            raise ValueError("title_map cannot reassign a title that is already an expected ID")
    titles, rows = _read_csv(Path(path))
    grouped: dict[str, list[dict[str, Any]]] = {identity: [] for identity in identities}
    for line, row in rows:
        raw_titles = list(dict.fromkeys(row[column] for column in titles if row[column]))
        matches = {aliases.get(title, title) for title in raw_titles
                   if aliases.get(title, title) in expected}
        if len(matches) != 1:
            reason = "Conflicting" if len(matches) > 1 else "Unmapped"
            raise ValueError(f"{reason} QikProp title(s) on line {line}: {raw_titles!r}; supply an explicit title_map")
        identity = matches.pop()
        grouped[identity].append(_variant(row, line, raw_titles))
    return {identity: _aggregate(variants) for identity, variants in grouped.items()}


class QikPropEvaluator:
    """Explicit opt-in adapter for an already installed, licensed vendor suite."""

    def __init__(self, schrodinger_root: str | Path | None, work_dir: str | Path,
                 timeout: float = 3600):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("QikProp timeout must be a finite positive number of seconds")
        self.schrodinger_root = Path(schrodinger_root).expanduser().resolve() if schrodinger_root else None
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.timeout = timeout

    @staticmethod
    def _add_execution_summary(result: dict[str, Any], folder: Path) -> None:
        """Retain vendor-reported failures of states absent from the CSV."""
        summaries = set()
        pattern = re.compile(
            r"(\d+)\s+molecules\s+were\s+successfully\s+processed\s*,\s*"
            r"(\d+)\s+failed\s*,\s*(\d+)\s+skipped", re.IGNORECASE
        )
        logs = list(folder.glob("*.out")) + [folder / "qikprop.console.log"]
        for path in logs:
            if path.is_file():
                summaries.update(tuple(int(value) for value in match)
                                 for match in pattern.findall(path.read_text(errors="replace")))
        if len(summaries) != 1:
            result["execution_summary"] = None
            result["execution_complete"] = None
            result["warnings"].append(
                "No unambiguous vendor completion summary was found; completeness covers observed CSV variants only"
            )
            return
        processed, failed, skipped = summaries.pop()
        result["execution_summary"] = {"processed": processed, "failed": failed, "skipped": skipped}
        known_complete = failed == skipped == 0 and processed == result["variants_observed"]
        result["execution_complete"] = known_complete
        if not known_complete:
            reason = (f"Vendor reports processed={processed}, failed={failed}, skipped={skipped}; "
                      f"CSV contains {result['variants_observed']} variants")
            result["complete"] = False
            result["warnings"].append(reason)
            if result["passed"] is not True:
                result.update(status="incomplete", passed=None, error=reason)

    def _launch(self, command: list[str], folder: Path, stage: str) -> None:
        environment = os.environ.copy()
        environment["SCHRODINGER"] = str(self.schrodinger_root)
        scratch = folder / "scratch"
        scratch.mkdir(exist_ok=True)
        environment["SCHRODINGER_TMPDIR"] = str(scratch)
        (folder / f"{stage}.command.json").write_text(
            json.dumps(command, indent=2) + "\n", encoding="utf-8"
        )
        try:
            result = subprocess.run(command, cwd=folder, env=environment, capture_output=True,
                                    text=True, timeout=self.timeout, check=False)
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            (folder / f"{stage}.console.log").write_text(output, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(f"{stage} exited with code {result.returncode}; see {stage}.console.log")
        except subprocess.TimeoutExpired as error:
            def text(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
            (folder / f"{stage}.console.log").write_text(
                text(error.stdout) + "\n" + text(error.stderr), encoding="utf-8"
            )
            raise RuntimeError(
                f"{stage} launcher exceeded {self.timeout}s; inspect the vendor Job Monitor for any "
                "surviving backend job. Output and job IDs, when reported, are in the console log."
            ) from error

    def evaluate(self, records: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
        if any(not isinstance(record, Mapping) for record in records):
            raise ValueError("QikProp records must be objects with id and smiles")
        identities = _ids(record.get("id") for record in records)
        if self.schrodinger_root is None:
            return {identity: _unavailable("not_run", "No explicit Schrödinger installation was configured")
                    for identity in identities}
        binaries = {name: self.schrodinger_root / name for name in ("ligprep", "qikprop")}
        missing = [str(path) for path in binaries.values() if not path.is_file() or not os.access(path, os.X_OK)]
        if missing:
            return {identity: _unavailable("failed", "Executable not found or not executable: " + ", ".join(missing))
                    for identity in identities}
        self.work_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for record in records:
            identity, smiles = record["id"], record.get("smiles")
            if not isinstance(smiles, str) or not smiles or any(char.isspace() or char == "\0" for char in smiles):
                results[identity] = _unavailable("failed", "A nonempty SMILES without whitespace or a title is required")
                continue
            token = "PROBE_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
            folder = Path(tempfile.mkdtemp(prefix=token + "-", dir=self.work_dir))
            (folder / "input.smi").write_text(f"{smiles}\t{token}\n", encoding="utf-8")
            (folder / "input.json").write_text(json.dumps({
                "id": identity, "smiles": smiles, "input_title": token,
                "identity_policy": "one input per isolated job; all generated states belong to this input",
            }, indent=2) + "\n", encoding="utf-8")
            try:
                self._launch([
                    str(binaries["ligprep"]), "-ismi", "input.smi", "-omae", "prepared.maegz",
                    "-bff", "16", "-epik", "-s", "1", "-WAIT", "-HOST", "localhost",
                ], folder, "ligprep")
                if not (folder / "prepared.maegz").is_file():
                    raise RuntimeError("LigPrep returned without prepared.maegz; inspect its logs")
                self._launch([
                    str(binaries["qikprop"]), "prepared.maegz", "-outname", "properties",
                    "-WAIT", "-HOST", "localhost",
                ], folder, "qikprop")
                exports = [path for path in folder.iterdir()
                           if path.is_file() and path.suffix.casefold() == ".csv"]
                if len(exports) != 1:
                    raise RuntimeError(f"Expected exactly one QikProp CSV, found {len(exports)}")
                csv_path = exports[0]
                title_columns, rows = _read_csv(csv_path)
                # This is an explicit provenance map, not a row-number guess:
                # there was exactly one input molecule in this fresh directory.
                title_map = {row[column]: identity for _, row in rows for column in title_columns if row[column]}
                result = parse_qikprop_csv(csv_path, [identity], title_map=title_map)[identity]
                result["csv_path"] = str(csv_path)
                result["title_map"] = title_map
                result["identity_policy"] = "one input per isolated job"
                self._add_execution_summary(result, folder)
            except (OSError, ValueError, csv.Error, RuntimeError) as error:
                result = _unavailable("failed", str(error))
            result["work_dir"] = str(folder)
            (folder / "result.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            results[identity] = result
        return results
