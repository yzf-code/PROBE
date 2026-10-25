"""Evaluate one saved final molecule per input and produce the paper table."""

from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import mean

from probe.records import load_run_records, resolve_record_file
from .metrics import score_molecule, success

PAPER_COLUMNS = ("Method", "Vina", "QED", "SA", "Success", "MRR", "QikProp", "Lipinski", "LogP", "MW", "Rank")
METRIC_COLUMNS = {"Vina": "vina_score", "QED": "qed", "SA": "sa", "Lipinski": "lipinski", "LogP": "logp", "MW": "mw"}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _write_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def format_table(rows):
    lines = ["| " + " | ".join(PAPER_COLUMNS) + " |", "| " + " | ".join(["---"] * len(PAPER_COLUMNS)) + " |"]
    for row in rows:
        cells = []
        for column in PAPER_COLUMNS:
            value = row.get(column)
            if value is None:
                cell = "n/a"
            elif column == "Method":
                cell = str(value).replace("|", "\\|").replace("\n", " ")
            elif column in {"Success", "MRR", "QikProp"}:
                cell = f"{value:.2f}%"
            elif column == "Rank":
                cell = str(value)
            else:
                cell = f"{value:.2f}" if column == "MW" else f"{value:.3f}"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def summarize(rows, *, method, mrr_mode, input_ids, vina_source):
    total = len(rows)
    if not total:
        raise ValueError("Cannot evaluate an empty run")
    paper = {column: None for column in PAPER_COLUMNS}
    paper["Method"] = method
    coverage = {}
    for column, key in METRIC_COLUMNS.items():
        values = [row[key] for row in rows if _finite(row.get(key))]
        paper[column] = mean(values) if values else None
        coverage[column] = {"observed": len(values), "total": total}
    for column, key in (("Success", "success"), ("MRR", "mrr")):
        observed = [row[key] for row in rows if type(row.get(key)) is bool]
        passed = sum(value is True for value in observed)
        paper[column] = 100.0 * passed / total
        coverage[column] = {"passed": passed, "observed": len(observed), "total": total,
                            "denominator": "all_requested_inputs",
                            "observed_only_percent": 100.0 * passed / len(observed) if observed else None}
    qp = [row["qikprop_pass"] for row in rows if type(row.get("qikprop_pass")) is bool]
    paper["QikProp"] = 100.0 * sum(qp) / total if len(qp) == total else None
    coverage["QikProp"] = {"passed": sum(qp), "observed": len(qp), "total": total,
                           "observed_only_percent": 100.0 * sum(qp) / len(qp) if qp else None,
                           "table_requires_complete_coverage": True}
    cohort = sorted(input_ids)
    return {
        "schema_version": 1, "method": method, "paper_row": paper,
        "counts": {"requested": total, "evaluated": sum(row["status"] == "ok" for row in rows),
                   "statuses": dict(Counter(row["status"] for row in rows)),
                   "qikprop_statuses": dict(Counter(row["qikprop_status"] for row in rows))},
        "coverage": coverage, "input_ids": cohort,
        "protocol": {
            "selection": "one saved final winner per original input; no best-of-n reselection",
            "vina_source": vina_source, "mrr_mode": mrr_mode,
            "continuous_aggregation": "mean over finite available measurements, counts reported per column",
            "success": "Vina < -8.18 and QED > 0.25 and normalized SA > 0.59; MRR is separate",
            "pass_ratio_denominator": "all requested inputs for Success/MRR; failed inputs remain counted",
            "qikprop": "all 41 configured properties pass; any complete passing LigPrep variant suffices",
            "rank": "requires multiple comparable summaries; descending Success, competition ties",
        },
    }


def evaluate_run(run_dir, output_dir=None, *, method="PROBE", mrr_mode="legacy", redock=False,
                 exhaustiveness=16, seed=42, cpu=1, qikprop_csv=None, qikprop_title_map=None,
                 schrodinger_root=None, qikprop_timeout=3600, include_diagnostics=False,
                 evaluator_factory=None, qikprop_factory=None):
    root = Path(run_dir).expanduser().resolve()
    records, manifest = load_run_records(root)
    output = Path(output_dir).expanduser().resolve() if output_dir else root / "evaluation"
    if qikprop_csv and schrodinger_root:
        raise ValueError("Choose either imported QikProp CSV or running QikProp")
    if qikprop_title_map and not qikprop_csv:
        raise ValueError("A QikProp title map requires --qikprop-csv")
    if mrr_mode not in {"legacy", "corrected"}:
        raise ValueError("mrr_mode must be legacy or corrected")
    if any(type(x) is not int or x < 1 for x in (exhaustiveness, cpu, seed)) or seed >= 2**31:
        raise ValueError("Positive docking parameters and a seed below 2**31 are required")
    output.mkdir(parents=True, exist_ok=False)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    rows = []
    for record in records:
        row = {"id": record["id"], "status": "failed", "generation_status": record.get("status"),
               "smiles": None, "vina_score": None, "vina_source": "redock" if redock else "saved",
               "qed": None, "sa": None, "logp": None, "mw": None, "lipinski": None,
               "mrr": None, "success": None, "error": None,
               "qikprop_status": "not_run", "qikprop_pass": None, "qikprop_error": None}
        rows.append(row)
        selected = record.get("selected")
        if record.get("status") != "ok" or not isinstance(selected, dict) or selected.get("status") != "ok":
            row["error"] = record.get("error") or "No successfully selected final molecule"
            continue
        row["smiles"] = selected.get("smiles")
        try:
            row.update(score_molecule(row["smiles"], mrr_mode=mrr_mode))
        except Exception as exc:
            row["error"] = f"Molecular property evaluation failed: {exc}"
            continue
        try:
            if redock:
                if evaluator_factory is None:
                    from probe.chemistry.docking import VinaEvaluator
                    factory = VinaEvaluator
                else:
                    factory = evaluator_factory
                inputs = record["input"]
                evaluator = factory(inputs["receptor"], inputs["reference_ligand"],
                                    output / "docking" / f"case_{len(rows):06d}",
                                    receptor_pdbqt=inputs.get("receptor_pdbqt"),
                                    exhaustiveness=exhaustiveness, seed=seed, cpu=cpu)
                evaluated = evaluator.evaluate(row["smiles"])
                vina = evaluated.get("vina_score")
                row["docking_protocol"] = evaluated.get("docking_protocol")
            else:
                saved = selected.get("metrics") or {}
                vina = saved.get("vina_score")
                row["docking_protocol"] = saved.get("docking_protocol")
            if not _finite(vina):
                raise ValueError("No finite Vina Dock score; use --redock with available receptor/reference files")
            row["vina_score"] = float(vina)
            row["success"] = success(vina, row["qed"], row["sa"])
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = "partial"
            row["error"] = f"Vina evaluation unavailable: {exc}"

    qp_results = {}
    if qikprop_csv:
        from .qikprop import parse_qikprop_csv
        title_map = json.loads(Path(qikprop_title_map).read_text()) if qikprop_title_map else None
        qp_results = parse_qikprop_csv(qikprop_csv, [row["id"] for row in rows], title_map=title_map)
    elif schrodinger_root:
        from .qikprop import QikPropEvaluator
        factory = qikprop_factory or QikPropEvaluator
        qp = factory(schrodinger_root, output / "qikprop", timeout=qikprop_timeout)
        qp_inputs = [{"id": row["id"], "smiles": row["smiles"]} for row in rows if row["mrr"] is not None]
        qp_results = qp.evaluate(qp_inputs)
    if qikprop_csv or schrodinger_root:
        for row in rows:
            result = qp_results.get(row["id"], {"status": "missing", "passed": None,
                                                "error": "No QikProp observation for this input"})
            if row["mrr"] is None:
                result = {"status": "unavailable", "passed": None,
                          "error": "No valid saved final molecule to associate with QikProp measurements",
                          "ignored_measurement": qp_results.get(row["id"])}
                qp_results[row["id"]] = result
            row.update(qikprop_status=result["status"], qikprop_pass=result.get("passed"),
                       qikprop_error=result.get("error"))
        _write_json(output / "qikprop_details.json", qp_results)

    summary = summarize(rows, method=method, mrr_mode=mrr_mode,
                        input_ids=[record["id"] for record in records], vina_source="redock" if redock else "saved")
    from .comparability import comparison_metadata
    summary["comparison"] = comparison_metadata(
        records, rows, manifest, redock=redock, exhaustiveness=exhaustiveness,
        seed=seed, cpu=cpu, mrr_mode=mrr_mode,
    )
    summary["cohort_id"] = summary["comparison"]["cohort_id"]
    summary["generation_manifest"] = manifest
    summary["evaluation_config"] = {"run_dir": str(root), "redock": redock,
                                    "redock_exhaustiveness": exhaustiveness if redock else None,
                                    "redock_seed": seed if redock else None, "redock_cpu": cpu if redock else None,
                                    "qikprop_source": "csv" if qikprop_csv else "installed" if schrodinger_root else "not_run"}
    _write_json(output / "per_molecule.json", rows)
    columns = ("id", "status", "generation_status", "smiles", "vina_score", "qed", "sa", "success", "mrr",
               "qikprop_pass", "lipinski", "logp", "mw", "vina_source", "mrr_mode", "qikprop_status", "error", "qikprop_error")
    _write_csv(output / "per_molecule.csv", rows, columns)
    _write_json(output / "summary.json", summary)
    _write_csv(output / "paper_table.csv", [summary["paper_row"]], PAPER_COLUMNS)
    (output / "paper_table.md").write_text(format_table([summary["paper_row"]]), encoding="utf-8")
    if include_diagnostics:
        from .diagnostics import evaluate_diagnostics
        trajectories = []
        for record in records:
            if record.get("result_file"):
                path = resolve_record_file(root, record["result_file"])
                trajectory = json.loads(path.read_text(encoding="utf-8"))
                trajectory["case_id"] = record["id"]
                trajectories.append(trajectory)
        _write_json(output / "diagnostics.json", evaluate_diagnostics(trajectories))
    return summary


def rank_summaries(paths, output_dir):
    summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if len(summaries) < 2:
        raise ValueError("Rank requires at least two method summaries")
    comparisons = [s.get("comparison", {}) for s in summaries]
    if any(not c.get("cohort_complete") or not c.get("protocol_complete") for c in comparisons):
        raise ValueError("Rank requires complete input identities and docking provenance; inspect comparison.warnings")
    if len({s.get("cohort_id") for s in summaries}) != 1 or not summaries[0].get("cohort_id"):
        raise ValueError("Rank requires the same input cohort, including molecules and receptor/reference contents")
    if len({c.get("protocol_id") for c in comparisons}) != 1 or not comparisons[0].get("protocol_id"):
        raise ValueError("Rank requires matching metric and docking protocols; inspect comparison.protocol")
    if len({s.get("protocol", {}).get("mrr_mode") for s in summaries}) != 1:
        raise ValueError("Do not mix legacy and corrected MRR protocols in one table")
    names = [s["paper_row"]["Method"] for s in summaries]
    if len(set(names)) != len(names):
        raise ValueError("Use a distinct --method name for each summary")
    rows = [dict(s["paper_row"]) for s in summaries]
    if any(not _finite(row.get("Success")) for row in rows):
        raise ValueError("Every method requires a finite Success percentage")
    for row in rows:
        row["Rank"] = 1 + sum(other["Success"] > row["Success"] for other in rows)
    rows.sort(key=lambda row: row["Rank"])
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    _write_csv(output / "paper_table.csv", rows, PAPER_COLUMNS)
    (output / "paper_table.md").write_text(format_table(rows), encoding="utf-8")
    _write_json(output / "ranked_summary.json", {"rows": rows, "source_summaries": [str(p) for p in paths],
                                                "ranking": "descending Success; competition ties"})
    return rows
