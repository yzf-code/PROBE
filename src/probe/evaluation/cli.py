"""Evaluation commands are independent of language-model configuration."""

from pathlib import Path


def add_commands(commands):
    evaluate = commands.add_parser("evaluate", help="Read a saved run directory and emit the paper metric table")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, help="New output directory; defaults to RUN_DIR/evaluation")
    evaluate.add_argument("--method", default="PROBE", help="Row label in the paper table")
    evaluate.add_argument("--mrr-mode", choices=("legacy", "corrected"), default="legacy")
    evaluate.add_argument("--redock", action="store_true", help="Re-dock final molecules instead of using saved Vina Dock")
    evaluate.add_argument("--exhaustiveness", type=int, default=16)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--cpu", type=int, default=1)
    qp = evaluate.add_mutually_exclusive_group()
    qp.add_argument("--qikprop-csv", type=Path, help="Import measured QikProp properties with explicit molecule IDs")
    qp.add_argument("--schrodinger-root", type=Path, help="Explicitly run installed licensed LigPrep and QikProp")
    evaluate.add_argument("--qikprop-title-map", type=Path, help="Optional JSON mapping CSV titles to run IDs")
    evaluate.add_argument("--qikprop-timeout", type=float, default=3600)
    evaluate.add_argument("--diagnostics", action="store_true", help="Also summarize saved optimization trajectories")
    rank = commands.add_parser("rank", help="Combine comparable evaluation summaries and rank by Success")
    rank.add_argument("--summaries", type=Path, nargs="+", required=True)
    rank.add_argument("--output-dir", type=Path, required=True)


def dispatch(args):
    from .runner import evaluate_run, rank_summaries, format_table
    if args.command == "rank":
        print(format_table(rank_summaries(args.summaries, args.output_dir)))
        return 0
    summary = evaluate_run(
        args.run_dir, args.output_dir, method=args.method, mrr_mode=args.mrr_mode,
        redock=args.redock, exhaustiveness=args.exhaustiveness, seed=args.seed, cpu=args.cpu,
        qikprop_csv=args.qikprop_csv, qikprop_title_map=args.qikprop_title_map,
        schrodinger_root=args.schrodinger_root, qikprop_timeout=args.qikprop_timeout,
        include_diagnostics=args.diagnostics,
    )
    print(format_table([summary["paper_row"]]))
    output = args.output_dir or args.run_dir / "evaluation"
    print(f"Saved metric table, per-molecule records, and coverage to {output}")
    if summary["counts"]["evaluated"] != summary["counts"]["requested"]:
        print("Some inputs lack complete core metrics; inspect summary.json coverage and per_molecule.csv.")
        return 1
    if (args.qikprop_csv or args.schrodinger_root) and summary["paper_row"]["QikProp"] is None:
        print("Requested QikProp evaluation is incomplete; inspect qikprop_details.json and coverage.")
        return 1
    return 0
