# PROBE

**Probe Before You Edit: Probing-Guided Molecular Optimization for LLM Agents in Structure-Based Drug Design**  
Zaifei Yang, Weiyu Chen, Yaqing Wang, and James Kwok · NeurIPS 2026

PROBE uses controlled molecular probes to guide multi-agent ligand optimization
in a protein pocket. The implementation includes InteractionAgent, DesignAgent,
InductionAgent, ExecutionAgent, and MoleculeGenerationAgent.

## Installation

Python 3.10 or newer is required. Linux is recommended for docking and PLIP.

```bash
conda env create -f environment.yml
conda activate probe
probe --help
```

For an existing environment with RDKit and Open Babel installed:

```bash
python -m pip install -e '.[docking]'
```

## Inputs

Follow the [CIDD data-preparation instructions](https://github.com/bowen-gao/CIDD#data-preparation)
to prepare the CrossDocked2020 inputs. Supply a receptor PDB and a 3D reference
ligand SDF/MOL or PDB for each case. Datasets and fragment libraries are not bundled.

Copy `examples/inputs.csv` and replace its example values:

| Column | Input |
|---|---|
| `id` | Unique identifier using letters, digits, `_`, or `-` |
| `smiles` | Starting molecule SMILES |
| `receptor` | Receptor PDB path |
| `reference_ligand` | Reference ligand defining the docking box |
| `receptor_pdbqt` | Optional prepared receptor PDBQT path |

Relative paths are resolved against `--data-root`. If automatic receptor
preparation fails, provide a compatible PDBQT through `receptor_pdbqt`.

Supply an annotated BRICS fragment JSON file with `--fragment-db`, or build one:

```bash
python -m pip install chembl_webresource_client
python scripts/download_chembl.py --output assets/chembl_clinical.smi
python scripts/build_fragment_library.py \
  --input assets/chembl_clinical.smi \
  --output assets/fragments.json
```

## Run

Configure an OpenAI-compatible model using environment variables:

```bash
export OPENAI_API_KEY='your-api-key'
export OPENAI_MODEL='your-model-id'
# Optional:
# export OPENAI_BASE_URL='https://your-provider.example/v1'
```

`.env.example` lists the variables; `.env` files are not loaded automatically.

```bash
probe check \
  --input my_inputs.csv --data-root /path/to/crossdocked \
  --fragment-db assets/fragments.json

probe run \
  --input my_inputs.csv --data-root /path/to/crossdocked \
  --fragment-db assets/fragments.json \
  --output-dir runs/first_run \
  --rounds 5 --exhaustiveness 16 --seed 42 --cpu 1
```

Use a new output directory for each run. `results.jsonl` contains one record per
completed case, including failures. Keep the full run directory for evaluation.
`python PROBE_generation.py` and `python -m probe` provide the same commands.

## Evaluation

```bash
python evaluation/evaluate.py --run-dir runs/first_run
# Equivalent:
# probe evaluate --run-dir runs/first_run
```

Evaluation reads the saved run, reuses Vina scores, and computes molecular
properties without model calls. Outputs are saved under `runs/first_run/evaluation/`:
`paper_table.csv`, `paper_table.md`, `summary.json`, `per_molecule.csv`, and
`per_molecule.json`.

```text
Method | Vina | QED | SA | Success | MRR | QikProp | Lipinski | LogP | MW | Rank
```

QikProp requires `--qikprop-csv PATH` or a licensed installation specified with
`--schrodinger-root PATH`; unavailable values are `n/a`. Single-run Rank is also
`n/a`. Use `--redock` for fresh docking, or `--output-dir PATH` for another output
location. Run `python evaluation/evaluate.py --help` for all options.

To rank evaluations of the same inputs and matching evaluation settings, use
separate `--method` labels when evaluating and then run:

```bash
python evaluation/rank.py \
  --summaries runs/method_a/evaluation/summary.json runs/method_b/evaluation/summary.json \
  --output-dir runs/comparison
```

## Citation

```bibtex
@inproceedings{yang2026probe,
  title={Probe Before You Edit: Probing-Guided Molecular Optimization for {LLM} Agents in Structure-Based Drug Design},
  author={Yang, Zaifei and Chen, Weiyu and Wang, Yaqing and Kwok, James},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

Includes code adapted from [CIDD](https://github.com/bowen-gao/CIDD) under the
MIT license; its copyright notice is retained in [LICENSE](LICENSE).
External software and datasets retain their respective licenses.
