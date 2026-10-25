"""Download clinical-stage source SMILES through the official ChEMBL client."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new filename")
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        parser.error("Install the optional client: pip install chembl_webresource_client")
    query = new_client.molecule.filter(max_phase__gte=1).only(
        ["molecule_chembl_id", "molecule_structures", "max_phase"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in query:
            smiles = (record.get("molecule_structures") or {}).get("canonical_smiles")
            if smiles:
                handle.write(f"{smiles}\t{record['molecule_chembl_id']}\n")
                count += 1
    temporary.replace(args.output)
    manifest = {
        "source": "https://www.ebi.ac.uk/chembl/api/data/molecule",
        "filter": {"max_phase__gte": 1},
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "molecules": count,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {count} molecules to {args.output}")


if __name__ == "__main__":
    main()
