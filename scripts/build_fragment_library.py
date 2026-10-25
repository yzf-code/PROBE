#!/usr/bin/env python3
"""Build an external BRICS library from user-supplied compound/fragment SMILES."""
import argparse
import json
from probe.chemistry.fragments import build_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='SMILES text file; optional name after whitespace')
    parser.add_argument('--output', required=True, help='Output JSON path outside the source tree')
    parser.add_argument('--fragments', action='store_true', help='Input already contains BRICS fragment SMILES')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_library(args.input, args.output, input_is_fragments=args.fragments, seed=args.seed), indent=2))


if __name__ == '__main__':
    main()
