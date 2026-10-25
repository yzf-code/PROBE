"""Combine saved evaluation summaries and calculate the paper's Success rank."""

import sys
from probe.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["rank", *sys.argv[1:]]))
