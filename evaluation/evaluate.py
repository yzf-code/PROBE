"""Read a completed PROBE run and produce the manuscript's metric columns."""

import sys
from probe.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
