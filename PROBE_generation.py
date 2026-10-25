"""Thin script entry point; install PROBE first with `pip install -e .`."""

from probe.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
