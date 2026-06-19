#!/usr/bin/env python
"""Summarize preserved legacy manifests."""
from pathlib import Path
import csv

for manifest in sorted(Path("legacy/manifests").glob("*.csv")):
    with manifest.open(encoding="utf-8") as handle:
        n = sum(1 for _ in csv.DictReader(handle))
    print(f"{manifest}: {n} files")
