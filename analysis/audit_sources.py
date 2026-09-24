#!/usr/bin/env python3
"""Write a compact machine-readable audit of all supplied source tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core import extract_classification_anchors, extract_peyrusson_oxacillin, extract_real_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryad-dir", type=Path, required=True)
    parser.add_argument("--peyrusson-workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    real = extract_real_data(args.dryad_dir)
    peyrusson = extract_peyrusson_oxacillin(args.peyrusson_workbook)
    audit = {
        "umetani_observed_rows": real.groupby("condition").size().to_dict(),
        "umetani_replicate_counts_by_time": {
            condition: group.groupby("time_h", sort=True).size().tolist()
            for condition, group in real.groupby("condition")
        },
        "classification_rows": extract_classification_anchors(args.dryad_dir),
        "peyrusson_observed_rows": len(peyrusson),
        "peyrusson_replicate_counts_by_time": peyrusson.groupby("time_h", sort=True).size().tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
