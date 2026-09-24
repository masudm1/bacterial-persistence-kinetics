#!/usr/bin/env python3
"""Regenerate and collect the complete manuscript figure set."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
FIGURE_CODE = ROOT / "figure_code"
ERLANG = ROOT / "erlang_extension"
FINAL = ROOT / "final_manuscript_figures"


def run(script: Path) -> None:
    print(f"Running {script.relative_to(ROOT)}")
    subprocess.run([sys.executable, str(script)], check=True, cwd=script.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-erlang-refit",
        action="store_true",
        help="Keep the supplied Figure 12 instead of rerunning the slower Erlang fits.",
    )
    args = parser.parse_args()

    FINAL.mkdir(exist_ok=True)

    # The base builder creates Figures 0--11 from the included processed tables.
    run(FIGURE_CODE / "make_all_manuscript_figures.py")

    # These scripts apply the final layouts for Figures 2, 4, 5, and 11.
    run(FIGURE_CODE / "generate_fig2_architecture.py")
    run(FIGURE_CODE / "generate_updated_figures.py")

    if not args.skip_erlang_refit:
        run(ERLANG / "erlang_recovery_extension.py")

    for number in range(12):
        matches = sorted(FIGURE_CODE.glob(f"fig{number}_*.png"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one generated Figure {number}, found: {matches}"
            )
        shutil.copy2(matches[0], FINAL / matches[0].name)

    erlang_figure = ERLANG / "fig12_nonexponential_recovery.png"
    if erlang_figure.exists():
        shutil.copy2(erlang_figure, FINAL / erlang_figure.name)
    elif not (FINAL / erlang_figure.name).exists():
        raise FileNotFoundError(erlang_figure)

    print(f"Final figures are in: {FINAL}")


if __name__ == "__main__":
    main()
