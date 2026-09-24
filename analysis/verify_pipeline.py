#!/usr/bin/env python3
"""Fast invariant checks for the repaired pipeline."""

from pathlib import Path

import numpy as np

from core import SYNTHETIC_INITIAL, extract_classification_anchors, extract_real_data, make_time_split, mask_for_times, synthetic_dataset
from neural import fit_neural, initial_condition_error


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dryad-dir", type=Path, required=True)
    args = parser.parse_args()

    data = extract_real_data(args.dryad_dir)
    amp = data[data.condition.eq("ampicillin")]
    counts = amp.groupby("time_h", sort=True).size().to_numpy()
    assert np.array_equal(counts, [6, 5, 6, 6, 6, 6, 3, 6, 6, 3, 4, 3])
    assert set(amp.replicate.unique()) == {1, 2, 3, 4, 5, 6}
    assert np.allclose(data[data.time_h.eq(0)].relative_abundance, 1.0)
    anchors = extract_classification_anchors(args.dryad_dir)
    assert anchors["ampicillin"]["growing"] == 6
    assert anchors["ampicillin"]["total_classified_persisters"] == 6
    assert anchors["ciprofloxacin"]["growing"] == 29
    assert anchors["ciprofloxacin"]["indistinguishable"] == 3
    assert anchors["ciprofloxacin"]["total_classified_persisters"] == 32

    synthetic = synthetic_dataset(1, 0.10)
    split = make_time_split(synthetic.time_h, 123)
    train = mask_for_times(synthetic.time_h, split.train_times)
    test = mask_for_times(synthetic.time_h, split.test_times)
    assert np.all(train ^ test)
    assert not np.any(train & test)
    assert 0.0 in split.train_times and 0.0 not in split.test_times

    neural = fit_neural(
        synthetic.time_h.to_numpy()[train],
        synthetic.log_observed_total.to_numpy()[train],
        SYNTHETIC_INITIAL,
        12.0,
        seed=1,
        physics=True,
        real=False,
        epochs=2,
        lbfgs_steps=0,
        collocation_points=20,
    )
    assert initial_condition_error(neural) < 1e-12
    print("PASS: extraction, split isolation, and exact neural initial conditions")


if __name__ == "__main__":
    main()
