"""Unit tests for the statistics helpers used in the validation scripts
(bootstrap confidence intervals, McNemar significance test).

Requires numpy/scipy (see requirements.txt) — these are the same functions
used to produce the bootstrap CIs and McNemar p-values reported in the paper
(Section VII, "Statistical notes").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.validation_and_statistics.bootstrap_confidence_intervals import bootstrap_ci
from experiments.validation_and_statistics.significance_tests import mcnemar_test


class TestBootstrapCI:
    def test_constant_values_give_a_degenerate_interval(self):
        mean, lo, hi = bootstrap_ci([0.0, 0.0, 0.0, 0.0], n_boot=500)
        assert mean == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_interval_contains_the_sample_mean(self):
        values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        mean, lo, hi = bootstrap_ci(values, n_boot=2000)
        assert lo <= mean <= hi

    def test_empty_input_returns_nan(self):
        mean, lo, hi = bootstrap_ci([])
        assert mean != mean  # NaN != NaN


class TestMcNemarTest:
    def test_identical_configs_give_no_discordant_pairs(self):
        a = [1, 0, 1, 0, 1]
        b = [1, 0, 1, 0, 1]
        chi2, p, note = mcnemar_test(a, b)
        assert chi2 == 0.0
        assert p == 1.0
        assert "no discordant pairs" in note

    def test_a_strictly_worse_than_b_is_directionally_reported(self):
        # A succeeds (=attacked more often) where B never does.
        a = [1, 1, 1, 1, 0, 0, 0, 0]
        b = [0, 0, 0, 0, 0, 0, 0, 0]
        chi2, p, note = mcnemar_test(a, b)
        assert chi2 > 0
        assert "A causes more attacks" in note

    def test_symmetric_disagreement_has_low_chi2(self):
        a = [1, 0, 1, 0]
        b = [0, 1, 0, 1]
        chi2, p, note = mcnemar_test(a, b)
        assert "no directional difference" in note
