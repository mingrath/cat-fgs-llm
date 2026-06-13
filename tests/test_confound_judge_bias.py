"""F2 — judge-bias confound axis over the VLM-as-AU-rater.

Pins the shift/flip math and the one-directional NO_CONFOUND_MSG framing (a small shift
never reads as "no bias", only "no confound detected at this power").
"""

import numpy as np

from src.eval.confound import (
    JUDGE_BIAS_PERTURBATIONS,
    NO_CONFOUND_MSG,
    judge_bias,
    judge_bias_shift,
)


def test_identical_scores_give_zero_shift_and_flip():
    cell = judge_bias_shift([0, 1, 2, 1], [0, 1, 2, 1])
    assert cell["shift"] == 0.0
    assert cell["flip_rate"] == 0.0
    assert cell["note"] == NO_CONFOUND_MSG


def test_known_shift_and_flip_rate():
    # baseline vs perturbed: |1-0|+|1-1|+|2-2| = 1 over 3 rows -> shift 1/3; one flip -> 1/3
    cell = judge_bias_shift([0, 1, 2], [1, 1, 2])
    assert np.isclose(cell["shift"], 1 / 3)
    assert np.isclose(cell["flip_rate"], 1 / 3)
    assert cell["n"] == 3


def test_judge_bias_table_shape_and_one_directional_note():
    baseline = {"ear": [0, 0], "orbital": [2, 2]}
    perturbed = {
        "position": {"ear": [0, 1], "orbital": [2, 2]},
        "verbosity": {"ear": [0, 0], "orbital": [2, 2]},
    }
    out = judge_bias(baseline, perturbed)
    assert out["note"] == NO_CONFOUND_MSG
    assert set(out["per_perturbation"]) == {"position", "verbosity"}
    # position perturbation flips one of two ear rows
    assert np.isclose(out["per_perturbation"]["position"]["ear"]["flip_rate"], 0.5)
    # verbosity perturbation changes nothing -> no detection at this power
    assert out["per_perturbation"]["verbosity"]["ear"]["shift"] == 0.0


def test_named_perturbation_template_is_present():
    # the DiffuJudge-style named template the protocol advertises
    assert "position" in JUDGE_BIAS_PERTURBATIONS
    assert "verbosity" in JUDGE_BIAS_PERTURBATIONS
    assert "self_enhancement" in JUDGE_BIAS_PERTURBATIONS
