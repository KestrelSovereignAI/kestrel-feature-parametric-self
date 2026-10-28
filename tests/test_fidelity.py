"""Tests for the fidelity gate + log parsers (pure, no MLX).

The June fixture is Emma's first live run (candidate ``ded33cd017d9``, #11):
the untuned base scored 8.977, the iter-200 checkpoint 1.866 and the final
iter-400 checkpoint 2.688. The run overfit in its back half; the pre-#30 gate
promoted its final checkpoint anyway because 2.688 < 3.0. Those numbers are
the run's logged validation losses. The fixture uses them as the scores of
the named checkpoint files, which is what the cycle now measures.
"""

from __future__ import annotations

import math

import pytest

from kestrel_feature_parametric_self import (
    Baseline,
    Checkpoint,
    FidelityGate,
    GateOutcome,
    GateReason,
    ValSet,
    parse_final_val_loss,
    parse_test_loss,
    recorded_inputs,
    saved_checkpoints,
)
from kestrel_feature_parametric_self.fidelity import (
    FINAL_ADAPTER_FILE,
    GATE_RECORD_FIELDS,
    loss_ratio,
    margin_threshold,
)

ITER_200 = "0000200_adapters.safetensors"
ITER_400 = "0000400_adapters.safetensors"
JUNE_LOG = (
    "Starting training..., iters: 400\n"
    "Iter 1: Val loss 8.977, Val took 41.512s\n"
    "Iter 10: Train loss 7.012, Learning Rate 1.000e-04, It/sec 0.4, Tokens/sec 190.1, "
    "Trained Tokens 4096, Peak mem 30.9 GB\n"
    "Iter 100: Saved adapter weights to a/adapters.safetensors and a/0000100_adapters.safetensors.\n"
    "Iter 200: Val loss 1.866, Val took 42.083s\n"
    "Iter 200: Saved adapter weights to a/adapters.safetensors and a/0000200_adapters.safetensors.\n"
    "Iter 300: Saved adapter weights to a/adapters.safetensors and a/0000300_adapters.safetensors.\n"
    "Iter 400: Val loss 2.688, Val took 41.900s\n"
    "Iter 400: Saved adapter weights to a/adapters.safetensors and a/0000400_adapters.safetensors.\n"
    "Saved final weights to a/adapters.safetensors.\n"
)
JUNE_FILES = (
    "0000100_adapters.safetensors", ITER_200, "0000300_adapters.safetensors", ITER_400,
    FINAL_ADAPTER_FILE,
)
VAL = ValSet(digest="sha256:" + "ab" * 32, examples=75, batch_size=4)
BASE = "mlx-community/gemma-4-31b-it-4bit"


def _june():
    checkpoints = (Checkpoint(ITER_200, 200, 1.866), Checkpoint(ITER_400, 400, 2.688))
    return checkpoints, Baseline("base", 8.977)


def _run(*points, baseline_loss=8.0, source="base"):
    """Checkpoints (iteration, loss) plus a baseline, without a log."""
    checkpoints = tuple(
        Checkpoint(f"{iteration:07d}_adapters.safetensors", iteration, loss)
        for iteration, loss in points
    )
    return checkpoints, Baseline(source, baseline_loss)


# ----------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------

def test_parse_returns_last_val_loss():
    log = (
        "Iter 1: Val loss 4.506, Val took 1.5s\n"
        "Iter 200: Val loss 2.635, Val took 1.0s\n"
        "Iter 400: Val loss 2.208, Val took 1.0s\n"
    )
    assert parse_final_val_loss(log) == 2.208


def test_parse_returns_none_when_absent():
    assert parse_final_val_loss("Iter 20: Train loss 3.3\n") is None
    assert parse_final_val_loss("") is None


def test_parse_test_loss_keeps_non_finite_losses_visible():
    assert math.isnan(parse_test_loss("Test loss nan, Test ppl nan."))
    assert parse_test_loss("Test loss inf, Test ppl inf.") == math.inf


def test_parse_test_loss():
    assert parse_test_loss("Testing\nTest loss 2.208, Test ppl 9.098.\n") == 2.208
    assert parse_test_loss("Testing\n") is None


def test_saved_checkpoints_lists_each_saved_file_once_and_unscored():
    """The final ``adapters.safetensors`` repeats the iter-400 save, so it is not listed twice."""
    checkpoints = saved_checkpoints(JUNE_FILES + ("adapter_config.json", "train.log"), final_iteration=400)
    assert [(c.checkpoint_id, c.iteration, c.val_loss, c.error) for c in checkpoints] == [
        ("0000100_adapters.safetensors", 100, None, None),
        (ITER_200, 200, None, None),
        ("0000300_adapters.safetensors", 300, None, None),
        (ITER_400, 400, None, None),
    ]


def test_the_final_weights_file_is_a_checkpoint_when_no_periodic_save_holds_them():
    checkpoints = saved_checkpoints((FINAL_ADAPTER_FILE, ITER_200), final_iteration=350)
    assert [(c.checkpoint_id, c.iteration) for c in checkpoints] == [
        (ITER_200, 200), (FINAL_ADAPTER_FILE, 350),
    ]
    assert saved_checkpoints((), final_iteration=400) == ()


# ----------------------------------------------------------------------
# June fixture (Emma's run): R1, R2, R3
# ----------------------------------------------------------------------

def test_june_r1_baseline_present_decision_is_not_inconclusive():
    checkpoints, baseline = _june()
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)
    assert decision.outcome is not GateOutcome.INCONCLUSIVE


def test_june_r2a_default_config_rejects_overfit_and_promotes_nothing():
    checkpoints, baseline = _june()
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)
    assert decision.outcome is GateOutcome.REJECT
    assert decision.reason is GateReason.OVERFIT
    assert decision.promote is False
    assert decision.promoted_checkpoint_id is None
    assert decision.to_record()["promoted_checkpoint_id"] is None
    assert decision.best.checkpoint_id == ITER_200
    assert decision.final.checkpoint_id == ITER_400
    assert decision.ratio == pytest.approx(2.688 / 1.866)


@pytest.mark.parametrize(
    "gate", [FidelityGate(overfit_check=False), FidelityGate(overfit_ratio=1.50)],
    ids=["overfit_check_off", "overfit_ratio_1.50"],
)
def test_june_r2b_relaxed_overfit_promotes_the_iter_200_checkpoint(gate):
    checkpoints, baseline = _june()
    decision = gate.evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)
    assert decision.outcome is GateOutcome.PROMOTE
    assert decision.promoted_checkpoint_id == ITER_200


@pytest.mark.parametrize("gate", [
    FidelityGate(),
    FidelityGate(overfit_check=False),
    FidelityGate(overfit_ratio=1.50),
    FidelityGate(margin=0.0, overfit_check=False),
    FidelityGate(margin=0.5, overfit_ratio=10.0),
])
def test_june_r2c_iter_400_is_never_promoted(gate):
    checkpoints, baseline = _june()
    decision = gate.evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)
    assert decision.promoted_checkpoint_id != ITER_400


def _pre_30_gate(log_text, prior_val_loss=None, max_val_loss=3.0, max_regression=0.25):
    """The gate this change replaced: final val loss against a 3.0 ceiling.

    Kept here only so the legacy failure stays visible. It returned the
    candidate directory whose served weights are its final
    ``adapters.safetensors`` (the iter-400 weights), or None.
    """
    final = parse_final_val_loss(log_text)
    if final is None or final > max_val_loss:
        return None
    if prior_val_loss is not None and final > prior_val_loss + max_regression:
        return None
    return FINAL_ADAPTER_FILE


def test_june_r3_legacy_gate_promoted_the_overfit_final_checkpoint():
    """Legacy-gate assertion: the old gate passes the June run at iter 400."""
    checkpoints, _ = _june()
    promoted = _pre_30_gate(JUNE_LOG)
    assert promoted == FINAL_ADAPTER_FILE
    final = max(checkpoints, key=lambda c: c.iteration)
    assert final.checkpoint_id == ITER_400
    assert parse_final_val_loss(JUNE_LOG) == final.val_loss == 2.688
    # The new gate, on the same fixture, does not.
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=_june()[1], val_set=VAL, base_model=BASE)
    assert decision.promote is False


# ----------------------------------------------------------------------
# Inconclusive (R4): never a pass
# ----------------------------------------------------------------------

def _assert_inconclusive(decision, reason):
    assert decision.outcome is GateOutcome.INCONCLUSIVE
    assert decision.reason is reason
    assert decision.detail
    assert decision.promote is False
    assert decision.promoted_checkpoint_id is None


def test_inconclusive_when_val_set_missing():
    checkpoints, baseline = _june()
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=baseline, val_set=None, base_model=BASE)
    _assert_inconclusive(decision, GateReason.VAL_SET_MISSING)


def test_inconclusive_when_val_set_empty():
    checkpoints, baseline = _june()
    decision = FidelityGate().evaluate(
        checkpoints=checkpoints, baseline=baseline, val_set=ValSet(VAL.digest, 0, 4), base_model=BASE,
    )
    _assert_inconclusive(decision, GateReason.VAL_SET_EMPTY)


@pytest.mark.parametrize("examples", [1, 3])
def test_inconclusive_when_val_set_is_smaller_than_one_scoring_batch(examples):
    """mlx_lm refuses to score a set smaller than one batch, so nothing could be scored."""
    checkpoints, baseline = _june()
    decision = FidelityGate().evaluate(
        checkpoints=checkpoints, baseline=baseline, val_set=ValSet(VAL.digest, examples, 4), base_model=BASE,
    )
    _assert_inconclusive(decision, GateReason.VAL_SET_TOO_SMALL)
    assert f"{examples} example(s), fewer than one scoring batch of 4" in decision.detail


def test_one_full_scoring_batch_is_enough():
    checkpoints, baseline = _june()
    decision = FidelityGate(overfit_check=False).evaluate(
        checkpoints=checkpoints, baseline=baseline, val_set=ValSet(VAL.digest, 4, 4), base_model=BASE,
    )
    assert decision.promoted_checkpoint_id == ITER_200


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0, -1.0], ids=["nan", "inf", "zero", "negative"])
def test_inconclusive_when_a_candidate_loss_is_unusable(bad):
    checkpoints, baseline = _run((200, 2.0), (400, bad))
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)
    _assert_inconclusive(decision, GateReason.CANDIDATE_LOSS_INVALID)


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0, -1.0], ids=["nan", "inf", "zero", "negative"])
def test_inconclusive_when_the_baseline_loss_is_unusable(bad):
    checkpoints, baseline = _run((200, 2.0), (400, 2.0), baseline_loss=bad)
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)
    _assert_inconclusive(decision, GateReason.BASELINE_LOSS_INVALID)


def test_non_finite_losses_stay_visible_in_the_record():
    checkpoints, baseline = _run((200, math.nan), (400, 2.0), baseline_loss=math.inf)
    record = FidelityGate().evaluate(
        checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE,
    ).to_record()
    assert record["final_val_loss"] == 2.0
    assert record["baseline_val_loss"] == "inf"


def test_inconclusive_when_there_are_no_checkpoints():
    decision = FidelityGate().evaluate(checkpoints=(), baseline=Baseline("base", 8.0), val_set=VAL, base_model=BASE)
    _assert_inconclusive(decision, GateReason.NO_CHECKPOINTS)


MISSING_LOSS_GATES = [
    FidelityGate(), FidelityGate(overfit_check=False), FidelityGate(overfit_ratio=1.50),
]
MISSING_LOSS_GATE_IDS = ["default", "overfit_check_off", "ratio_1.50"]


@pytest.mark.parametrize("gate", MISSING_LOSS_GATES, ids=MISSING_LOSS_GATE_IDS)
@pytest.mark.parametrize("losses", [
    (None, 2.688), (1.866, None), (None, None), (1.866, None, 2.688),
], ids=["intermediate", "final", "all", "middle"])
def test_inconclusive_when_any_checkpoint_has_no_val_loss(gate, losses):
    """An unscored checkpoint could have been the best; nothing is promoted, whatever the config."""
    checkpoints, baseline = _run(*((100 * (i + 1), loss) for i, loss in enumerate(losses)), baseline_loss=8.977)
    decision = gate.evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)
    _assert_inconclusive(decision, GateReason.CANDIDATE_LOSS_MISSING)
    assert decision.best is None
    assert decision.ratio is None
    unscored = next(c for c in checkpoints if c.val_loss is None)
    assert unscored.checkpoint_id in decision.detail
    record = decision.to_record()
    assert record["best_checkpoint_id"] is None
    assert record["final_checkpoint_id"] == checkpoints[-1].checkpoint_id
    assert record["promoted_checkpoint_id"] is None


@pytest.mark.parametrize("gate", MISSING_LOSS_GATES, ids=MISSING_LOSS_GATE_IDS)
def test_a_june_record_missing_its_iter_200_score_replays_inconclusive(gate):
    """The scored iter-400 file must not win just because iter 200's score was lost."""
    checkpoints, baseline = _june()
    record = gate.evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE).to_record()
    record["checkpoints"][0]["val_loss"] = None
    replayed, recorded_baseline, val_set, base_model = recorded_inputs(record)
    assert replayed == (Checkpoint(ITER_200, 200, None), Checkpoint(ITER_400, 400, 2.688))
    decision = gate.evaluate(
        checkpoints=replayed, baseline=recorded_baseline, val_set=val_set, base_model=base_model,
    )
    _assert_inconclusive(decision, GateReason.CANDIDATE_LOSS_MISSING)
    assert ITER_200 in decision.detail


def test_inconclusive_when_the_base_was_not_scored():
    checkpoints, baseline = _run((200, 1.866), (400, 1.800), baseline_loss=None)
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)
    _assert_inconclusive(decision, GateReason.BASELINE_UNAVAILABLE)


def test_inconclusive_when_a_checkpoint_could_not_be_scored():
    """One unscored file could have been the best, so nothing is promoted."""
    checkpoints = (
        Checkpoint(ITER_200, 200, None, error="evaluation reported no test loss"),
        Checkpoint(ITER_400, 400, 1.5),
    )
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=Baseline("base", 9.0), val_set=VAL, base_model=BASE)
    _assert_inconclusive(decision, GateReason.CANDIDATE_EVAL_FAILED)
    assert "no test loss" in decision.detail


def test_inconclusive_when_there_is_no_baseline():
    checkpoints, _ = _run((200, 2.0), (400, 2.0))
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=None, val_set=VAL, base_model=BASE)
    _assert_inconclusive(decision, GateReason.BASELINE_UNAVAILABLE)


def test_inconclusive_when_scoring_the_baseline_failed():
    checkpoints, _ = _run((200, 2.0), (400, 2.0))
    decision = FidelityGate().evaluate(
        checkpoints=checkpoints,
        baseline=Baseline("served:abc/0000200_adapters.safetensors", error="RuntimeError: boom"),
        val_set=VAL, base_model=BASE,
    )
    _assert_inconclusive(decision, GateReason.BASELINE_EVAL_FAILED)
    assert "boom" in decision.detail


def test_inconclusive_when_the_baseline_changed_before_the_decision():
    """A baseline that stopped being the baseline is never compared, whatever it scored."""
    checkpoints, _ = _run((200, 2.0), (400, 2.0))
    decision = FidelityGate().evaluate(
        checkpoints=checkpoints,
        baseline=Baseline(
            "served:abc/0000200_adapters.safetensors", 50.0,
            changed="the served adapter changed during this run",
        ),
        val_set=VAL, base_model=BASE,
    )
    _assert_inconclusive(decision, GateReason.BASELINE_CHANGED)
    assert "changed during this run" in decision.detail
    assert decision.to_record()["baseline_changed"] == "the served adapter changed during this run"


def test_inconclusive_checks_run_before_overfit_and_margin():
    """A run that would also fail overfit and margin reports what blocked it first."""
    checkpoints, _ = _run((200, 1.0), (400, 9.0))
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=None, val_set=VAL, base_model=BASE)
    _assert_inconclusive(decision, GateReason.BASELINE_UNAVAILABLE)


def test_overfit_is_checked_before_margin():
    checkpoints, baseline = _run((200, 7.99), (400, 9.6), baseline_loss=8.0)
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)
    assert decision.reason is GateReason.OVERFIT


# ----------------------------------------------------------------------
# Margin and overfit boundaries (R6), built with the gate's own expressions
# ----------------------------------------------------------------------

BASELINE_LOSS = 2.5


def _margin_case(best_loss):
    checkpoints, baseline = _run((200, best_loss), (400, best_loss), baseline_loss=BASELINE_LOSS)
    return FidelityGate().evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)


def test_margin_exactly_at_the_boundary_rejects():
    boundary = margin_threshold(BASELINE_LOSS, FidelityGate().margin)
    decision = _margin_case(boundary)
    assert decision.best.val_loss == decision.required_below  # exact equality, no rounding slack
    assert decision.reason is GateReason.MARGIN


def test_margin_just_inside_the_boundary_promotes():
    inside = math.nextafter(margin_threshold(BASELINE_LOSS, FidelityGate().margin), 0.0)
    decision = _margin_case(inside)
    assert decision.best.val_loss < decision.required_below
    assert decision.outcome is GateOutcome.PROMOTE


def test_margin_just_outside_the_boundary_rejects():
    outside = math.nextafter(margin_threshold(BASELINE_LOSS, FidelityGate().margin), math.inf)
    decision = _margin_case(outside)
    assert decision.best.val_loss > decision.required_below
    assert decision.reason is GateReason.MARGIN


def _ratio_case(final_loss, best_loss=1.25):
    checkpoints, baseline = _run((200, best_loss), (400, final_loss), baseline_loss=8.0)
    return FidelityGate().evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)


def test_ratio_exactly_at_the_limit_passes_the_overfit_check():
    # 1.25 and 1.5 are exact binary fractions; their quotient is exactly 1.2.
    assert loss_ratio(1.5, 1.25) == FidelityGate().overfit_ratio
    decision = _ratio_case(1.5)
    assert decision.ratio == decision.gate.overfit_ratio
    assert decision.outcome is GateOutcome.PROMOTE


def test_ratio_just_under_the_limit_passes_the_overfit_check():
    final = math.nextafter(1.5, 0.0)
    assert loss_ratio(final, 1.25) < FidelityGate().overfit_ratio
    assert _ratio_case(final).outcome is GateOutcome.PROMOTE


def test_ratio_just_over_the_limit_rejects_overfit():
    final = math.nextafter(1.5, math.inf)
    assert loss_ratio(final, 1.25) > FidelityGate().overfit_ratio
    decision = _ratio_case(final)
    assert decision.reason is GateReason.OVERFIT


# ----------------------------------------------------------------------
# The stricter gate must not block everything
# ----------------------------------------------------------------------

def test_a_clearly_improving_run_promotes_its_best_checkpoint():
    checkpoints, baseline = _run((100, 4.0), (200, 2.4), (300, 2.1), (400, 2.0), baseline_loss=9.0)
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)
    assert decision.outcome is GateOutcome.PROMOTE
    assert decision.reason is GateReason.PASSED
    assert decision.promoted_checkpoint_id == ITER_400


def test_a_mild_back_half_rise_promotes_the_best_not_the_final():
    checkpoints, baseline = _run((200, 2.0), (400, 2.2), baseline_loss=9.0)
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)
    assert decision.promoted_checkpoint_id == ITER_200


def test_ties_go_to_the_earlier_checkpoint():
    checkpoints, baseline = _run((200, 2.0), (400, 2.0), baseline_loss=9.0)
    decision = FidelityGate().evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE)
    assert decision.promoted_checkpoint_id == ITER_200


# ----------------------------------------------------------------------
# Decision record (R9): every field, every path
# ----------------------------------------------------------------------

def _decisions_by_path():
    june_checkpoints, june_base = _june()
    margin_checkpoints, margin_base = _run((200, 2.45), (400, 2.45), baseline_loss=BASELINE_LOSS)
    return {
        "promote": FidelityGate(overfit_ratio=1.5).evaluate(
            checkpoints=june_checkpoints, baseline=june_base, val_set=VAL, base_model=BASE),
        "reject-overfit": FidelityGate().evaluate(
            checkpoints=june_checkpoints, baseline=june_base, val_set=VAL, base_model=BASE),
        "reject-margin": FidelityGate().evaluate(
            checkpoints=margin_checkpoints, baseline=margin_base, val_set=VAL, base_model=BASE),
        "inconclusive": FidelityGate().evaluate(
            checkpoints=june_checkpoints, baseline=None, val_set=None, base_model=BASE),
    }


@pytest.mark.parametrize("path", ["promote", "reject-overfit", "reject-margin", "inconclusive"])
def test_every_path_records_every_field(path):
    decision = _decisions_by_path()[path]
    record = decision.to_record()
    assert tuple(record) == GATE_RECORD_FIELDS
    assert record["decision"] == decision.outcome.value
    assert record["reason"] == decision.reason.value
    assert record["detail"]
    assert record["margin"] == decision.gate.margin
    assert record["overfit_ratio"] == decision.gate.overfit_ratio
    assert record["overfit_check"] is decision.gate.overfit_check


def test_records_carry_the_comparison_and_explicit_nulls():
    decisions = _decisions_by_path()
    promote = decisions["promote"].to_record()
    assert promote["best_checkpoint_id"] == ITER_200 and promote["best_val_loss"] == 1.866
    assert promote["final_checkpoint_id"] == ITER_400 and promote["final_val_loss"] == 2.688
    assert promote["baseline_source"] == "base" and promote["baseline_val_loss"] == 8.977
    assert promote["required_below"] == margin_threshold(8.977, 0.02)
    assert promote["ratio"] == pytest.approx(1.4405, abs=1e-4)
    assert promote["val_set_digest"] == VAL.digest and promote["val_set_examples"] == 75
    assert promote["val_set_batch_size"] == 4
    assert promote["base_model"] == BASE and promote["baseline_changed"] is None
    assert promote["promoted_checkpoint_id"] == ITER_200
    assert promote["checkpoints"] == [
        {"checkpoint_id": ITER_200, "iteration": 200, "val_loss": 1.866, "error": None},
        {"checkpoint_id": ITER_400, "iteration": 400, "val_loss": 2.688, "error": None},
    ]

    inconclusive = decisions["inconclusive"].to_record()
    for field in (
        "best_checkpoint_id", "best_val_loss", "final_checkpoint_id", "final_val_loss",
        "baseline_source", "baseline_val_loss", "baseline_error", "baseline_changed",
        "required_below", "ratio", "val_set_digest", "val_set_examples", "val_set_batch_size",
        "promoted_checkpoint_id",
    ):
        assert field in inconclusive and inconclusive[field] is None


@pytest.mark.parametrize("path", ["promote", "reject-overfit", "reject-margin", "inconclusive"])
def test_a_record_gives_back_the_inputs_its_gate_run_scored(path):
    decision = _decisions_by_path()[path]
    checkpoints, baseline, val_set, base_model = recorded_inputs(decision.to_record())
    assert checkpoints == decision.checkpoints
    assert baseline == decision.baseline
    assert val_set == decision.val_set
    assert base_model == decision.base_model == BASE


def test_a_record_keeps_failed_and_non_finite_scores():
    checkpoints = (Checkpoint(ITER_200, 200, math.inf), Checkpoint(ITER_400, 400, None, error="boom"))
    record = FidelityGate().evaluate(
        checkpoints=checkpoints, baseline=Baseline("base", error="lost"), val_set=VAL, base_model=BASE,
    ).to_record()
    recorded, baseline, _, _ = recorded_inputs(record)
    assert recorded[0].val_loss == math.inf
    assert recorded[1] == Checkpoint(ITER_400, 400, None, error="boom")
    assert baseline == Baseline("base", error="lost")


@pytest.mark.parametrize("damage", [
    {"checkpoints": None},
    {"checkpoints": [{"checkpoint_id": ITER_200}]},
    {"checkpoints": [{"checkpoint_id": ITER_200, "iteration": 200, "val_loss": "low"}]},
    {"checkpoints": [{"checkpoint_id": "../served/adapters.safetensors", "iteration": 200, "val_loss": 1.0}]},
    {"baseline_val_loss": [8.0]},
    {"baseline_changed": 1},
    {"val_set_digest": None},
    {"val_set_batch_size": None},
    {"val_set_batch_size": True},
    {"base_model": ["gemma"]},
])
def test_a_malformed_record_is_refused(damage):
    checkpoints, baseline = _june()
    record = FidelityGate().evaluate(checkpoints=checkpoints, baseline=baseline, val_set=VAL, base_model=BASE).to_record()
    with pytest.raises(ValueError):
        recorded_inputs({**record, **damage})


# ----------------------------------------------------------------------
# Config (gate.margin / gate.overfit_ratio / gate.overfit_check)
# ----------------------------------------------------------------------

def test_pinned_defaults():
    assert FidelityGate().to_config() == {"margin": 0.02, "overfit_ratio": 1.20, "overfit_check": True}


def test_from_config_overrides_only_what_it_names():
    gate = FidelityGate.from_config({"margin": 0.05}, base=FidelityGate(overfit_ratio=1.3))
    assert gate == FidelityGate(margin=0.05, overfit_ratio=1.3)


@pytest.mark.parametrize("config", [
    {"margin": -0.01}, {"margin": 1.0}, {"margin": math.nan}, {"margin": "0.02"}, {"margin": True},
    {"overfit_ratio": 0.99}, {"overfit_ratio": math.inf}, {"overfit_ratio": None},
    {"overfit_check": "false"}, {"overfit_check": 0},
    {"max_val_loss": 3.0},
])
def test_invalid_gate_config_is_refused(config):
    with pytest.raises(ValueError):
        FidelityGate.from_config(config)


def test_gate_config_must_be_a_mapping():
    with pytest.raises(ValueError):
        FidelityGate.from_config(0.02)
