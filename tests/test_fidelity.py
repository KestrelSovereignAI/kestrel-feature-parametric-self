"""Tests for the fidelity gate + val-loss parser (pure, no MLX)."""

from __future__ import annotations

import pytest

from kestrel_feature_parametric_self import FidelityGate, parse_final_val_loss


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


def test_gate_promotes_within_bounds():
    gate = FidelityGate(max_val_loss=3.0, max_regression=0.25)
    d = gate.evaluate(2.2, prior_val_loss=None)
    assert d.promote is True
    assert d.val_loss == 2.2


def test_gate_rejects_above_ceiling():
    gate = FidelityGate(max_val_loss=3.0)
    d = gate.evaluate(3.5)
    assert d.promote is False
    assert "ceiling" in d.reason


def test_gate_rejects_regression():
    gate = FidelityGate(max_val_loss=10.0, max_regression=0.25)
    d = gate.evaluate(2.6, prior_val_loss=2.2)  # +0.4 > 0.25 allowed
    assert d.promote is False
    assert "regress" in d.reason


def test_gate_allows_small_regression_within_tolerance():
    gate = FidelityGate(max_val_loss=10.0, max_regression=0.25)
    d = gate.evaluate(2.4, prior_val_loss=2.2)  # +0.2 <= 0.25
    assert d.promote is True


def test_gate_rejects_unverifiable():
    gate = FidelityGate()
    d = gate.evaluate(None)
    assert d.promote is False
    assert "cannot verify" in d.reason
