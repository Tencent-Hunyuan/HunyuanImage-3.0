# -*- coding: utf-8 -*-
"""Regression tests for hunyuan_image_3.cache_utils.CacheWithFreqsContainer."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hunyuan_image_3.cache_utils import CacheWithFreqsContainer  # noqa: E402


def _run_steps(container, low_freqs_order, high_freqs_order, steps=5):
    """Drive full computations the way modeling_hunyuan_image_3.py does."""
    for step in range(steps):
        x = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1) * (step + 1)
        container.derivatives_computation(
            x,
            distance=1,
            low_freqs_order=low_freqs_order,
            high_freqs_order=high_freqs_order,
        )


def test_low_freqs_orders_survive_a_shorter_high_freqs_chain():
    """A shorter high-freq chain must not truncate the low-freq derivatives.

    ``low_freqs_order`` and ``high_freqs_order`` are independent knobs
    (--taylor-cache-low-freqs-order / --taylor-cache-high-freqs-order), so
    computing fewer high-freq orders must not discard low-freq derivatives
    that derivatives_computation() already produced.
    """
    container = CacheWithFreqsContainer(max_order=3)
    _run_steps(container, low_freqs_order=3, high_freqs_order=1)

    for order in range(4):
        assert container.get_derivative(order, "low_freqs") is not None, (
            f"low_freqs derivative of order {order} was computed but discarded"
        )
    assert len(container.get_all_filled_derivatives("low_freqs")) == 4
    assert len(container.get_all_filled_derivatives("high_freqs")) == 2


def test_high_freqs_orders_survive_a_shorter_low_freqs_chain():
    """The symmetric case must keep working as before."""
    container = CacheWithFreqsContainer(max_order=3)
    _run_steps(container, low_freqs_order=1, high_freqs_order=3)

    assert len(container.get_all_filled_derivatives("low_freqs")) == 2
    assert len(container.get_all_filled_derivatives("high_freqs")) == 4


def test_equal_orders_are_unchanged():
    """The default configuration keeps every order of both chains."""
    container = CacheWithFreqsContainer(max_order=2)
    _run_steps(container, low_freqs_order=2, high_freqs_order=2)

    assert len(container.get_all_filled_derivatives("low_freqs")) == 3
    assert len(container.get_all_filled_derivatives("high_freqs")) == 3
