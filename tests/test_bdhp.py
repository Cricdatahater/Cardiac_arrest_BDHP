import numpy as np

from src.bdhp import simulate_bdhp


def test_outputs_are_valid_and_reproducible():
    first = simulate_bdhp(180, -6, 150, 1.1, 20, seed=42)
    second = simulate_bdhp(180, -6, 150, 1.1, 20, seed=42)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert set(np.unique(first[0])).issubset({0, 1})
    assert np.all((first[1] >= 0) & (first[1] <= 1))


def test_no_excitation_has_constant_probability():
    _, probabilities = simulate_bdhp(20, -2, 0, 1.1, 20, seed=1)
    assert np.allclose(probabilities, probabilities[0])

