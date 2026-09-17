"""Bernoulli discrete Hawkes process with a logistic link."""

from __future__ import annotations

import numpy as np


def simulate_bdhp(n_steps: int, baseline_logit: float, excitation: float, decay: float,
                  shift: float, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulate binary events and their conditional probabilities."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if excitation < 0 or decay <= 0 or shift < 0:
        raise ValueError("excitation and shift must be non-negative; decay must be positive")
    rng = np.random.default_rng(seed)
    events = np.zeros(n_steps, dtype=int)
    probabilities = np.zeros(n_steps, dtype=float)
    lags = np.arange(1, n_steps + 1, dtype=float)
    weights = excitation / np.power(lags + shift, decay)
    for t in range(n_steps):
        history = 0.0 if t == 0 else float(events[t - 1 :: -1] @ weights[:t])
        logit = np.clip(baseline_logit + history, -709.0, 709.0)
        probability = 1.0 / (1.0 + np.exp(-logit))
        probabilities[t] = probability
        events[t] = rng.binomial(1, probability)
    return events, probabilities

