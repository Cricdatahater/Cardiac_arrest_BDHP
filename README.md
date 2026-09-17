# Bernoulli Discrete Hawkes Process Simulation

A compact simulation study of a self-exciting binary event process with a logistic link and shifted power-law (Omori–Utsu-style) memory kernel.

> **Scope:** this is a methodological simulation, not a clinical prediction model. It uses synthetic events and must not be interpreted as estimating patient risk or supporting medical decisions.

## Model

For binary event \(Y_t\), the conditional event probability is

\[
p_t = \operatorname{logit}^{-1}\left(\mu + \sum_{i<t}Y_i\frac{\alpha}{(t-i+c)^\beta}\right).
\]

`mu` controls baseline log-odds, `alpha` the excitation magnitude, `beta` the decay rate, and `c` the near-event shift. The implementation clips logits before applying the logistic function for numerical stability.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
jupyter lab BDHP_sim.ipynb
```

```python
from src.bdhp import simulate_bdhp

events, probabilities = simulate_bdhp(
    n_steps=180, baseline_logit=-6, excitation=150,
    decay=1.1, shift=20, seed=42,
)
```

## What was improved

- reusable simulation code with explicit parameter validation;
- local random-number generation instead of global state;
- tests for reproducibility, binary outputs, and valid probabilities;
- documentation that separates synthetic-method demonstration from clinical claims.

The notebook currently conditions its displayed example by repeatedly simulating until exactly one event occurs. That is useful for illustration but changes the sampling distribution. Unconditional simulations from `simulate_bdhp` should be used for statistical summaries.

## Structure

```text
.
├── BDHP_NOTES.pdf       # project notes
├── BDHP_sim.ipynb       # original exploratory simulation
├── src/bdhp.py          # tested simulator
├── tests/test_bdhp.py
└── requirements.txt
```

## Research extensions

- sensitivity analysis over all kernel parameters;
- calibration and out-of-sample validation on an ethically sourced dataset;
- comparison with non-self-exciting Bernoulli baselines;
- uncertainty intervals and posterior predictive checks.

