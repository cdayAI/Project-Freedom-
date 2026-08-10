"""Generate the golden reference values pinned in tests/test_dsr.py and
tests/test_pbo.py. Run once; paste outputs. Any future port must reproduce
these to the stated tolerance before replacing the Python reference."""

import numpy as np

from alpha_forge.gates.dsr import deflated_sharpe_ratio
from alpha_forge.gates.pbo import cscv_pbo

rng = np.random.default_rng(42)
r = rng.normal(0.0008, 0.012, 756)
out = deflated_sharpe_ratio(r, n_trials=50, var_trial_sr=0.01)
print("DSR GOLDEN:")
for k in ("sr_per_period", "sr0_benchmark", "dsr_probability"):
    print(f'    "{k}": {out[k]!r},')

rng = np.random.default_rng(123)
m = rng.normal(0.0002, 0.01, size=(320, 16))
out = cscv_pbo(m)
print("PBO GOLDEN:")
print(f'    "pbo": {out["pbo"]!r},')
print(f'    "lambda_mean": {out["lambda_mean"]!r},')
