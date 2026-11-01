# Experimental ledger — 2026-11-01

**Status:** A minimal, deterministic synthetic benchmark ran locally on CPU with Python 3.13, NumPy 2.3.5 and Torch 2.10.0+cpu. Five unit tests passed. This is not a real-image benchmark.

## Main results

Train noise scale 0.85; test noise scale 0.85. Train translations `0..31`, **unseen** test translations `32..63`. 2,500 train / 1,500 test examples per seed, 350 optimization steps for learned models, seeds 3, 17, 27. Scores are mean ± sample SD across three independent test sets.

| Method | Accuracy |
| --- | ---: |
| Magnitude-only MLP | 50.58% ± 1.10% |
| Zero-delay phase comparison | 50.40% ± 1.14% |
| Shuffled reference | 51.04% ± 0.76% |
| Uniform phase scan | 71.56% ± 0.23% |
| Learned fixed gains | 71.53% ± 0.93% |
| Probe available only as output bias | 72.44% ± 1.17% |
| **Learned context-to-channel gains** | **84.96% ± 0.60%** |
| Shuffled probe with learned gains | 71.67% ± 0.64% |
| **Classical inverse-noise channel gains** | **93.93% ± 0.57%** |

Learned branch gains respond in the expected direction: when low-frequency channels are corrupted, their combined gain is small and cleaner high-frequency channels dominate. When high-frequency channels are corrupted, the pattern reverses. This is actual conditional routing, rather than a network whose prediction merely uses the probe to shift the final threshold.

## Shifted noise — same trained model configuration, changed test noise only

The models are retrained for each run with **training** noise scale 0.85. Only the test noise is changed. Values are the mean of the same three seeds.

| Method | Test noise 0.50 | Test noise 0.85 | Test noise 1.25 |
| --- | ---: | ---: | ---: |
| Uniform phase scan | 83.18% | 71.56% | 63.76% |
| Fixed gains | 80.58% | 71.53% | 64.53% |
| Bias-only probe | 82.47% | 72.44% | 64.93% |
| Learned adaptive gains | 88.98% | 84.96% | 79.67% |
| Shuffled probe | 80.64% | 71.67% | 67.33% |
| Classical inverse-noise gains | **94.31%** | **93.93%** | **90.89%** |

The improvement persists under both of these test-noise changes, **but does not close the gap with a classical estimator**.

## What this does and does not mean

- Positive: prior environmental evidence can guide gain selection of multiple phase-sensitive channels, on held-out translations, in this synthetic setting.
- Negative: merely allowing a probe to change the output bias is insufficient; learned context-to-band gains helped substantially more.
- Negative: the established inverse-noise rule is better than the learned gate. The result is **not** evidence of novel signal-processing performance.
- Scope: perfect periodic boundaries, six known frequencies, known availability of a clean earlier reference and a noise-only probe, additive independent noise, no naturalistic motion or clutter.
- A standard phase correlation computation can reproduce the fixed phase-delay scan. The test suite checks this equivalence explicitly.
- These few seeds and one designed task do not establish generalization to real cortical processing, videos, or world models.

Raw per-seed accuracy values are in [accuracy.csv](accuracy.csv). Re-run from the project root to generate full JSON including learned gains, Brier score, seed metadata and training/test parameters.
