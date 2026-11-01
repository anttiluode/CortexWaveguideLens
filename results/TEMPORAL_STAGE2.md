# Temporal phase-copy study — 2026-11-01

## Status and claim boundary

**Complete local CPU experiment, not a learned wave medium.** This stage tests the third component of the CortexWaveguideLens idea: carrying visual evidence across time with phase-corrected translation and an uncertainty-aware update. The data are artificial, globally translated, and periodic. The baseline that matters most—**ordinary motion-aligned per-pixel averaging with the SAME causal reset detector**—matches the zero-process-noise phase-copy estimator to numerical precision. There is **no claim of a new or superior denoiser**.

## Registration and design

- A deterministic 48×48 scene is composed of locally oriented, blurred rectangle edges and random spatial sinusoids. Every episode uses a fresh seeded scene; no actual images or neurons.
- Each episode lasts 24 frames. The full scene shifts each step by one fixed integer velocity selected from `(0,1), (1,0), (1,-1), (-1,2)`, with periodic wraparound.
- Independent Gaussian camera noise with standard deviation 0.55 is added in the main test. Large vertical occluders conceal 17 columns during frames 7–10 and 17–20. The scene is **entirely replaced** at frame 13, when it is visible, creating an unambiguous violation of the old memory.
- Methods are tested on MSE against the unobserved clean scene, including behind the occluder. The mask and sensor noise variance are available to the estimator. These are generous simplifying assumptions.
- For the principal *learned?* question: nothing learns here. Phase propagation is an exact Fourier translation. The velocity is estimated causally from frames 0 and 1 by a bounded template search (25 integer shifts), then held fixed.
- Hyperparameters were selected on 12 validation seeds, 100–111, **before** test seeds 200–279 were examined. `tune_temporal.py` records the tested candidates. Validation selected process noise `q=0.0`, innovation reset threshold `1.3`, EMA factor `.30`, and Gaussian blur width `.60`. Test data were not used for selection.
- At each step, compare the new observation against the propagated prediction. If the observed normalized squared residual exceeds 1.3, invalidate the entire previous scene estimate and reset from the new measurement. This is a deliberately simple **global change alarm**, not a robust local change detector.

## Dynamics

For a spatial Fourier mode \(k=(k_y,k_x)\) and an estimated shift \(v=(v_y,v_x)\):

\[\widetilde{z}_{t,k}=e^{-2\pi i(k\cdot v)}z_{t-1,k}.\]

The prior mean is obtained by an inverse transform. For observed pixels, a diagonal-variance update is:

\[K_t=\frac{P_t}{P_t+R}, \qquad z_t=\widetilde z_t+K_t(y_t-\widetilde z_t),\]

and unobserved pixels retain their propagated prediction. Here \(R=\sigma^2\) is **known** observation noise variance and \(P_t\) is the propagated state uncertainty plus an optional process noise \(q\). In the chosen setting, \(q=0\); the filter is essentially a causal, motion-aligned running average, up to numerical precision. Reset logic is shared with comparison methods. Importantly, the exponential factor is exactly a *translation*; a Schrödinger-like dispersion factor \(e^{-i\alpha k^2 t}\) would not align a translating scene.

## Held-out main result (80 independent episodes)

All values are mean squared error; **lower is better**.

| Method | All pixels | Occluded pixels | Frames 13–16 after replacement |
| --- | ---: | ---: | ---: |
| Raw noisy image (zeros behind mask) | 0.3819 | 0.9746 | 0.3026 |
| Per-frame Gaussian smoothing | 0.1898 | 0.9062 | **0.0940** |
| Unaligned temporal EMA | 0.5190 | 1.0711 | 0.5164 |
| Motion-aligned EMA (no reset) | 0.1596 | 0.1084 | 0.5025 |
| Motion-aligned EMA, **same change alarm** | 0.0959 | 0.0689 | 0.1681 |
| **Classical motion-aligned running mean, same alarm** | **0.0816** | **0.0581** | 0.1572 |
| **Fourier phase-copy state, inferred velocity, same alarm** | **0.0816** | **0.0581** | 0.1572 |
| Fourier phase-copy, no reset | 0.5123 | 0.5478 | 1.3831 |
| Fourier phase-copy, wrong motion (0,0) | 0.3731 | 0.9645 | 0.2964 |

The zero-process-noise phase state and classical counted-average outputs coincide numerically (unit test). The inferred velocity was correct in all 80 main synthetic episodes; this is an easy translation-estimation setup, **not** demonstrated robustness to VHS time-base error or unknown complex motion. The global change detector fired once at frame 13 for all 80 changed-scene episodes, and did not fire in 30 separately generated non-changing test episodes.

**Important tradeoff:** Per-frame smoothing scores better immediately after replacement, since it does not retain the previous scene, but fails to reconstruct occluded content. This is why a single whole-video MSE is not a sufficient measure of memory quality.

## Sensitivity checks (fresh test episodes)

| Noise SD | Episodes | Raw overall MSE | Matched classical running mean | Phase-copy with inferred motion | Wrong motion |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.35 | 40 | 0.2257 | 0.0332 | 0.0332 | 0.2254 |
| 0.55 | 80 | 0.3819 | 0.0816 | 0.0816 | 0.3731 |
| 0.85 | 40 | 0.7537 | 0.1948 | 0.1948 | 0.6662 |

For the 30 *no-replacement* episodes (noise SD 0.55), the phase-copy MSE was 0.0495 and the change detector emitted **zero** resets. All tuning choices were retained from validation. The independent seeds here are 400–439, 500–539, and 600–629 for the extra conditions.

## Interpretation

The **positive** result is evidence accumulation: the prior scene can be translated into the present, retain detail behind an occluder, and use incoming evidence to correct noise. The **negative** result is just as important: the mechanism is an alternative representation of known motion-compensated averaging; it adds no information, and without a scene-change alarm it retains catastrophic ghost images. State needs both persistence and an expiry/correction rule. A simple phase term alone does not solve motion estimation, nonrigid deformation, or phase corruption.

The next scientific question is whether a *learned, spatially structured and off-diagonally coupled* state transition gives an advantage on non-rigid motion or partial frequency observations against a compute-matched state-space model, classical flow-guided temporal denoiser, and no-memory ablation. A small model of this sort would need a trainable transition and uncertainty calibration, plus held-out deformations and real-world data. Keeping the Sihti6 coordinated-atom experiment as an independent study is the right boundary.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tune_temporal.py  # validation search, 12 seeds
python temporal_wavecopy.py --count 80 --q 0 --reset-threshold 1.3 --alpha .30 --blur .6 --output results/temporal_main
python temporal_wavecopy.py --count 40 --test-start 400 --q 0 --reset-threshold 1.3 --alpha .30 --blur .6 --noise .35 --output results/temporal_lownoise
python temporal_wavecopy.py --count 40 --test-start 500 --q 0 --reset-threshold 1.3 --alpha .30 --blur .6 --noise .85 --output results/temporal_highnoise
python temporal_wavecopy.py --count 30 --test-start 600 --q 0 --reset-threshold 1.3 --alpha .30 --blur .6 --noise .55 --no-switch --output results/temporal_noswitch
```

The JSON summaries and per-episode CSVs are generated alongside the outputs. The test suite now has 11 passing tests including causal invariance and arithmetic equivalence checks.
