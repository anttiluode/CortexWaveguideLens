# CortexWaveguideLens

**A falsifiable signal-processing experiment inspired by dendritic filtering, apical feedback, and phase-aligned frequency channels.**

This is **not** a biophysical neuron simulator, a claim that dendrites transmit lossless waves, or a working general-purpose vision model. Its first experiment asks a narrower question: *can earlier environmental context dynamically reweight a phase-sensitive filter bank to recognize a remembered signal under unknown translation and changing spectral noise?*

## Why this experiment

The Sihti / SighImageSuper / VHSImageStitch work suggests keeping multiple filtered views instead of collapsing them into one enhanced image. TransformerX suggests that **the state carried from earlier observations can alter current computation**. The DaFuc / "chocolate-wafer" idea suggests eventually splitting components when one component can no longer represent distinct contexts.

The missing causal test was whether a **contextual branch gain** does more than a fixed filter or an extra context-dependent output bias. This repository starts with that test and leaves dendritic growth for later.

## Mechanism

Each 64-sample real signal contains six complex Fourier components at integer frequencies `3, 5, 7, 11, 14, 19`. The model receives:

- a **reference** waveform stored from earlier;
- an independent **noise-only calibration probe** measured before the query (eight observations per frequency); and
- a **query** waveform that might be a translated version of the reference or an unrelated pattern with matching clean Fourier magnitudes.

For each band, it forms a phase-sensitive comparison

```text
p_b = z_query,b * conjugate(z_ref,b) / (|z_query,b| |z_ref,b| + eps)
```

and scores the possible spatial translations `d = 0..63`:

```text
S = max_d sum_b softmax(router(noise_probe))_b * Re[p_b exp(+i 2*pi*f_b*d/64)]
```

A small trainable router (`6 -> 16 -> 6`) learns **which branches to trust** from the earlier noise observation. A final trainable scalar slope/bias maps the best score to a probability. During inference, the true translation, label, noise class, and generator parameters are never supplied to the lens.

This is a **frequency-domain analogy** to an apical influence on branch contributions, *not* evidence of phase-conjugate mirrors in neural tissue. The phase-delay scan is classical phase-only correlation written explicitly as branch channels (verified by a test). The **learned, time-dependent per-band gain** is the element being investigated.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python waveguide_lens.py --seeds 3 17 27 --steps 350 --output results/main.json
# held-out noise environments, still trained at 0.85
python waveguide_lens.py --test-noise-scale 1.25 --output results/noisier.json
```

CPU is sufficient; the script uses reproducible NumPy and Torch seeds. Set `OMP_NUM_THREADS=1` for small CPU runs if desired. The main run trains on translations `0..31` and tests on unseen translations `32..63`. It uses 2,500 training and 1,500 test examples per seed, with binary classes balanced in expectation.

## Baselines and controls

| Method | Causal question |
| --- | --- |
| Magnitude-only MLP | Is there class information without phase? |
| Zero-delay phase comparison | Does phase comparison work without delay alignment? |
| Shuffled reference | Is the earlier reference actually necessary? |
| Uniform phase scan | What does classical phase correlation give without tuning? |
| Learned fixed gains | Does learning global weights suffice? |
| Context bias only | Is simply giving the network probe information sufficient? |
| Adaptive gains ('apical probe') | Does the earlier probe help specifically by tuning branches? |
| Shuffled apical probe | Does tuning depend on correct *per-example* context? |
| Classical inverse-noise weighting | Does the learned lens outperform a simple, strong conventional method? |

**Crucial benchmark honesty:** on this purposely frequency-coded synthetic task, the classical noise-aware model currently wins. The experiment demonstrates that the trainable router uses prior context, not a new superior phase-recovery technique.

See [results/README.md](results/README.md) for numbers and limitations. `results/accuracy.csv` stores the per-seed experiment data.

## What is not established

1. The test signals are global sums of six Fourier modes, **not natural images**. No edge coherence, occlusions, non-rigid VHS scanline distortion, learned Gabor atoms, or real-time camera operation is evaluated here.
2. The calibration probe is a deliberately useful independent observation. Its noise statistics are stable into the query, by construction. The positive result depends on that assumption.
3. The router changes **gains**, not cable morphology, physical path lengths, membrane conductances, or the complex phases themselves. The delay bank is fixed and enumerated.
4. There is **no online weight plasticity, structural growth, or claim of a new biological mechanism**.
5. For this task, even a simple inverse-noise weighting strategy does better. Any next-stage learned model must beat that control, not merely a fixed-gain lens.

## Next experiments

1. Replace periodic waves with **localized 2-D structures** (edges, corners, textures), plus moving occlusions; explicitly test cross-scale phase alignment.
2. Estimate spectral reliability from **past raw video frames**, not a privileged noise-only probe, and preserve causal temporal ordering (TransformerX-style state).
3. Compare learned context gates to **classical noise-aware phase correlation**, regularized cross-power spectra, and a compute-matched CNN/attention baseline.
4. Only after that, allow **branch splitting** if a held-out improvement exceeds a complexity penalty. Test split vs no split on repeated conflicting contexts, with newly created weights registered in the optimizer.

## Relevant research (inspiration, not validation of this model)

- Morrone & Owens, *Feature detection from local energy* (1987): phase congruence across harmonics, [DOI](https://doi.org/10.1016/0167-8655(87)90013-4).
- Rall's equivalent-cylinder conditions involve membrane properties, endpoint conditions, and equal electrotonic distances **in addition to** the `3/2` branching relation; see [discussion](https://pubmed.ncbi.nlm.nih.gov/5007242/).
- For the neural analogy, local apical modulation and dendritic nonlinearity motivate **computational hypotheses**; they do not license treating passive cables as lossless traveling-wave guides.

MIT license.
