# HEKATE state briefing

Handoff for a new AI. Read this before proposing work. The two older docs are plans, not a description of the tree.

- `Docs/implementation.md` is the physics-chain spec. Much of it is now code. The file layout it names (`si_phmpy/physics/`) was never created.
- `Docs/morbius_ofc2027_plan.md` is an OFC 2027 execution plan dated 25 July 2026, with an Optica abstract deadline of 20 October 2026. Today is 28 October 2026. That plan's file layout (`morbius/photonics/`, link budget, noise, dataplane, fuzzy extractor, RTL) was never created. Do not treat the deadline calendar inside that file as current.

Repo: `github.com/MOAMMAH/HEKATE`, branch `main`, one commit (`a5c5d9d`, 7 Aug 2026, "Initial commit: HEKATE modulator and PUF models"). Everything after that commit is uncommitted local work, including the multimode model. Do not commit unless asked.

---

## What the project is

HEKATE models **MORBIUS**: a silicon-photonic physical unclonable function (PUF) built from a contra-directional coupler (contra-DC), interrogated by a WDM carrier grid.

A contra-DC is two dissimilar strip waveguides with a periodic corrugation between them. Light launched forward in waveguide 1 couples into the backward wave of waveguide 2 only near phase match. Fabrication disorder (index, coupling, grating period) is the secret. The device is passive, linear, and time-invariant. Its whole behavior is one complex reflection spectrum `r(ω)`, or the impulse response `h(t) = IFFT{r}`.

The node has two optical modes, switched by an MZI. Data mode carries traffic. Auth mode routes a challenge through the contra-DC and reads a fingerprint. The digital side (fuzzy extractor, BCH, SHA-256, HMAC, AES) is specified as future work. It is not in this repo. Security cannot rest on "the map is hard to model": a linear filter is learnable from enough challenge-response pairs. The argument is that raw responses stay inside the node enclave.

The simulator exists to answer one question:

**Do the challenge bits change the fingerprint, or is the fingerprint only `|r|²` at the carrier wavelengths?**

If the symbol period `T` is much longer than the grating delay spread `τ_max = 2 · n_g · L / c`, each symbol settles before the next arrives. The sample is the bit times `|r(ω_i)|²`. The challenge is inert. Pattern dependence appears only when `T` is comparable to `τ_max`, so earlier symbols are still ringing in the grating. The spec calls the ISI energy fraction `eta`. `eta` is not implemented. The readout script uses a transient-versus-steady-state comparison as a stand-in.

---

## Signal chain that exists

```mermaid
flowchart LR
  bits["challenge bits per channel"] --> mod["Modulator"]
  mod --> x["complex baseband x(t)"]
  x --> puf["ContraDCPUF r of omega"]
  puf --> y["reflected field y(t)"]
  y --> demux["Lorentzian demux 30 GHz"]
  demux --> sq["square law power"]
  sq --> adc["4 samples per symbol, 4-bit Gray"]
  adc --> key["key, shape channels by samples"]
```

Units are SI inside the kernels: meters, seconds, hertz, field in `sqrt(W)` so `|E|²` is watts. One complex baseband, reference `f0 = c / 1550 nm`. Channels are frequency offsets inside that baseband. `SimGrid` default is `fs = 4e12` Hz (`dt = 0.25 ps`), `n = 16384`.

There is no photodetector noise, no TIA, no ENOB, no fiber link, and no data-plane BER model. The key difference you see in the plots is noiseless.

---

## Actual tree

```
puf/puf_model.py          ContraDCPUF, SimGrid. This is the device.
puf/modes.py              EIM slab solver and coupling matrices. Uncommitted.
puf/model_decision.py     Old design-sweep script. Not the production model.
modulator/modulator_model.py   WDMGrid, Modulator, ModulatedField
modulator/send_to_puf.py  Minimal receiver and the A/B readout demo
modulator/plot_hamming_distance.py   One-pair inter-challenge HD, single vs 3-mode
modulator/out/            PNG figures only. No CSV.
tests/puf_test.py         Physics gate for ContraDCPUF. No modulator or chain tests.
requirements.txt          numpy, scipy, matplotlib
```

`ContraDCPuf` is an alias of `ContraDCPUF`.

---

## Device model (`puf/puf_model.py`)

4×4 coupled-mode transfer matrix, two waveguides, forward and backward in each. Section matrices are vectorized over wavelength and cascaded with `expm`. Boundary conditions: unit launch into waveguide 1 at `z = 0`, backward amplitudes zero at `z = L`. The drop port is the backward field in waveguide 2 at `z = 0` (`B2`), returned as a complex amplitude. It is not peak-normalized. Four-port power is available from `port_amplitudes` and is what the passivity tests check.

Disorder is seeded by `device_id` (`numpy` Generator). Three spatially correlated profiles, white noise smoothed by a Gaussian of width `corr_length` then rescaled:

| Profile | Applied to | Default sigma |
|---|---|---|
| `dn1`, `dn2` | effective index per section | `2e-3` |
| `dkappa` | relative, multiplies `k12` | `0.05` |
| `dLambda` | grating period | `0.5 nm` |

Correlation length default `2 µm`. Anti-apodization (`center_dip=0.90`, `edge_boost=2.0`) is the deterministic backbone. Disorder multiplies on top. These sigmas are suggested values from the spec. They are not cited to a PDK or a paper. Do not present them as measured.

Temperature: `dn/dT = 1.8e-4 /K`, uniform, about `0.08 nm/K`. It is part of the spectrum cache key and can be changed after construction. Other geometry changes need `clear_caches()`.

`impulse_response` uses `np.fft.fft(r) * (fs / n)`, not `ifft`. Tier-3 causality tests pass under that convention. Do not flip the FFT direction without re-running `tests/puf_test.py`.

Class defaults (the object you get from `ContraDCPUF()`):

| Parameter | Default |
|---|---|
| `L` | `210 µm` |
| `n_sect` | `80` (`dz ≈ 2.6 µm`) |
| `Lambda0` | `305.72 nm`, so `(n1+n2)·Λ ≈ 1550 nm` |
| `n1_0`, `n2_0` | `2.580`, `2.490` |
| `ng1`, `ng2` | `3.65`, `3.55` |
| `kappa12` | `16300 /m` |
| `kappa_mult` | `3.0` |
| `loss` | `5 dB/cm` |
| `mode_model` | `"single"` |

`κ12 · κ_mult · L` at these defaults is about `10`. Peak reflectivity is saturated. A flat stopband has little spectral structure and little entropy. The spec already says stronger coupling is not better.

`τ_max` at `L = 210 µm` is about **5 ps**. At 50 GBd, `T = 20 ps`. On the class-default geometry the challenge is expected to be nearly inert. That is a finding, not a bug.

### Multimode (uncommitted, opt-in)

`mode_model`:

| Value | Behavior |
|---|---|
| `"single"` | `M1 = M2 = 1`, legacy 4×4 path. Default. |
| `"auto"` | Mode count from widths `w1`, `w2` via `puf/modes.py` (220 nm SOI, vertical effective index, then a lateral slab). |
| `"fixed"` | Cap at `n_modes_max`. Widths if given, otherwise a synthetic `n_eff` ladder. |

State size is `d = 2·(M1+M2)`. Coupling matrices `K11`, `K12`, `K22` come from facing-sidewall field products, scaled so `K12[0,0] = kappa12`. Each mode pair phase-matches at `λ_ij = Λ·(n1_i + n2_j)`, so one grating has several stopbands. The receiver sums mode powers on one photodiode per wavelength (`reflection_power_spectrum`). The key array shape does not grow with mode count.

`force_general_cmt=True` runs the general solver at `M = 1`. A test requires it to match the legacy 4×4 path.

Ill-conditioned multimode BVP falls back to `pinv`. Legacy 4×4 still raises on a singular solve.

### Tests that gate the device

`tests/puf_test.py`, six tiers plus multimode:

0. Determinism, distinct `device_id`, finite spectrum, `n_sect` convergence under 1% on a uniform grating, frequency-sampling convergence of `h(t)`.
1. Four-port passivity, lossless power sums to 1, loss is monotonic, zero coupling transmits `exp(-αL)` on the through port.
2. Bragg wavelength, `tanh(κL)²` peak reflectivity within 1%, stopband width within 10%, scaling of width with `κ` and length.
3. Causality, Parseval, no FFT wrap, delay spread versus the analytic penetration depth `2·n_g·tanh(κL)/(κ·c)`.
4. Disorder moves the spectrum monotonically, inter-device residual correlation is low, longer correlation length broadens spectral autocorrelation, feature count tracks `τ_eff · B` within about 2×, temperature shifts the peak by about `0.08 nm/K` and is reversible.
5. Four carriers at 1.2 nm spacing, centered on the Bragg peak, each sit above 10% of peak `|r|`.

Run with the repo `.venv` and `pytest tests/puf_test.py`. There are no tests for `Modulator` or `send_to_puf`.

### Do not use `puf/model_decision.py` as the model

It is an old parameter-sweep script. It peak-normalizes to dB, launches into both waveguides (`ka1=0.99`, `ka2=0.14`), uses `Lambda0 = 300 nm` and `wl0 = 1521 nm`, and adds chirp, sampled grating, π/2 phase shifts, and a cascade offset that `ContraDCPUF` does not implement. Its `__main__` writes `outputs/puf_design_exploration.png` and the docstring names a file that does not exist. Leave it unless someone asks to delete or port a sweep.

---

## Modulator (`modulator/modulator_model.py`)

`WDMGrid` default: start `1548 nm`, spacing `1.2 nm`, 4 channels. The channel plan is a system input. Retune the grating to the plan. Do not slide the plan onto an old Bragg peak.

`Modulator.modulate(challenge, simgrid, f0)` expects `challenge` shaped `(n_channels, n_bits)`, independent bits per channel.

Per channel:

1. Bits to symbols. NRZ is 1 bit/symbol. PAM4 is 2 bits/symbol, Gray by default.
2. Symbols to optical **power** in `[1/ER, 1]`. Default extinction is 6 dB. Field is `sqrt(power)`.
3. Rectangular hold on `SimGrid`, then a Gaussian low-pass at `0.75 · symbol_rate`. Filtering power is an approximation. The kernel is Gaussian so power cannot go negative.
4. MRM chirp, first cut, not a ring model: `φ = (α/2)·log(P/P_mean)`, `α = 3`.
5. Shift by the carrier offset and sum into one baseband field.

Launch power is normalized over the burst only, default total `p_launch = 1 mW` in the class, **overridden to 40 mW (10 mW per channel) in the readout scripts**. Idle tail is `p_idle = 1/ER`. The call raises if the burst does not fit `simgrid.n`, and if a carrier offset exceeds `fs/2`. It warns if samples per symbol drop below 16.

`check_carriers_in_stopband` compares each carrier to the **global** peak reflectivity. A local peak around the grid would pass a dark grid. Threshold is 10% of peak.

PAM4 doubles challenge bits per symbol. It does not raise delay-spread dependence and it does not raise device entropy. It costs noise margin. Do not use it in an entropy argument.

---

## Readout demo (`modulator/send_to_puf.py`)

This is the only end-to-end experiment. It is a script, not a library of metrics.

Pipeline: modulate, multiply by `r(f)` in the FFT domain (or, if multimode, filter each drop mode and sum powers, keeping the fundamental phase), Lorentzian demux with **FWHM = 30 GHz**, square-law, sample, 4-bit Gray quantize.

Sampling is front-loaded at fractions `(0.1, 0.25, 0.45, 0.75)` of the symbol, measured from the symbol edge. A second pass at `0.9` is the steady-state control. A CW (unmodulated, equal-split) launch is the carrier-only control: theory is `(p_launch / N) · |r(f_i)|²`. Any A/B key difference that survives in the steady-state or CW path is not pattern dependence.

Full-scale is frozen from challenge A's peak (`1.05×`) and reused for B. That is a one-shot demo calibration, not enrollment.

The script builds a **different device** than the class defaults:

| Knob | Readout value | Why |
|---|---|---|
| `L` | `2.0 mm` | `τ_max ≈ 48 ps`, longer than `T = 20 ps` at 50 GBd, so the challenge can matter |
| `Lambda0` | `312.5 nm` | Moves the Bragg peak to about `1584 nm`. Carriers sit on the band-edge shoulder instead of a saturated plateau. At `305.72 nm` the 2 mm device gives `|r|² ≈ 0.999` and flat. |
| `n_sect` | `200` single-mode, `80` multimode | Runtime. See the resolution warning below. |
| `symbol_rate` | `50e9` baud, NRZ | `T = 20 ps` |
| `p_launch` | `40 mW` total | `10 mW` per channel |
| challenges | 32 bits/channel, RNG seeds 0 and 1 | One pair, not a population |
| `device_id` | `0` | One instance |

Multimode toggle at the top of the file: `_MODE_MODEL`, `_W1 = 1.2 µm`, `_W2 = 1.0 µm`, `_N_MODES_MAX = 3`, `_GAP = 200 nm`. For multimode the script retunes `Λ` so the fundamental Bragg stays near 1550 nm, then applies the same fractional edge offset (`312.5 / 305.72`). The time grid drops to `n = 8192` because the general TMM is heavier.

**Section length on this device is too coarse for the disorder model.** `dz = L / n_sect` is `10 µm` (single-mode) or `25 µm` (3-mode). The disorder correlation length is `2 µm`, and the spec wants `dz < 1 µm` so `δ·dz` stays small across a wide band. The HD figures below were produced with under-resolved disorder. Do not treat them as a converged uniqueness result. Raising `n_sect` on a 2 mm device is expensive (`expm` per section per wavelength chunk).

A 30 GHz drop filter in front of a 50 GBd signal strips most of the modulation sidebands. Those sidebands are where pattern dependence lives. The FWHM is a hard ceiling. It has not been swept. Do not widen it to make Hamming distance look better.

### Figures in `modulator/out/`

Regenerated by the scripts. The old unsuffixed `key_difference.png`, `key_difference_bits.png`, and `puf_readout_summary.png` were deleted in the working tree and replaced by `_single` and `_auto_m3` versions.

`plot_hamming_distance.py` compares one challenge pair (same seeds, same grid, same `device_id`) under single-mode and 3-mode. Numbers read off the figure titles:

| Case | Challenge bit HD | Key bit HD | Mean flips per 4-bit sample |
|---|---|---|---|
| Single-mode, `M = 1`, `n = 16384`, `n_sect = 200` | `0.484` | `0.399` (`818/2048`) | `1.60` |
| 3-mode, `M1 = M2 = 3`, `n = 8192`, `n_sect = 80` | `0.484` | `0.446` (`913/2048`) | `1.78` |

2048 bits is `4 channels × 32 symbols × 4 offsets × 4 bits`. The spec's inter-challenge target is near 50%. These two numbers are one pair, noiseless, on an under-resolved grating, and the two rows do not share `n_sect` or `simgrid.n`. They show that challenge bits move the key. They are not an inter-PUF histogram and they are not an entropy estimate.

`send_to_puf.py` also writes `key_difference_*.png`, `key_difference_bits_*.png`, and `puf_readout_summary_*.png` (spectrum, CW versus theory, waveforms, both keys, transient versus steady-state difference). Printed RMS and CW-deviation numbers are not saved anywhere except a terminal. Re-run the script if you need them.

---

## What is not built

From `Docs/implementation.md`, still missing:

- `eta` (ISI energy ratio). The central design curve, `eta` versus bitrate for several `L`, does not exist.
- Photodetector: shot noise, thermal noise, TIA density, Bessel LPF. Noise is not reproducible because it is absent.
- `ADC` and `Fingerprint` as classes. Quantization lives as functions in `send_to_puf.py`. Output is a 2D Gray-code array, not a flat 2048-bit `Fingerprint` with metadata.
- `PUFChain`.
- Hamming-distance suite beyond one pair: inter-device (≥100 seeds), inter-grid (the spec cites a prior ~55% at `1.2 nm`, including a 64% spike at `1.0 nm`), intra-device versus temperature and versus noise. Target intra-device HD is under 10% so an ECC can absorb it. Unmeasured.
- Entropy: analytic degrees of freedom and PCA on raw samples. The spec's worked example at 4 channels, 50 GHz, `τ_max = 13 ps` is about 19 real numbers, on the order of **70 bits**, which does not support a 256-bit `K_PUF`. The suggested fix is about 8 channels and `L ≈ 2 mm` (`τ_max ≈ 52 ps`, ~50 complex degrees of freedom). That length is what the readout script uses. The entropy check itself has not been run. Redundant samples are not entropy.
- Stopband-versus-bitrate sweeps, drop-filter FWHM sweeps, CSV-then-plot experiment layout.

From the OFC plan, still missing: link budget, SINAD, ENOB (kill line was ENOB < 3), data-plane BER and the security-branch penalty, BCH sizing, helper-data storage, RTL. Decisions D1–D5 were never written into a README. Implicit choices in the code: analytic CMT (D1), the three disorder sigmas above (D2), no analog equalizer (D3 is open), alpha-parameter MRM (D4), no helper data (D5 is open).

There is no SimPy model in this repo. The spec says the physics package must not import one. Fingerprints would cross that boundary as saved arrays.

---

## Constraints for any next change

- Keep the physics chain and any future protocol/queue model in separate modules.
- Keep single-mode 4×4 as the default. Multimode stays opt-in.
- Vectorize over frequency, loop over sections. Cache spectra. A Python double loop will not survive a 100-device sweep.
- Zero disorder must keep working. It is the analytic check (`tanh(κL)²`, Bragg wavelength, passivity).
- Do not peak-normalize `r`. Do not turn a singular solve into a zero. Do not truncate a burst that does not fit the grid.
- Do not add cross-channel beat notes at the photodiode. At 1.2 nm the beat is about 150 GHz and both the demux and the receiver LPF remove it.
- Do not silently widen the 30 GHz demux, raise `kappa_mult`, or lengthen the device and then quote a higher Hamming distance as a security result. Say what changed.
- State the simulation basis and the parameter source. Unsourced numbers presented as measured is the failure mode called out in the OFC plan.
- The data-mode MZI, if it ever gets modeled, defaults to data mode when unpowered. Auth mode is the switched state.

---

## Open points worth starting from

1. **Is the 2 mm, band-edge device the one to keep?** It was chosen so `T < τ_max` and so carriers are not on a flat saturated stopband. `κL` is still enormous (`κ12·κ_mult·L ~ 100`). Entropy wants spectral features, which wants lower `κ` or a shorter interaction, which fights delay spread. Those two knobs have not been swept together.
2. **Section resolution.** HD figures used `dz` of 10–25 µm against a 2 µm correlation length. A converged `n_sect` (spec suggests 500–1000 on the short device; the long device needs `dz ≲ 1 µm`) will change both runtime and the numbers.
3. **`eta` before more Hamming plots.** One pair at 0.40 and 0.45 does not locate the design. The spec's first figure is `eta` versus bitrate for `L` in `{0.1, 0.5, 1, 2, 5} mm`, with the line `T = τ_max`.
4. **Noise and temperature are unmeasured.** Intra-device HD sets the ECC. Without it, a 2048-bit key has no reliability claim.
5. **Entropy versus the 256-bit key.** Run the degrees-of-freedom estimate and a PCA on `raw_samples` before anyone writes "256-bit PUF" into a paper.
6. **Multimode: what question is it answering?** Power-summed detection threw away per-mode phase. The 3-mode HD is 0.45 versus 0.40 on a coarser grid. That is not evidence that more modes add entropy. Per-mode spectra and a degrees-of-freedom count would be.
7. **OFC plan versus this tree.** The plan's gating result was link budget and ENOB, then a 100-device inter-PUF histogram. The code went to the physics chain instead. Decide which document is the backlog before adding files.

---

## How to run

From the repo root, with `.venv`:

```bash
python -m pytest tests/puf_test.py
python puf/puf_model.py
python modulator/modulator_model.py
python modulator/send_to_puf.py
python modulator/plot_hamming_distance.py
```

`send_to_puf.py` follows `_MODE_MODEL` at the top of the file. `plot_hamming_distance.py` always runs both single-mode and 3-mode. Both write PNGs under `modulator/out/`.
