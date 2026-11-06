# Implementation Spec: Optical PUF Physics Chain (`si_phmpy.physics`)

*Briefing for an implementing agent. You have not seen the design conversation. Everything needed is below. Follow the phases in order and run the validation gate at the end of each phase before moving on.*

---

## 1. TL;DR

Build four Python classes that model, end to end, how a silicon-photonic contra-directional-coupler (contra-DC) PUF turns a modulated WDM optical signal into a digital fingerprint:

```
WDMGrid + Challenge bits -> Modulator -> optical field x(t)
x(t) -> ContraDCPUF (linear filter h(t)) -> y(t)
y(t) -> Demux -> Photodetector -> ADC -> 2048-bit fingerprint
```

This is a **physical-layer** simulator: complex optical field envelopes, femtosecond-to-picosecond time grids, coupled-mode theory. It is separate from and feeds into the existing SimPy discrete-event simulator, which models queues and protocol timing and knows nothing about physics. Do not mix them in the same module.

The single most important thing this simulator must answer: **do the challenge bits actually change the response, or is the fingerprint determined only by the carrier wavelengths?** A metric called `eta` (defined in Section 8) measures this. If `eta` is near zero, the design is broken and the simulator must report that loudly rather than quietly producing a plausible-looking fingerprint.

---

## 2. Physical background you need

### 2.1 What the PUF is

A contra-DC is two dissimilar strip waveguides side by side with a periodic corrugation between them. Light entering the forward direction in waveguide 1 is reflected into the backward direction in waveguide 2, but only near the phase-matching wavelength. Fabrication randomness (sidewall roughness, width variation, etch depth variation) perturbs the coupling strength and effective index along the length of the device. That randomness is the secret. It is baked into silicon at manufacture, is not stored in any memory, and cannot be read out except optically.

Physically the device is **passive and linear and time-invariant**. It has no gain, no nonlinearity, no state. That means its entire behavior is captured by one complex-valued function: the reflection spectrum `r(omega)`, or equivalently its inverse Fourier transform, the impulse response `h(t)`. Everything the PUF can ever do is a convolution with `h(t)`.

### 2.2 Delay spread, and why it is the crux

Light entering the grating can turn around at the front, at the back, or anywhere in between. The reflected field is therefore a superposition of paths with different round-trip delays, ranging from 0 to

```
tau_max = 2 * n_g * L / c
```

with `n_g` the group index and `L` the device length. For `L = 500 um`, `n_g = 4`: `tau_max = 13.3 ps`.

That range is the **delay spread** of `h(t)`. It is the memory of the filter, and it is the reason bit patterns can matter:

- If the symbol period `T` is much longer than `tau_max`, each bit passes through the device and settles before the next arrives. The output at the sampling instant is just the input bit multiplied by `|r(omega_i)|^2`. The bit pattern contributes nothing but a known scale factor. **The challenge is inert.**
- If `T` is comparable to or shorter than `tau_max`, energy from previous symbols is still ringing around inside the grating when the current symbol arrives. The interference is set by the random grating profile. Different bit patterns then produce genuinely different waveforms. **The challenge is load-bearing.**

At 10 Gb/s, `T = 100 ps` versus `tau_max = 13 ps`, so there is essentially no pattern dependence. At 50 Gb/s, `T = 20 ps`, and inter-symbol interference is severe and pattern-specific. The simulator must be able to demonstrate this crossover.

### 2.3 Why different carrier grids give different fingerprints from the same device

Per channel `i`, in complex baseband:

```
y_i(t) = integral over tau from 0 to tau_max of  h(tau) * exp(-j*omega_i*tau) * x_i(t - tau)  d(tau)
```

The same physical `h(tau)` applies to every channel, but each optical carrier applies a **different phase ramp** `exp(-j*omega_i*tau)` across the delay axis. Paths that add constructively at one carrier add destructively at another. This is the mathematical reason one physical PUF probed on different WDM grids yields uncorrelated fingerprints (previously measured at 55% inter-grid Hamming distance). Your model must reproduce this. It falls out automatically if you implement Section 5 correctly.

---

## 3. Conventions, units, global grid

Fix these once and never deviate. Most bugs in this kind of simulator are unit and convention bugs.

| Quantity | Unit | Notes |
|---|---|---|
| Length | m | 500 um is `500e-6` |
| Time | s | |
| Frequency | Hz (not rad/s) | convert with `omega = 2*pi*f` only inside the physics kernels |
| Optical field | sqrt(W) | complex envelope, so `abs(E)**2` is instantaneous power in W |
| Power | W | |
| Wavelength | m | 1550 nm is `1550e-9` |
| Angles/phase | rad | |

**One global time grid, shared by every stage.** Define it once and pass it around:

```python
@dataclass(frozen=True)
class SimGrid:
    fs: float      # sample rate, Hz
    n: int         # number of samples
    # derived: dt = 1/fs, t = arange(n)/fs, f = fftfreq(n, dt)
```

Default: `fs = 4e12` (4 THz, so `dt = 0.25 ps`), `n = 16384` (window `T_win = 4.1 ns`, resolution `df = 244 MHz`).

Rationale for these numbers, so you can rescale sensibly:

- `dt = 0.25 ps` puts ~53 samples across a 13.3 ps impulse response. Enough to resolve its structure. Do not go coarser than ~20 samples across `tau_max`.
- `fs = 4 THz` gives a simulated band of +/-2 THz around the reference carrier. It must comfortably cover all WDM channels plus their modulation sidebands. Four channels at 1.2 nm spacing span 450 GHz; 50 Gb/s sidebands add ~100 GHz. 4 THz is generous, which is fine.
- `T_win = 4.1 ns` must be at least ~10x `tau_max` so the impulse response decays inside the window and the circular convolution done by FFT does not wrap around. Also must hold the whole symbol burst (32 symbols at 50 Gb/s is 640 ps). Both satisfied.

**Everything runs in one complex baseband centered at a reference optical frequency `f0`** (take `f0 = c / 1550e-9`). Individual WDM channels are frequency offsets `f_i - f0` within that single baseband signal. Do not model each channel as a separate real passband signal. The single-baseband approach is simpler, faster, and it produces the per-channel phase ramps of Section 2.3 for free.

---

## 4. Class 1: `ContraDCPUF`

Generates the device and its impulse response. This is the only class that contains fabrication randomness.

> **Read Section 4.7 first.** A working prototype named `ContraDCPuf` already exists. It uses a 4x4 two-waveguide coupled-mode formulation which is physically more faithful than the 2x2 Bragg formulation described in 4.1 below. Keep the existing 4x4 formulation. Sections 4.1 to 4.5 give the surrounding structure (disorder model, interface, temperature); Section 4.7 lists exactly what has to change in the existing code.

### 4.1 Physics: coupled-mode theory with transfer matrices

Split the device of length `L` into `M` uniform segments of length `dz`. Each segment has its own coupling coefficient and detuning, drawn from the disorder model. Within a segment the coupled-mode equations have a closed-form 2x2 transfer matrix relating forward and backward field amplitudes.

Detuning of segment `m` at angular frequency `omega`:

```
delta_m(omega) = delta_0(omega) + (2*pi / lambda_0) * dn_m
delta_0(omega) = (beta_1 + beta_2)/2 - pi/Lambda
```

with `beta_j(omega) = 2*pi*n_j/lambda + n_gj*(omega - omega_0)/c`, `Lambda` the grating period, and `dn_m` the per-segment index disorder. To first order it is enough to write

```
delta_0(omega) = delta_0(omega_0) + n_g_avg * (omega - omega_0) / c
```

where `n_g_avg = (n_g1 + n_g2)/2`. Choose `Lambda` so `delta_0(omega_0) = 0`, i.e. `Lambda = lambda_0 / (n_1 + n_2)`. For `n_1 = 2.4`, `n_2 = 2.2`, `lambda_0 = 1550 nm`: `Lambda = 337 nm`.

Segment transfer matrix, with `kappa_m` the segment coupling and `gamma = sqrt(kappa_m**2 - delta_m**2)` (complex; `numpy.sqrt` on a complex input handles both inside and outside the stopband, do not branch manually):

```
F_m = [[ cosh(gamma*dz) - 1j*(delta_m/gamma)*sinh(gamma*dz),  -1j*(kappa_m/gamma)*sinh(gamma*dz) ],
       [  1j*(kappa_m/gamma)*sinh(gamma*dz),                   cosh(gamma*dz) + 1j*(delta_m/gamma)*sinh(gamma*dz) ]]
```

Guard the `gamma -> 0` case: when `abs(gamma*dz) < 1e-12`, use the limits `cosh -> 1`, `sinh(gamma*dz)/gamma -> dz`.

Cascade `F_total = F_M @ ... @ F_2 @ F_1`, then

```
r(omega) = -F_total[1,0] / F_total[1,1]
t(omega) =  1.0        / F_total[1,1]
```

**Numerical stability warning.** `cosh` and `sinh` of large real arguments overflow. Deep in the stopband `gamma` is real and `gamma*L` can reach 5 or more; with `M = 1000` segments each `gamma*dz` is small, so per-segment overflow will not occur, but the cascaded product can still become badly conditioned for very strong or very long gratings. If you see `inf`, `nan`, or `abs(r) > 1`, switch the cascade to a scattering-matrix formulation combined by the Redheffer star product, which is unconditionally stable. Implement TMM first; keep the S-matrix path as a documented fallback.

### 4.2 Critical clarification on segment length

`dz` does **not** need to resolve the grating period. Coupled-mode theory has already averaged over the corrugation; `kappa` and `delta` are envelope quantities. The segment transfer matrix is exact for a uniform segment of any length. `dz` only needs to be small enough to (a) resolve the spatial disorder profile and (b) keep `abs(delta*dz)` and `abs(kappa*dz)` well below 1 across the simulated band.

Across +/-2 THz, `abs(delta)` reaches `n_g*2*pi*2e12/c` which is about `1.7e5 /m`, so `dz < 1 um` keeps `delta*dz < 0.17`. Use `dz = 0.5e-6`, giving `M = 1000` for a 500 um device. This is cheap. **Do not** use `dz = lambda/(10*n_eff)`; that is an FDTD rule of thumb and does not apply here. It would give `M = 7750` for no benefit.

### 4.3 Disorder model

Two independent perturbation profiles along `z`, both spatially correlated (fabrication variation is not white noise):

- `dn(z)`: effective-index disorder, Gaussian, `sigma_dn = 2e-3`, correlation length `L_corr = 2e-6` m.
- `dkappa(z)`: relative coupling disorder, Gaussian, `sigma_kappa_rel = 0.05` (5%), same correlation length.

Generate each by drawing white Gaussian samples on the `z` grid and convolving with a Gaussian kernel of standard deviation `L_corr / dz` samples, then renormalizing to the target standard deviation (the smoothing reduces the variance, so rescale after smoothing, not before). Seed from `device_id` so devices are reproducible and distinct.

Nominal (pre-disorder) coupling `kappa_0 = 4e3 /m`, giving `kappa_0 * L = 2` and peak reflectivity `tanh(2)**2 = 0.93` for a 500 um device.

### 4.4 Interface

```python
class ContraDCPUF:
    def __init__(
        self,
        device_id: int,                 # seeds the disorder RNG; distinct id = distinct device
        length: float = 500e-6,
        dz: float = 0.5e-6,
        kappa_0: float = 4e3,
        n1: float = 2.4,
        n2: float = 2.2,
        ng1: float = 4.2,
        ng2: float = 4.0,
        lambda_0: float = 1550e-9,
        sigma_dn: float = 2e-3,
        sigma_kappa_rel: float = 0.05,
        corr_length: float = 2e-6,
        temperature_offset_K: float = 0.0,
    ): ...

    @property
    def tau_max(self) -> float:
        """2 * n_g_avg * L / c. Delay spread in seconds."""

    def reflection_spectrum(self, f_abs: np.ndarray) -> np.ndarray:
        """Complex r at each absolute optical frequency in f_abs (Hz). Vectorize over
        frequency: build an (n_freq, 2, 2) array per segment and use matrix multiply,
        or loop over M=1000 segments with vectorized frequency inside. Do not loop
        over both."""

    def impulse_response(self, grid: SimGrid, f0: float) -> np.ndarray:
        """Baseband-equivalent h(t), complex, length grid.n.
        1. f_abs = f0 + grid.f      (grid.f is the fftfreq array, already ordered
                                     to match numpy's FFT convention)
        2. r = self.reflection_spectrum(f_abs)
        3. h = ifft(r) * grid.fs    (the fs factor makes convolution-by-multiplication
                                     consistent with a continuous-time convolution
                                     integral; verify against the passivity test)
        Cache the result keyed on (id(grid), f0)."""
```

### 4.5 Temperature

Silicon thermo-optic coefficient: `dn/dT = 1.8e-4 /K`. Apply as a uniform index shift `dn_thermal = 1.8e-4 * temperature_offset_K` added to every segment's `dn_m`. This shifts the whole spectrum by roughly `0.08 nm/K`. Used later to characterize how well the TEC must hold temperature for the fingerprint to stay stable.

### 4.6 Validation gate for Phase 1

Do not proceed until all of these pass.

1. **Passivity.** `abs(r)**2 + abs(t)**2 <= 1 + 1e-9` at every frequency. Violation means a sign or convention error in the transfer matrix.
2. **Uniform-grating analytic check.** With all disorder set to zero, peak reflectivity must equal `tanh(kappa_0 * L)**2` to within 1%, and the stopband half-width in wavelength must be approximately `lambda_0**2 * kappa_0 / (pi * n_g_avg)`.
3. **Causality.** `h(t)` must be negligible for `t < 0`. Numerically: total energy in the negative-time half of the window below 1% of total. Failure usually means the frequency array ordering does not match numpy's FFT convention.
4. **Delay spread.** Energy of `h(t)` must be concentrated in `0 < t < tau_max`, with less than 5% beyond `1.5 * tau_max`.
5. **Device distinctness.** Two `ContraDCPUF` instances with different `device_id` must have visibly different `abs(r)` spectra (correlation coefficient below ~0.5 across the band).
6. **Determinism.** Same `device_id` reproduces bit-identical `h(t)` across runs and across processes.

### 4.7 Existing prototype `ContraDCPuf`: required changes

A prototype exists. It builds a 4x4 coupled-mode transfer matrix over two dissimilar waveguides (forward and backward in each), splits each section as `expm(S1*dL) @ expm(S2*dL)`, solves a boundary-value problem, and returns `(wl_nm, H_drop_dB)`.

**Keep the 4x4 two-waveguide formulation.** It models `kappa11`, `kappa22` (self-coupling) and `kappa12` (cross-coupling) separately, includes propagation loss, and includes first-order dispersion via `ng1`, `ng2`. That is closer to a real contra-DC than the 2x2 Bragg reduction in Section 4.1. The changes below are additive and corrective, not a rewrite.

Current defaults for reference: `L = 210 um`, `n_sect = 80`, `Lambda0 = 305.72 nm` (so drop peak `(n1+n2)·Lambda ≈ 1550 nm` lands on the WDM plan), `n1_0 = 2.580`, `n2_0 = 2.490`, `ng1 = 3.65`, `ng2 = 3.55`, `wl0 = 1550 nm`, `kappa12 = 16300 /m`, `kappa_mult = 3.0`, `loss = 5 dB/cm`, wavelength grid `linspace(1520, 1580, 300)` nm. The channel plan is the fixed system input; retune `Lambda0 = lambda_target/(n1+n2)`, do not slide the WDM grid onto an old Bragg peak.

#### Group A: blockers, the chain cannot run without these

**A1. Return complex, not dB.** `drop_port_response()` returns `10*log10(abs(h)**2)`. The chain needs complex `r`. Add `reflection_spectrum(f_abs) -> complex ndarray` as the primary method and keep the dB version as a thin plotting wrapper on top of it. All phase information is currently discarded at the last line, and phase is what carries the delay spread.

**A2. Extract the drop-port field properly.** `_bvp_solve` currently computes `td = (abs(eb[0])**2 + abs(eb[1])**2) / p_in` and `ph = angle(eb[0] + eb[1])`, then reconstructs `sqrt(td) * exp(1j*ph)`. Two problems: it sums power across both waveguides, and it takes the phase of a sum of fields living in different waveguides, which is not a well-defined transfer phase. The drop port of a contra-DC is the backward-propagating field in waveguide 2 at `z = 0`. Return that single complex amplitude divided by the input amplitude. One complex number, no magnitude/phase round trip.

**A3. Stop normalizing by the peak.** `h /= mx` divides out the maximum response magnitude. This discards absolute reflectivity, makes the passivity test of Section 4.6 impossible to run, and erases genuine device-to-device amplitude variation. Return absolute values. Normalize only inside plotting code.

**A4. Uniform frequency grid.** The wavelength grid is `linspace` in wavelength, therefore nonuniform in frequency, therefore not IFFT-able. `impulse_response()` must evaluate on a grid uniform in `f`, supplied by `SimGrid` as `f0 + grid.f`. Either evaluate the transfer matrix directly at those frequencies (preferred, it is the same cost) or resample. Do not resample by interpolating the dB magnitude; interpolate the complex value, or better, just evaluate where you need it.

**A5. Add fabrication disorder.** This is the largest gap. The current model is fully deterministic: `k12_arr` comes from `anti_apodization()` and `lam_arr` is constant, so two instances differ only if a human changes a constructor argument. There is no device population, so inter-device Hamming distance, entropy estimation, and every security claim that rests on them cannot be evaluated.

Add a `device_id` argument seeding a dedicated `np.random.default_rng`, and draw three spatially correlated profiles over the `n_sect` sections, per Section 4.3 (draw white, smooth with a Gaussian kernel of width `corr_length/dL` samples, then rescale to the target standard deviation):

| Profile | Applied to | Suggested sigma |
|---|---|---|
| `dn1[i]`, `dn2[i]` | added to `n1`, `n2` inside `_section_C` | 2e-3 |
| `dkappa[i]` | multiplies `k12_arr[i]` | 5% relative |
| `dLambda[i]` | replaces the constant `lam_arr` | 0.5 nm |

Correlation length 2 um. Keep `anti_apodization()` as the deterministic backbone and multiply the disorder on top of it. The engineered ripple and the random ripple serve different purposes and should coexist.

#### Group B: correctness

**B1. `n_sect = 80` is under-resolved.** Section length is `dL = 2.6 um`. Across the 70 nm span the detuning reaches roughly `2*pi*n_g*d_lambda/lambda**2 ~ 6.8e5 /m`, so `abs(delta*dL) ~ 1.8 rad` at the band edges. The requirement is well below 1. Compounding this, `expm(S1*dL) @ expm(S2*dL)` is a first-order Lie-Trotter split with O(dL^2) error, so the operator splitting is also degrading. Raise `n_sect` to 500 to 1000 for this span, and add a convergence test: double `n_sect` and require the resulting spectrum to change by less than 1%. Note this makes B3 mandatory rather than optional.

**B2. Verify the boundary-value problem.** The physical condition for a contra-DC is: forward amplitudes known at `z = 0`, backward amplitudes **zero at `z = L`**. The current `_bvp_solve` inverts `M[0:2, 0:2]` against the input vector and does not visibly impose the zero-backward-input condition at the far end. Partition the full 4x4 explicitly and verify against a case with a known analytic answer. Separately, `ein = [ka1, ka2] = [0.99, 0.14]` launches into both waveguides simultaneously; standard interrogation launches into waveguide 1 only, `[1, 0]`. If the two-waveguide launch is intentional, document why.

**B3. Silent failure.** `except np.linalg.LinAlgError: return 0.0, 0.0` fabricates a legitimate-looking data point out of a numerical failure. Raise, or return NaN and propagate it.

**B4. Loss convention.** Loss enters the detunings as `beta = 2*pi*n/wl - 1j*alpha`, but the phase factors `p11`, `p22`, `p12` are built from `np.real(beta)`. The forward and backward blocks of `S2` then carry `-1j*d` and `+1j*d`, so the imaginary part appears with opposite signs and the backward-propagating wave may see gain rather than loss. The dB-to-field conversion in `_alpha()` is correct (`20*log10(e) = 8.686`); the issue is only the sign handling in the backward rows. The passivity test in Section 4.6 will catch this. Run that test before trusting any spectrum.

**B5. Coupling is saturated.** `kappa12 * kappa_mult * L = 16300 * 3 * 210e-6 = 10.3`, and anti-apodization boosts it further near the edges. `tanh(10.3)**2` is indistinguishable from 1, so the stopband is a flat-topped plateau. A flat spectrum has few resolvable features, which means **less** entropy, not more. Sweep `kappa_mult` down and check spectral feature count (count local extrema, or use the PCA measure from Section 8.3) rather than assuming stronger coupling is better.

#### Group C: performance

**C1. Two `expm` calls per section per wavelength.** Currently 300 x 80 x 2 = 48,000 4x4 matrix exponentials per device. After B1 raises `n_sect` to 1000 and the sweeps in Phase 6 call for 100+ devices, this becomes hours.

Three fixes, in order of payoff:
- `S1` is diagonal. `expm(S1*dL)` is `np.exp` on the diagonal. Remove that `expm` call entirely.
- Vectorize over wavelength. Build `(n_freq, 4, 4)` arrays and use `np.matmul` broadcasting. Loop over sections only.
- If `expm` on `S2` is still the bottleneck, replace it with an eigendecomposition computed once per section and reused across all frequencies where the section parameters are frequency-independent.

**C2. No caching.** `drop_port_response()` recomputes from scratch on every call. Cache on `(device_id, frequency grid identity, temperature_offset_K)`.

#### Group D: interface alignment

**D1.** Add the `tau_max` property: `2 * ((ng1+ng2)/2) * L / c`.

**D2.** Add `temperature_offset_K`, applied as a uniform `dn = 1.8e-4 * dT` added to `n1` and `n2`.

**D3.** Rename the internal variable `h` (currently the transfer function) to `r` or `H_drop`. The name `h` is reserved throughout this spec for the time-domain impulse response, and the collision will cause confusion.

**D4.** Move the wavelength grid out of the constructor. The device should not own the frequency axis; the caller passes it in. `wl_nm` becomes a default for the plotting wrapper only.

#### Physics reality check on the current geometry

`L = 210 um` with `n_g ~ 3.6` gives `tau_max = 5.0 ps`. At 50 Gb/s the symbol period is 20 ps, so `T` is four times the delay spread and `eta` will come out near zero. **The current device geometry sits in the regime where the challenge bits are inert.** This is expected and is not a bug in the code; it is the finding the simulator exists to quantify. Run the `eta` sweep of Section 8.1 on this geometry first to establish the baseline, then use it to justify the longer device that the entropy accounting in Section 8.3 independently calls for.

### 4.8 Multimode extension (opt-in)

Single-mode (one guided mode per waveguide, 4x4 CMT) remains the **default**. Multimode is selected explicitly:

| `mode_model` | Behavior |
|---|---|
| `"single"` (default) | Force M1=M2=1, legacy 4x4 fast path |
| `"auto"` | Mode count from waveguide widths `w1`, `w2` via `puf/modes.py` (EIM slab) |
| `"fixed"` | Cap at `n_modes_max` (widths preferred; else synthetic n_eff ladder) |

Physics: state dimension `d = 2*(M1+M2)`, coupling becomes matrices `K11,K12,K22` from lateral sidewall overlaps (anchored so `K12[0,0] = kappa12`). Each mode pair phase-matches at `λ_ij = Λ·(n1_i + n2_j)`, so one grating produces multiple stopbands. The receiver sums mode powers on one PD per wavelength (`reflection_power_spectrum`); key shape is unchanged. Toggle in `modulator/send_to_puf.py` via `_MODE_MODEL`, `_W1`, `_W2`, `_N_MODES_MAX`.

---

## 5. Class 2: `WDMGrid` and Class 3: `Modulator`

### 5.1 `WDMGrid`

Thin, dumb, and pure. It exists so that channel plans are declared in one place.

```python
@dataclass(frozen=True)
class WDMGrid:
    lambda_start: float = 1548e-9
    delta_lambda: float = 1.2e-9
    n_channels: int = 4

    def wavelengths(self) -> np.ndarray: ...
    def frequencies(self) -> np.ndarray:        # absolute, Hz
    def offsets(self, f0: float) -> np.ndarray: # f_i - f0, baseband offsets
```

Note for the record: `delta_lambda` is a system-level parameter shared with the WDM data plane. It is **not** a PUF tuning knob. The simulator may sweep it to characterize sensitivity, but the design must work at whatever the channel plan dictates.

### 5.2 `Modulator`

Takes a challenge bit sequence and produces the complex optical field entering the PUF.

Per channel `i` with bits `b[i, :]`:

1. **Symbol sequence to waveform.** Map bit `1 -> 1.0`, bit `0 -> 1/ER` where `ER` is the linear extinction ratio (default 6 dB, so `ER = 3.98`). Finite extinction matters: with a perfect zero, the "off" symbol carries no light and part of the pattern dependence disappears.
2. **Finite modulator bandwidth.** Upsample the symbol sequence to the sim grid as a rectangular hold, then low-pass with a Gaussian filter of 3 dB bandwidth `0.75 * symbol_rate`. Physically the bandwidth limit acts on the modulator *drive*; filtering optical power is an approximation. Prefer a strictly positive kernel (Gaussian) so the filtered power cannot go negative — a Bessel on power would undershoot and force a clip that changes the eye.
3. **Chirp.** Microring modulators are strongly chirped. Use the standard alpha-parameter model on the resulting power envelope `P(t)`:

   ```
   phi(t) = (alpha / 2) * log(P(t) / P_mean)
   E_i(t) = sqrt(P(t)) * exp(1j * phi(t))
   ```

   with `alpha = 3.0` by default (MRMs typically fall in 1 to 5). This is a first-cut model, not a full ring model. Document that. It matters because a dispersive PUF converts this phase pattern into amplitude in a device-specific way, which is free entropy the design already gets.
4. **Frequency shift and sum.**

   ```
   x(t) = sum over i of  E_i(t) * exp(1j * 2*pi * (f_i - f0) * t)
   ```

   Keep the pre-sum per-channel fields as `channel_fields` on `ModulatedField` for demux debugging and per-channel launch checks. Reject the configuration if `max|f_i - f0| >= fs/2` (Nyquist).
5. Normalize total launch power to `p_launch` (default 1 mW) over the **burst window** `[0, n_sym * sps)` only. Including the idle tail makes the scale a window artifact, not a physical launch power, and that scale becomes the ADC's `p_launch_ref`. Fill the idle tail with an explicit `p_idle` (default `1/ER`), not with `power[0]`.

```python
class Modulator:
    def __init__(
        self,
        grid: WDMGrid,
        symbol_rate: float = 50e9,      # baud; bitrate is a derived property
        fmt: str = "nrz",               # "nrz" (1 bit/sym) or "pam4" (2 bits/sym)
        gray_map: bool = True,          # PAM4 only
        extinction_ratio_dB: float = 6.0,
        chirp_alpha: float = 3.0,
        mod_bw_factor: float = 0.75,
        p_launch: float = 1e-3,
        p_idle: float | None = None,    # default 1/ER
    ): ...

    @property
    def bitrate(self) -> float:
        """Bit rate: symbol_rate for NRZ, 2*symbol_rate for PAM4."""

    def modulate(self, challenge: np.ndarray, simgrid: SimGrid, f0: float) -> ModulatedField:
        """challenge: shape (n_channels, n_bits), dtype uint8, values 0/1.
        Returns ModulatedField(field, channel_fields, p_launch_ref, symbol_edges).
        Raises if n_sym * sps > simgrid.n (no silent truncation)."""
```

`symbol_edges` must be carried downstream. The ADC needs it to place sampling instants relative to symbol transitions. Warn if `sps < 16`.

**Challenge sequences must be independent per channel.** Draw them from separate RNG streams. Do not reuse one sequence across all four channels.

**PAM4 note.** PAM4 at a given symbol rate doubles the challenge bits carried per symbol, so it enlarges the challenge space. It does **not** raise `eta`, which is set by symbol period against delay spread, and it does **not** raise response entropy, which is set by the device time-bandwidth product. It costs noise margin, roughly 9.5 dB against NRZ at equal average power. Treat it as a challenge-space knob only, and do not let it appear in entropy arguments.

### 5.3 Existing prototype `Modulator`: required changes

A prototype exists. It takes a flat challenge bit vector, a symbol rate, a WDM grid specified as `(wl0_nm, spacing_nm, n_wavelengths)`, and a format flag `nrz | pam4`. It maps bits to symbols, symbols to normalized levels, and returns two flat arrays `(wavelengths, amplitudes)` each of length `n_symbols * n_lambdas`.

**Keep:** the bits-to-symbols-to-levels layering, the `fmt` flag and PAM4 support, the input validation on challenge values and PAM4 length parity, and the grid-from-`(wl0, spacing, count)` construction. That last one becomes `WDMGrid` per Section 5.1.

**Structural problem:** the output has no time axis, so it cannot drive a convolution.

#### Group A: blockers

**A1. There is no time axis.** `modulate()` returns per-symbol static levels. The chain needs a complex field sampled on `SimGrid` at `dt = 0.25 ps`. Without a time axis there is no convolution with `h(t)`, therefore no inter-symbol interference, therefore no pattern dependence, therefore no `eta`. Every claim about challenge bits mattering is unrepresentable in the current output format. The return type must become a time-domain complex array plus metadata, per the `ModulatedField` definition in Section 5.2.

**A2. `speed_gbps` has no effect.** `self.symbol_period_s` is computed in `__init__` and never read. Symbol rate against `tau_max` is the single most important parameter in this simulator, and right now sweeping it produces identical output. Fixing A1 fixes this, but verify it explicitly with a test: two instances differing only in `speed_gbps` must produce different fields.

**A3. Every carrier gets the same symbol.** `np.repeat(self.symbol_amps, self.n_lambdas)` broadcasts one sequence across all channels. Consequences: four channels carry one channel's worth of pattern information, the effective challenge space is reduced by a factor of `n_channels`, and the per-channel responses become correlated in a way that inflates apparent redundancy in the fingerprint. Change `challenge` to shape `(n_channels, n_symbols)` and require independent RNG streams per channel.

**A4. Field must be complex, and chirped.** `symbol_amps` is real. Apply the alpha-parameter chirp of Section 5.2 step 3 to get `E(t) = sqrt(P(t)) * exp(1j*phi(t))`. A dispersive PUF converts that phase pattern into amplitude in a device-specific way. It is entropy the hardware already produces and the current model throws away.

**A5. Wavelength is not a per-sample quantity.** `np.tile(self.lambdas_nm, self.n_symbols)` produces a wavelength value per output element, which conflates channel index with time index. Wavelength identifies a carrier. It belongs in `WDMGrid`, referenced once. The modulator output should be the single complex baseband sum of Section 5.2 step 4, with the grid available separately for the demux stage.

#### Group B: correctness

**B1. Infinite extinction ratio.** `amp_off = 0.0`. Real microring modulators give 4 to 8 dB. A perfect zero means the off symbol carries no light at all, which suppresses part of the interference that makes patterns distinguishable, and it makes the readout look cleaner than hardware will. Default to 6 dB.

**B2. No modulator bandwidth limit.** Symbols become a rectangular hold with infinitely sharp edges and unbounded sideband energy. This is unphysical and it will inflate `eta` and the entropy estimate, both in the optimistic direction. Apply the Gaussian or 2nd-order Bessel filter at `mod_bw_factor * symbol_rate` from Section 5.2 step 2.

**B3. Amplitude versus power is ambiguous.** `NRZ_LEVELS` is documented as normalized intensity, but the variable is named `amp` and is used directly as an amplitude. `PAM4_LEVELS = [0, 1/3, 2/3, 1]` is equally spaced, which is the optical *power* convention for PAM4 on an intensity-modulated link. Resolve it explicitly: levels are **power**, and the field is `sqrt(power)`. State it in the docstring. Getting this wrong changes PAM4 level spacing at the detector and silently changes every noise-margin result.

**B4. PAM4 mapping is natural binary.** `(pairs[:,0] << 1) | pairs[:,1]` means a one-level slicing error can flip two bits. Gray-map it so a one-level error flips exactly one bit, same argument as the ADC in Section 7.3. Keep natural binary available behind the `gray_map` flag for comparison.

**B5. Missing outputs the downstream stages need.** No `p_launch` normalization and no tap power reference, so the ADC has nothing to normalize against and the fingerprint will track laser drift. No `symbol_edges`, so the ADC cannot place its sample instants relative to symbol transitions, which Section 7.1 identifies as mattering more than sample count. Both must be returned in `ModulatedField`.

#### Group C: defaults and carrier placement

**C1. Channel spacing default is wrong by an order of magnitude.** The demo uses `spacing_nm = 10.0`, spanning 30 nm over four carriers. The project baseline is `delta_lambda = 1.2 nm`, and that value is fixed by the WDM data plane, not chosen here.

**C2. Carriers must land inside the PUF stopband.** This is the more serious version of C1. A contra-DC reflects strongly only within its stopband, a few nm wide. At 10 nm spacing most carriers fall outside it, `abs(r)` collapses toward the noise floor, and the readout SNR goes with it. Add an explicit check: compute `abs(r)` at every carrier frequency and warn if any carrier sits below, say, 10% of peak reflectivity.

This is the concrete form of the standing constraint that PUF free spectral range and feature density must be co-designed with the channel plan. The channel plan is the fixed input. Stopband width and center are the design variables, set by grating length, coupling strength, and period. Add a helper that reports stopband edges from the computed spectrum so carrier placement can be checked rather than assumed.

**C3. Demo bitrate is inert.** `speed_gbps = 10.0` gives `T = 100 ps` against `tau_max = 5 ps` for the current 210 um device. Twenty times over, so `eta` will be indistinguishable from zero. Use it as the documented baseline case, then sweep upward.

**C4. Default grid vs default grating were independently wrong.** With the old `Lambda0 = 300 nm`, the contra-DC drop peak sits at `(n1+n2)·Lambda = 1521 nm`, while `kappa11` self-reflect is at `2·n1·Lambda = 1548 nm`. `WDMGrid` defaulted to `lambda_start = 1548 nm` — 27 nm off the drop port, and exactly on the parasitic F1→B1 self-reflection that does not reach the drop. End-to-end runs were interrogating the darkest carrier. Fix by retuning `Lambda0 = lambda_target/(n1+n2)` (305.72 nm for 1550 nm), not by sliding the channel plan. The plan is a system input; the grating is the design variable.

**C5. `check_carriers_in_stopband` must use a global peak.** Taking `peak = max|r|` over a window around the WDM grid fails open: if the whole grid is outside the stopband, the local max is also tiny, the ratio is near 1, and the check passes while every carrier is dark. Peak reference must come from the device's full response band (`ContraDCPUF.peak_reflectivity()`, or the PUF's own scan range). C4 is exactly what this check exists to catch.

#### Group D: interface alignment

**D1.** `n_lambdas` duplicates `n_wavelengths`. Drop one.

**D2.** Grid construction moves to `WDMGrid` (Section 5.1). The modulator takes a `WDMGrid` instance rather than three scalars.

**D3.** Work in SI internally. Accept nm at the constructor boundary if preferred for readability, but convert immediately and keep every internal quantity in meters and Hz per Section 3.

**D4.** `modulate()` returns a `ModulatedField` dataclass, not a bare tuple. Two same-length flat arrays that mean different things are an invitation to transpose them.

#### Group E: post-rewrite correctness (time-domain `Modulator`)

**E1. Silent burst truncation.** `if start >= simgrid.n: break` shortens the challenge without error. At 10 GBd with `sps = 400`, 16 symbols need 6400 samples against a 4096-point grid — the bitrate-sweep smoke test truncated and still printed a difference RMS. Raise when `n_sym * sps > simgrid.n`. Configuration error, worst at the low rates needed for the `eta` baseline.

**E2. Launch normalization is a window artifact.** `mean(|x|**2)` over the whole array includes the idle tail. Change burst length or window length and the scale changes, so `p_launch_ref` is not physical launch power. Normalize over `[0, n_sym*sps)` only.

**E3. No Nyquist guard on carrier offsets.** `max|f_i - f0|` must stay under `fs/2`. At 1.2 nm spacing this is fine (~250 GHz vs 2 THz); at the old 10 nm demo spacing it aliases silently.

**E4. No minimum on `sps`.** Below ~16 samples/symbol the rectangular hold stops being meaningful. Warn.

**E5. `bitrate` naming.** Constructor parameter is `symbol_rate` (baud). Expose `bitrate` as a derived property (`×2` for PAM4). Otherwise every PAM4 result is misreported by a factor of two.

**E6. Idle level.** Tail fill must be an explicit `p_idle` (default `1/ER`). Filling with `power[0]` only keeps the FFT wrap continuous by accident and breaks the moment anyone changes the first symbol.

**E7. Keep per-channel fields.** `ModulatedField.channel_fields` holds the pre-sum carriers. Demux will re-separate them anyway; having the launch-side version makes per-channel launch references and debugging much easier.

---

## 6. Propagation, demux, detection

### 6.1 Propagation through the PUF

```python
def propagate(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """y = ifft(fft(x) * fft(h)) / fs   -- equivalently multiply spectra directly.
    Because both live on the same SimGrid this is one multiply in the frequency
    domain. The result is the reflected field returning through the MZI."""
```

Sanity check: `sum(abs(y)**2) <= sum(abs(x)**2)`. The device is passive; reflected energy cannot exceed launched energy.

### 6.2 `ChannelDemux`

The node's receiver uses microring drop filters, one per channel, before the photodiodes. Model each as a Lorentzian in the frequency domain:

```
H_drop_i(f) = 1 / (1 + 2j * (f - f_i) / fwhm)
```

Default `fwhm = 30e9` (30 GHz).

**Flag this to the user as a design tension.** A 30 GHz drop filter in front of a 50 Gb/s signal removes most of the modulation sidebands, and the sidebands are precisely what carries the pattern-dependent information. Drop-filter bandwidth therefore places a hard ceiling on `eta`. The simulator should sweep `fwhm` and report its effect. Do not silently pick a wide filter to make the numbers look good.

Output: `n_channels` separate complex baseband fields, each frequency-shifted back to zero.

### 6.3 `Photodetector`

Per channel, square-law detection plus noise plus receiver bandwidth limiting:

1. `i_photo(t) = responsivity * abs(y_i(t))**2`, responsivity default 0.9 A/W.
2. Shot noise: Gaussian, variance `2 * q * i_photo * B_eff` per sample, `q = 1.602e-19`.
3. Thermal noise: Gaussian, variance `4 * k_B * T * B_eff / R_load`, `R_load = 50`, `T = 300`.
4. Optional TIA input-referred noise as a flat current-noise-density term, default `10e-12 A/sqrt(Hz)`.
5. Low-pass filter (4th-order Bessel), cutoff `0.7 * bitrate`, representing the TIA/LPF chain.

Because all channels are demultiplexed **before** detection, there are no cross-channel beat terms. If two carriers were ever placed within one photodiode's bandwidth they would beat and the relative pattern alignment across channels would matter, but at 1.2 nm spacing the beat sits at 150 GHz and is filtered out. Do not add cross-channel mixing terms to the model unless the channel plan is deliberately changed.

Noise must be reproducible: take an explicit `rng` argument so intra-device reliability runs can be repeated.

---

## 7. Class 4: `ADC` and fingerprint assembly

### 7.1 Sampling

Target output: 2048 bits = 4 channels x 128 samples x 4 bits.

**Where to sample matters more than how many samples you take.** Steady-state mid-symbol levels carry only `abs(r(f_i))`, one number per channel. The pattern-dependent information lives in the transient after each symbol transition, while the grating is still ringing. So:

- Use a burst of 32 symbols per channel.
- Take 4 samples per symbol.
- Place them at fractional offsets `[0.1, 0.25, 0.45, 0.75]` of the symbol period, measured from the symbol edge given by `symbol_edges`. Front-loaded deliberately, to catch the transient.
- Make the offsets a constructor parameter so they can be swept.

### 7.2 Normalization

Divide each sample by the tap launch-power reference before quantizing. Without this, the fingerprint tracks laser power drift rather than the PUF. Then apply a fixed full-scale range (a constructor parameter, calibrated once per node at enrollment, not per session) and clip.

### 7.3 Quantization

Uniform 4-bit, 16 levels, **Gray-coded**. Gray coding matters: a one-level analog error then flips exactly one bit instead of up to four, which directly reduces the load on the fuzzy extractor's error-correcting code.

The deployed pipeline is 16-level. Some figures show 4-level for illustration only. Make the bit depth a parameter, default 4.

```python
class ADC:
    def __init__(
        self,
        bits: int = 4,
        samples_per_symbol: int = 4,
        sample_offsets: tuple = (0.1, 0.25, 0.45, 0.75),
        full_scale: float = None,   # None = auto-calibrate on first call, then freeze
        gray_code: bool = True,
    ): ...

    def digitize(self, currents: list[np.ndarray], symbol_edges, p_launch_ref) -> Fingerprint:
        """Returns Fingerprint(bits=uint8 array of length 2048,
                               raw_samples=float array (n_ch, 128),
                               metadata=dict)."""
```

Keep `raw_samples` on the object. Almost every diagnostic in Section 8 is easier on the analog values than on the bits, and the quantizer margin analysis needs them.

---

## 8. Metrics module (this is the point of the whole exercise)

Implement in `si_phmpy/physics/metrics.py`. These are the deliverables; the classes above are just the machinery that feeds them.

### 8.1 `eta`, the ISI energy ratio

The headline number. Compute the symbol-spaced tap sequence for channel `i`:

```
p_k(i) = integral of  h(tau) * exp(-1j*omega_i*tau) * g(k*T - tau)  d(tau)
```

where `g` is the transmit pulse shape after the modulator's bandwidth limiting. In practice: pass a single isolated symbol through the full chain up to the photodiode, sample at symbol centers, and read off the tap magnitudes. Then

```
eta_i = sum over k != 0 of abs(p_k)**2  /  sum over all k of abs(p_k)**2
```

Interpretation:

| `eta` | Meaning |
|---|---|
| < 0.05 | Challenge bits are inert. The fingerprint is determined by the carrier grid alone. **The design does not work as claimed.** |
| 0.1 to 0.3 | Weak but real pattern dependence. |
| > 0.3 | Strong pattern dependence. Target region. |

`eta` must be printed by default in every simulation run, not hidden behind a debug flag.

### 8.2 Hamming distance suite

All computed on the 2048-bit fingerprints.

- **Inter-challenge HD.** Fix the device, fix the grid, vary only the challenge bits. **Run this first, before anything else.** If it is near zero, the challenge is not a challenge and everything downstream is moot. Target: near 50%.
- **Inter-device HD.** Fix challenge and grid, vary `device_id` across at least 100 devices. Target: near 50%. This corresponds to an open experimental item that has been assumed but never measured, so the simulated value is genuinely informative.
- **Inter-grid HD.** Fix device and challenge, vary the WDM grid. A previous result gives 55% at `delta_lambda = 1.2 nm`. Reproducing that number is a strong validation that the phase-ramp physics of Section 2.3 is implemented correctly.
- **Intra-device HD (reliability).** Same device, same challenge, same grid, repeated with independent noise draws and with temperature offsets of 0.1, 0.5, 1.0, 2.0 K. Target: below 10%, which is the error budget the fuzzy extractor's code must absorb. This is currently uncharacterized and is a required input to the ECC design.

Report mean and standard deviation, and produce histograms.

### 8.3 Entropy accounting

**Confront this early; it may be the most consequential result the simulator produces.**

A linear passive filter has a finite number of degrees of freedom, set by its time-bandwidth product:

```
DOF_total (complex) ~ tau_max * B_available
```

At `tau_max = 13.3 ps` over 2 THz of usable band, that is about 27 independent complex features **for the entire device**.

But you only observe the device where you interrogate it. With `N_ch` CW carriers you sample the spectrum at `N_ch` points. Adding modulation of bandwidth `B` around each carrier resolves roughly `2*B*tau_max` additional independent points per carrier, because the sidebands probe frequencies the bare carrier does not reach. So:

```
DOF_observed (complex) ~ N_ch * (1 + 2*B*tau_max)
```

Worked example at the current baseline. `N_ch = 4`, `B = 50 GHz`, `tau_max = 13.3 ps`:

```
2*B*tau_max = 1.33
DOF_observed = 4 * 2.33 = 9.3 complex = ~18.6 real numbers
At ~4 bits each (optimistic; true entropy per sample is lower): ~74 bits
```

**That does not support a 256-bit `K_PUF`.** The 2048-bit fingerprint is roughly 19 real numbers with 100x redundancy. Redundancy helps reliability. It does not create entropy.

Getting to 256 bits requires roughly 100 real DOF, so about 50 complex. One solution point:

```
N_ch = 8, bitrate = 50 Gb/s, tau_max = 52 ps  ->  L = tau_max*c/(2*n_g) ~ 2.0 mm
8 * (1 + 2*50e9*52e-12) = 8 * 6.2 = 50 complex DOF
```

So: **8 channels and a roughly 2 mm contra-DC**, versus the 500 um currently assumed. Note that a 2 mm device also pushes `tau_max` to 52 ps, well past the 20 ps symbol period, which raises `eta` at the same time. The two requirements pull in the same direction, which is convenient.

Implement:

- `estimate_dof(puf, grid, modulator)` returning the analytic DOF from the formula above.
- `empirical_entropy(fingerprints)`: per-bit min-entropy, plus a PCA on `raw_samples` across many challenges reporting how many principal components hold 95% of the variance. The PCA number is the empirical check on the analytic DOF estimate. If they disagree by more than ~2x, one of them is wrong; investigate before trusting either.
- `bit_bias(fingerprints)`: fraction of ones per bit position. Should be near 0.5. Systematic bias means the quantizer full-scale is miscalibrated.

### 8.4 Honest limitation to state in any writeup

Because the device is linear and passive, the entire challenge-response space is determined by `r(omega)`, which has finitely many parameters. Given enough observed challenge-response pairs, the map is machine-learnable. The security argument therefore cannot be "the map is hard to model." It has to be that raw responses never leave the node enclave, so an attacker never collects the pairs. Write it that way. Do not overclaim modeling resistance.

---

## 9. File layout

```
si_phmpy/physics/
    __init__.py
    grid.py          # SimGrid, WDMGrid
    puf_model.py     # ContraDCPUF
    modulator.py     # Modulator, ModulatedField
    optics.py        # propagate(), ChannelDemux
    detector.py      # Photodetector
    adc.py           # ADC, Fingerprint
    chain.py         # PUFChain: wires all of the above into one call
    metrics.py       # eta, HD suite, entropy accounting
tests/physics/
    test_puf.py      # the Phase 1 validation gate
    test_chain.py    # end-to-end shape, determinism, energy conservation
    test_metrics.py  # metrics on synthetic inputs with known answers
scripts/
    sweep_eta.py
    sweep_hd.py
    sweep_entropy.py
```

`PUFChain` is the single convenience entry point:

```python
chain = PUFChain(puf=..., grid=..., modulator=..., demux=..., detector=..., adc=..., simgrid=...)
fp = chain.run(challenge, rng=...)     # -> Fingerprint
```

Everything below `PUFChain` must remain independently testable. Do not let the convenience wrapper become the only usable path.

---

## 10. Implementation phases

Each phase ends with a gate. Do not start the next phase until the gate passes.

**Phase 1: `ContraDCPUF`.** Transfer matrix, disorder, `r(omega)`, `h(t)`. Gate: all six checks in Section 4.6.

**Phase 2: `WDMGrid` and `Modulator`.** Gate: modulated spectrum shows `n_channels` distinct lobes at the right offsets; total power equals `p_launch`; a single isolated symbol produces the expected eye-shaped waveform.

**Phase 3: `propagate`, `ChannelDemux`, `Photodetector`.** Gate: energy conservation holds; with noise disabled and a CW input, per-channel photocurrent equals `responsivity * p_launch/n_ch * abs(r(f_i))**2` to within a few percent.

**Phase 4: `ADC` and `Fingerprint`.** Gate: output is exactly 2048 bits; bit bias within 0.45 to 0.55; repeated runs with a fixed seed are identical.

**Phase 5: metrics.** Gate: inter-grid HD reproduces the previously measured ~55% at `delta_lambda = 1.2 nm`. If it does not, the phase-ramp physics is wrong. Fix it before running anything else.

**Phase 6: sweeps.** Produce these figures:

1. `eta` versus bitrate (1 to 100 Gb/s), for `L` in {0.1, 0.5, 1, 2, 5} mm. This is the central design curve. Overlay the line `T = tau_max`.
2. Inter-challenge HD versus bitrate. Should track `eta`.
3. Inter-device HD histogram, at least 100 devices.
4. Intra-device HD versus temperature offset, 0 to 2 K. Overlay the ECC error budget.
5. Estimated entropy versus `(L, n_channels, bitrate)`, with the 256-bit target as a contour.
6. `eta` versus drop-filter FWHM, 10 to 200 GHz.
7. Inter-grid HD versus `delta_lambda`, reproducing the known 45 to 60% band, including the anomalous 64% spike at 1.0 nm attributed to aliasing against the PUF's spectral periodicity. If the simulator reproduces that spike from first principles, that is a significant validation and worth reporting.

---

## 11. Watch out for

- **FFT conventions.** Frequency array ordering, the `1/n` normalization, and where the `fs` factor goes are the top source of bugs here. The causality test in Section 4.6 is the canary. Get it passing before writing another line.
- **Do not use `dz = lambda/(10*n_eff)`.** See Section 4.2. It is 8x more expensive for zero benefit.
- **Do not add cross-channel beat terms** at the photodiode. At 1.2 nm spacing the beat is at 150 GHz and both the drop filter and the LPF remove it. A previous version of this analysis wrongly claimed relative pattern alignment across channels contributes for free; it does not, at this channel spacing.
- **Do not silently widen the drop filter** to raise `eta`. Report the tradeoff instead.
- **Do not let redundancy masquerade as entropy.** 2048 bits from 19 real numbers is 2048 bits of data and far less than 2048 bits of entropy. Every entropy claim goes through Section 8.3.
- **Do not import SimPy into this package.** The physics simulator and the discrete-event simulator communicate through saved fingerprint arrays and summary statistics, not shared objects.
- **Zero-disorder must still work.** Keep it as a code path; it is the analytic validation case.
- **Vectorize over frequency, loop over segments.** A double loop over 1000 segments x 16384 frequencies in pure Python will take minutes per device, and the sweeps in Phase 6 need hundreds of devices.

---

## 12. Preferences of the person you are working for

- Short, concise answers. No verbosity, no over-explanation, no restating what was just said.
- Hardware-level reasoning is preferred over software-stack abstraction. Push back on inflated block counts and on jargon.
- Descriptive, self-evident naming. "Enrollment DB", not "CRP + Commitment Store".
- Diagrams as bare Mermaid code blocks with no surrounding prose.
- No em-dashes in written output. No line breaks between sentences within a paragraph in LaTeX.
- If a result contradicts a prior assumption, say so directly. The entropy accounting in Section 8.3 is exactly this kind of result and should be surfaced plainly rather than softened.