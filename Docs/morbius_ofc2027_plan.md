# MORBIUS: OFC 2027 Execution Plan

**Target:** OFC 2027, Los Angeles, 7-11 March 2027
**Deadline:** 20 October 2026 (35-word abstract + 3-page summary, Optica style)
**Today:** 25 July 2026
**Runway:** 12.5 weeks

**Paper thesis for this venue:** a WDM-carrier-grid optical PUF whose uniqueness, reliability, and link penalty are all characterized. Digital cost is one sentence. The photonics is the paper.

---

## Week 0: blocking decisions (3 days, do not start code before these)

| # | Decision | Options | Default if undecided |
|---|---|---|---|
| D1 | Contra-DC compact model | Lumerical/Ansys S-params vs analytic coupled-mode | Analytic CMT, calibrated later |
| D2 | Per-instance variation set | grating period, dn_eff, kappa, corrugation depth; sigma sourced from PDK or literature | period + dn_eff + kappa, sigma from published SOI variation data |
| D3 | Receiver architecture (data plane) | analog CTLE+DFE vs ADC/DSP | analog CTLE+DFE (chiplet reach) |
| D4 | Modulator model | MRM (Lorentzian, photon-lifetime BW) vs MZM | MRM, matches the architecture |
| D5 | Helper data storage | per-node vs per-challenge | per-node, note DB growth as future work |

D1 and D2 set the credibility of every Hamming distance number in the paper. Write the chosen sigma values and their source into the repo README on day 1.

---

## Phase 1 (Weeks 1-2): Link budget and ENOB

The gating result. Everything downstream depends on it.

**Build**
- `linkbudget.py`: dB accounting for both modes.
  - Data mode: laser, MRM insertion + modulation loss, MZI bar state, coupling, fiber/waveguide, PD.
  - Auth mode: laser, MRM, MZI cross state, contra-DC through/drop, PD.
- `noise.py`: shot, TIA input-referred current noise, RIN, dark current, integrated over noise bandwidth. Returns SNR and SINAD.
- ENOB = (SINAD - 1.76) / 6.02.

**Done when**
- Auth-mode received power in dBm and ENOB at the ADC are both numbers on paper.
- Data-mode received power feeds the Phase 4 model.
- MZI leakage power at the off-chip port in dBm, compared against a coherent eavesdropper sensitivity.

**Kill criterion:** if ENOB < 3, stop and fix before proceeding. Levers: raise launch power, narrow the LPF, increase samples per channel and average, drop to 3 bits/sample. Do not carry a broken ENOB into Phase 3.

---

## Phase 2 (Weeks 3-4): SAX PUF model and inter-PUF Hamming distance

Closes open item 1.

**Build**
- `contradc.py`: SAX model per D1, seeded per-instance variation per D2. Sanity check FSR, stopband, group delay against analytic expectation.
- `readout.py`: carrier grid sampling, challenge bit modulation, PD/TIA, noise from Phase 1, LPF, 4-bit quantize. Signature: `fingerprint(seed, challenge, temperature) -> 2048 bits`.
- Cache the SAX sweep per (instance, temperature). Never call a jitted SAX function inside a loop.

**Experiment E1:** 100+ seeds, same carrier grid, same challenge. Inter-PUF HD histogram, mean and sigma. Also intra-instance challenge-to-challenge HD.

**Done when** the uniqueness histogram exists as a plot with mean and sigma printed. This is the figure the paper is judged on.

---

## Phase 3 (Weeks 5-6): Intra-node reliability and ECC sizing

Closes open item 2.

**Experiment E2:** one instance, sweep temperature across TEC residual error, plus shot and thermal noise, laser wavelength drift, MRM bias drift. Output BER distribution and worst-case flips per 2048 bits.

**Experiment E3 (the mixed-signal figure):** HD and intra-node BER vs ADC resolution, sweeping 2/3/4/5 bits with analog SNR fixed at the Phase 1 budget. Shows where quantization stops helping because analog noise dominates. Justifies the 4-bit choice.

**Then:** size BCH(n, k, t) from worst-case BER with margin. Check remaining entropy after syndrome leakage. Confirm 256-bit K_PUF is still supportable.

**Done when** the ECC is a specified code with a stated error budget, not a placeholder.

---

## Phase 4 (Week 7): Data-plane link model and insertion penalty

Strictly one figure. Do not let this grow.

**Chain (per D3, D4):**
```
PRBS -> symbol map -> TX FFE -> driver (BW + nonlinearity) -> package/RF
  -> laser (RIN) -> MRM -> MZI 2x2 -> waveguide
  -> PD -> TIA -> AGC -> CTLE -> DFE -> slicer (CDR loop around slicer)
```
numpy, a few hundred lines. Validate against OptiCommPy / PyBERT, do not depend on them.

**Experiment E4:** BER vs received optical power, back-to-back vs with-security-branch. Report the delta in dB, not the absolute curve. Stop at the KP4 pre-FEC threshold (2.4e-4), roughly 1e6 to 1e7 symbols.

**Free side result:** same model gives tap-induced insertion loss, which closes the power-telemetry open item.

---

## Phase 5 (Weeks 8-9): Writing

Three pages. Figures first, then text around them. Start week 8 regardless of what is unfinished.

**Figure budget (4 + 1 table):**
1. Node architecture, enclave boundary, MZI two modes
2. Inter-PUF HD histogram (E1)
3. Reliability: BER vs temperature with ECC threshold drawn (E2)
4. Link penalty: BER vs received optical power (E4)

Table: link budget, both modes, with ENOB.

If a figure slot frees up, E3 (HD/BER vs ADC bits) is the first alternate.

The 10-step protocol becomes prose plus a small inset. The digital implementation is one sentence with two numbers.

**Also due 20 Oct:** the 35-word abstract. Draft it in week 8, not on the deadline.

---

## Phase 6 (Weeks 10-11): RTL, only if the paper is finished

Enough for one sentence: gate count and handshake latency.
- `fp_assembler`, SHA-256, HMAC, AES-256 unwrap, key register with zeroize
- Fuzzy extractor against the Phase 3 Python golden model
- cocotb + Verilator, Yosys to Sky130

Full area/latency/SVA tables go to the arXiv extended version and the journal follow-up.

---

## Week 12: Buffer

It will be used.

---

## Repo layout

```
morbius/
  photonics/
    contradc.py        # SAX, seeded variation
    linkbudget.py      # dB accounting, both modes
    noise.py           # shot/thermal/RIN -> SNR, SINAD, ENOB
    readout.py         # PD/TIA/LPF/ADC -> 2048-bit fingerprint
  dataplane/
    link.py            # PRBS -> ... -> BER
  crypto/
    fuzzy.py           # golden model, BCH
  experiments/
    e1_uniqueness.py
    e2_reliability.py
    e3_adc_resolution.py
    e4_link_penalty.py
  results/             # csv + png, one subdir per experiment
  si_phmpy/            # existing SimPy model
  rtl/
  paper/
  README.md            # D1-D5 decisions and parameter sources
```

Every experiment writes CSV before it writes a plot. Plots are regenerated from CSV, never from a live simulation.

---

## Cut order if time compresses

**Never cut:** E1 uniqueness, E2 reliability, ECC sizing, link budget table.
**Cut in this order:** RTL (Phase 6), E3 ADC sweep, tap-detection quantification, protocol formalization detail, energy (already out of scope).

Without E1 and E2 an optics reviewer has nothing to evaluate. Protect those two above everything else.

---

## Standing notes

- Simulation basis stated once, plainly, with parameter sources listed. Unsourced numbers presented as measured is the failure mode.
- Data-mode MZI penalty is framed as the price of interrogation isolation, not as overhead. The permanently-coupled passive alternative gets one honest sentence: it removes the switch but exposes the response on live traffic.
- Keep the data-plane simulator and the auth readout simulator separate. Share component models and noise functions, not the chain.
- MZI default (unpowered) state is data mode. No heater power or thermo-optic drift in normal operation.
- Post the full version to arXiv and cite it from the 3-page summary.
