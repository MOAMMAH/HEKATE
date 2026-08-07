"""
Send modulated challenges into the PUF and compare responses.

Pipeline (minimal receiver, enough for a key-difference figure):
  challenge bits -> Modulator -> PUF reflect -> Lorentzian demux
  -> |·|^2 -> sample at symbol offsets -> 4-bit Gray key

Compares two runs that share the same WDM grid / rates / device and
differ only in the challenge bit sequence. Plots the 2D key difference.

A CW (unmodulated) launch is also run as the steady-state control: with no
transitions the drop power is just (p_launch/N)*|r(f_i)|^2, one number per
channel, so both challenges collapse to the same key. Any A/B difference in
the sampled key therefore comes from the transient, not from the carriers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulator.modulator_model import C_LIGHT, ModulatedField, Modulator, WDMGrid
from puf.puf_model import ContraDCPUF, SimGrid

# Grating period for the readout device. The channel plan is the fixed system
# input, so carrier placement is set by retuning the grating, not by moving the
# grid. At the class default (305.72 nm) the drop peak is centred on 1550 nm and
# the 2 mm device is coupling-saturated, so |r|^2 is a flat plateau: the four
# carriers see 0.9992 +/- 0.00006 and the spectrum carries no structure. 312.5 nm
# moves the Bragg peak to ~1584 nm, putting the 3 dB band edge at ~1547.8 nm so
# the carriers straddle the shoulder and sit on the first ripple lobes.
# Pushing further (>=312.9 nm) drops individual channels into ripple nulls and
# fails the 10%-of-peak stopband check.
_LAMBDA0_EDGE = 312.5e-9

# Front-loaded sample offsets (fraction of symbol period) — catch grating ring-down.
_SAMPLE_OFFSETS = (0.1, 0.25, 0.45, 0.75)
# Late in the symbol, after the grating has settled as far as it is going to.
_STEADY_OFFSETS = (0.9,)
_ADC_BITS = 4
_DEMUX_FWHM_HZ = 30e9


def reflect_field(
    field: np.ndarray,
    simgrid: SimGrid,
    f0: float,
    puf: ContraDCPUF,
    H: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the PUF drop-port transfer function to a baseband field."""
    X = np.fft.fft(field)
    if H is None:
        H = puf.reflection_spectrum(f0 + simgrid.f)
    return np.fft.ifft(X * H)


def send_challenge(
    puf: ContraDCPUF,
    mod: Modulator,
    challenge: np.ndarray,
    simgrid: SimGrid,
    f0: float,
    H: np.ndarray | None = None,
) -> tuple[ModulatedField, np.ndarray]:
    """Modulate a challenge and pass it through the PUF. Returns (tx, rx)."""
    tx = mod.modulate(challenge, simgrid, f0)
    rx = reflect_field(tx.field, simgrid, f0, puf, H=H)
    return tx, rx


def demux_channels(
    field: np.ndarray,
    simgrid: SimGrid,
    f0: float,
    carrier_freqs: np.ndarray,
    fwhm: float = _DEMUX_FWHM_HZ,
) -> np.ndarray:
    """Lorentzian drop filters; return shape (n_channels, n)."""
    f_bb = simgrid.f  # baseband freq axis, Hz
    Y = np.fft.fft(field)
    out = np.empty((carrier_freqs.size, field.size), dtype=complex)
    for i, fi in enumerate(carrier_freqs):
        # Filter centered on carrier offset from f0, then shift channel to DC.
        offset = float(fi) - float(f0)
        H = 1.0 / (1.0 + 2j * (f_bb - offset) / fwhm)
        y_i = np.fft.ifft(Y * H)
        out[i] = y_i * np.exp(-1j * 2.0 * np.pi * offset * simgrid.t)
    return out


def cw_reference(
    puf: ContraDCPUF,
    mod: Modulator,
    simgrid: SimGrid,
    f0: float,
    H: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Steady-state control: unmodulated carriers, equal-split launch.

    Returns (measured_W, theory_W) per channel. Theory is the Section 6 gate
    (p_launch / n_ch) * |r(f_i)|^2. Measured runs the same demux as the data
    path, so residual adjacent-channel leakage through the Lorentzian shows up
    as a small excess rather than being hidden.
    """
    carriers = mod.grid.frequencies()
    n_ch = carriers.size
    amp = np.sqrt(mod.p_launch / n_ch)
    x = np.zeros(simgrid.n, dtype=complex)
    for offset in carriers - f0:
        x += amp * np.exp(1j * 2.0 * np.pi * offset * simgrid.t)

    rx = reflect_field(x, simgrid, f0, puf, H=H)
    ch = demux_channels(rx, simgrid, f0, carriers)
    # Average the middle half: the FFT wrap disturbs both ends of the window.
    lo, hi = simgrid.n // 4, 3 * simgrid.n // 4
    measured = np.mean(np.abs(ch[:, lo:hi]) ** 2, axis=1)
    theory = (mod.p_launch / n_ch) * np.abs(puf.reflection_spectrum(carriers)) ** 2
    return measured, theory


def sample_response(
    channel_fields: np.ndarray,
    symbol_edges: np.ndarray,
    sps: int,
    offsets: tuple[float, ...] = _SAMPLE_OFFSETS,
) -> np.ndarray:
    """
    Square-law + sample at fractional symbol offsets.

    Returns raw_samples with shape (n_channels, n_symbols * n_offsets).
    """
    n_ch, n = channel_fields.shape
    power = np.abs(channel_fields) ** 2
    n_sym = int(symbol_edges.size)
    n_off = len(offsets)
    samples = np.empty((n_ch, n_sym * n_off), dtype=float)
    for ch in range(n_ch):
        col = 0
        for edge in symbol_edges:
            for frac in offsets:
                idx = int(edge) + int(round(frac * sps))
                idx = min(max(idx, 0), n - 1)
                samples[ch, col] = power[ch, idx]
                col += 1
    return samples


def quantize_gray(
    raw: np.ndarray,
    p_launch_ref: float,
    bits: int = _ADC_BITS,
    full_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize by launch ref, clip to full_scale, uniform quantize, Gray-encode.

    Returns (gray_codes, levels, full_scale_used) with same shape as raw.
    gray_codes are integer in [0, 2**bits).
    """
    n_levels = 1 << bits
    x = raw / max(float(p_launch_ref), 1e-30)
    if full_scale is None:
        # Freeze from this burst's peak (enrollment-style for a single demo).
        full_scale = float(np.max(x)) * 1.05 + 1e-18
    x = np.clip(x / full_scale, 0.0, 1.0)
    levels = np.clip(np.floor(x * n_levels).astype(int), 0, n_levels - 1)
    # Binary-reflected Gray: g = v ^ (v >> 1)
    gray = levels ^ (levels >> 1)
    return gray, levels, float(full_scale)


def gray_bit_flips(key_a: np.ndarray, key_b: np.ndarray) -> np.ndarray:
    """Per-sample Hamming weight of the Gray-code XOR, in [0, _ADC_BITS]."""
    tmp = np.bitwise_xor(key_a.astype(int), key_b.astype(int))
    flips = np.zeros(tmp.shape, dtype=float)
    while np.any(tmp):
        flips += tmp & 1
        tmp >>= 1
    return flips


def key_difference_plot(
    key_a: np.ndarray,
    key_b: np.ndarray,
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    out_path: Path,
    full_scale_power: float,
) -> dict:
    """2D plots: ADC-relative |Δ| and Gray-code XOR (channels × sample index)."""
    abs_diff = np.abs(raw_a - raw_b)
    # Fraction of the quantizer full-scale (same units as digitized key).
    # This is the scale that actually decides whether bits flip.
    rel_fs = abs_diff / max(float(full_scale_power), 1e-30)
    mean_amp = 0.5 * (np.abs(raw_a) + np.abs(raw_b))
    rel_mean = abs_diff / np.maximum(mean_amp, 1e-30)

    bit_flips = gray_bit_flips(key_a, key_b)

    n_ch = key_a.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)

    # Show keys on the same normalized scale the ADC uses (power / full_scale).
    norm_a = raw_a / max(float(full_scale_power), 1e-30)
    norm_b = raw_b / max(float(full_scale_power), 1e-30)
    vmax_n = float(np.percentile(np.concatenate([norm_a, norm_b]), 99.5))
    vmax_n = max(vmax_n, 1e-12)

    im_a = axes[0].imshow(
        norm_a, aspect="auto", origin="lower", interpolation="nearest",
        vmin=0.0, vmax=vmax_n,
    )
    axes[0].set_title("Key A (power / full-scale)")
    axes[0].set_xlabel("sample index")
    axes[0].set_ylabel("WDM channel")
    axes[0].set_yticks(range(n_ch))
    fig.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04)

    im_b = axes[1].imshow(
        norm_b, aspect="auto", origin="lower", interpolation="nearest",
        vmin=0.0, vmax=vmax_n,
    )
    axes[1].set_title("Key B (power / full-scale)")
    axes[1].set_xlabel("sample index")
    axes[1].set_ylabel("WDM channel")
    axes[1].set_yticks(range(n_ch))
    fig.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)

    vmax_d = float(np.percentile(rel_fs, 99.5))
    vmax_d = max(vmax_d, 1e-12)
    im0 = axes[2].imshow(
        rel_fs,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax_d,
    )
    axes[2].set_title("|A-B| / full-scale (auto clim)")
    axes[2].set_xlabel("sample index")
    axes[2].set_ylabel("WDM channel")
    axes[2].set_yticks(range(n_ch))
    fig.colorbar(im0, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Separate bit-flip figure kept on the same path's sibling for clarity.
    out_bits = out_path.with_name(out_path.stem + "_bits.png")
    fig2, ax2 = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    im1 = ax2.imshow(
        bit_flips,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=_ADC_BITS,
    )
    ax2.set_title("Gray-key bit flips (popcount of XOR)")
    ax2.set_xlabel("sample index (symbol × offset)")
    ax2.set_ylabel("WDM channel")
    ax2.set_yticks(range(n_ch))
    fig2.colorbar(im1, ax=ax2, fraction=0.046, pad=0.04)
    fig2.savefig(out_bits, dpi=150)
    plt.close(fig2)

    nibble_mismatch = float(np.mean(key_a != key_b))
    mean_bit_flips = float(np.mean(bit_flips))
    rms_analog = float(np.sqrt(np.mean(abs_diff**2)))
    rms_signal = float(np.sqrt(np.mean(mean_amp**2)))
    lsb = full_scale_power / (1 << _ADC_BITS)
    return {
        "rms_analog_diff": rms_analog,
        "rms_rel_to_signal": rms_analog / max(rms_signal, 1e-30),
        "rms_rel_to_full_scale": rms_analog / max(float(full_scale_power), 1e-30),
        "frac_above_1_lsb": float(np.mean(abs_diff > lsb)),
        "mean_rel_to_local": float(np.mean(rel_mean)),
        "nibble_mismatch_frac": nibble_mismatch,
        "mean_bit_flips_per_sample": mean_bit_flips,
        "plot": str(out_path),
        "plot_bits": str(out_bits),
    }


def extract_key(
    rx: np.ndarray,
    tx: ModulatedField,
    mod: Modulator,
    simgrid: SimGrid,
    f0: float,
    full_scale: float | None = None,
    offsets: tuple[float, ...] = _SAMPLE_OFFSETS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """rx field -> demux -> sample -> Gray key. Returns (gray, raw, full_scale)."""
    carriers = mod.grid.frequencies()
    ch = demux_channels(rx, simgrid, f0, carriers)
    sps = max(1, int(round(mod.symbol_period_s * simgrid.fs)))
    raw = sample_response(ch, tx.symbol_edges, sps, offsets=offsets)
    gray, _levels, fs = quantize_gray(raw, tx.p_launch_ref, full_scale=full_scale)
    return gray, raw, fs


def _heat(ax, data, title, vmin=None, vmax=None, cmap=None, cbar_fig=None):
    """Shared channel-vs-sample heatmap styling."""
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("sample index", fontsize=8)
    ax.set_ylabel("WDM channel", fontsize=8)
    ax.set_yticks(range(data.shape[0]))
    ax.tick_params(labelsize=7)
    if cbar_fig is not None:
        cb = cbar_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
    return im


def summary_figure(
    puf: ContraDCPUF,
    mod: Modulator,
    sim: SimGrid,
    f0: float,
    challenge_a: np.ndarray,
    challenge_b: np.ndarray,
    pow_a: np.ndarray,
    pow_b: np.ndarray,
    keys: dict,
    cw: tuple[np.ndarray, np.ndarray],
    full_scale_power: float,
    out_path: Path,
) -> None:
    """One figure: PUF response, steady state, challenge bits, and keys."""
    carriers_nm = mod.grid.wavelengths() * 1e9
    n_ch = mod.grid.n_channels
    sps = max(1, int(round(mod.symbol_period_s * sim.fs)))

    fig, axes = plt.subplots(3, 4, figsize=(19, 11), constrained_layout=True)

    # --- Row 0: PUF drop-port response and the steady-state control ---------
    bragg_nm = puf.Lambda0 * (puf.n1_0 + puf.n2_0) * 1e9
    wl_wide, hd_db = puf.drop_port_response(
        np.linspace(min(1480.0, carriers_nm.min() - 40.0), bragg_nm + 60.0, 801)
    )
    ax = axes[0, 0]
    ax.plot(wl_wide, hd_db, lw=1.0)
    for wl in carriers_nm:
        ax.axvline(wl, color="crimson", ls="--", lw=0.8)
    ax.axvline(bragg_nm, color="seagreen", ls="-", lw=1.0, label="Bragg peak")
    ax.set_title("PUF drop port $|r|^2$ (dB), carriers dashed", fontsize=9)
    ax.set_xlabel("wavelength (nm)", fontsize=8)
    ax.set_ylabel("$|r|^2$ (dB)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Wide enough to show the plateau, the band edge, and the ripples the
    # carriers now sit on — a few-nm zoom would hide the whole point.
    ax = axes[0, 1]
    lo = min(carriers_nm.min() - 12.0, bragg_nm - 45.0)
    wl_zoom = np.linspace(lo, carriers_nm.max() + 16.0, 700)
    r_zoom = np.abs(puf.reflection_spectrum(C_LIGHT / (wl_zoom * 1e-9))) ** 2
    ax.plot(wl_zoom, r_zoom, lw=1.0)
    r_ch = np.abs(puf.reflection_spectrum(mod.grid.frequencies())) ** 2
    ax.plot(carriers_nm, r_ch, "o", color="crimson", ms=6, zorder=3, label="carriers")
    ax.axhline(0.5 * r_zoom.max(), color="0.5", ls=":", lw=0.9, label="3 dB")
    ax.set_title(
        f"Band edge at the WDM grid — $|r|^2$ = {r_ch.min():.2f}..{r_ch.max():.2f}",
        fontsize=9,
    )
    ax.set_xlabel("wavelength (nm)", fontsize=8)
    ax.set_ylabel("$|r|^2$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    measured, theory = cw
    ax = axes[0, 2]
    idx = np.arange(n_ch)
    ax.bar(idx - 0.19, theory * 1e3, 0.38, label=r"theory $\frac{P}{N}|r|^2$")
    ax.bar(idx + 0.19, measured * 1e3, 0.38, label="measured (CW)")
    ax.set_title("Steady state: CW drop power per channel", fontsize=9)
    ax.set_xlabel("WDM channel", fontsize=8)
    ax.set_ylabel("power (mW)", fontsize=8)
    ax.set_xticks(idx)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    # Channel 0 waveform: the transient is what makes the two keys differ.
    n_show = min(6 * sps, pow_a.shape[1])
    t_ps = sim.t[:n_show] * 1e12
    ax = axes[0, 3]
    ax.plot(t_ps, pow_a[0, :n_show] * 1e3, lw=1.0, label="challenge A")
    ax.plot(t_ps, pow_b[0, :n_show] * 1e3, lw=1.0, label="challenge B")
    ax.axhline(measured[0] * 1e3, color="k", ls=":", lw=1.0, label="CW steady state")
    for k in range(n_show // sps + 1):
        ax.axvline(k * sps * sim.dt * 1e12, color="0.8", lw=0.6, zorder=0)
    for frac in _SAMPLE_OFFSETS:
        ax.plot(
            [(k + frac) * sps * sim.dt * 1e12 for k in range(n_show // sps)],
            [
                pow_a[0, min(int((k + frac) * sps), n_show - 1)] * 1e3
                for k in range(n_show // sps)
            ],
            "v",
            ms=4,
            color="crimson",
            zorder=3,
        )
    ax.set_title("Ch 0 drop power vs time (ADC instants ▾)", fontsize=9)
    ax.set_xlabel("time (ps)", fontsize=8)
    ax.set_ylabel("power (mW)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # --- Row 1: challenge bits and the resulting Gray-key bit flips ---------
    _heat(axes[1, 0], challenge_a, "Challenge A bits", 0, 1, "binary", fig)
    axes[1, 0].set_xlabel("bit index", fontsize=8)
    _heat(axes[1, 1], challenge_b, "Challenge B bits", 0, 1, "binary", fig)
    axes[1, 1].set_xlabel("bit index", fontsize=8)
    _heat(
        axes[1, 2],
        np.bitwise_xor(challenge_a, challenge_b),
        f"Challenge A XOR B (HD = {np.mean(challenge_a != challenge_b):.3f})",
        0,
        1,
        "binary",
        fig,
    )
    axes[1, 2].set_xlabel("bit index", fontsize=8)
    _heat(
        axes[1, 3],
        keys["flips_transient"],
        f"Gray-key bit flips, transient (mean {keys['flips_transient'].mean():.2f})",
        0,
        _ADC_BITS,
        None,
        fig,
    )

    # --- Row 2: keys, transient difference, steady-state difference ---------
    norm_a = keys["raw_a"] / max(full_scale_power, 1e-30)
    norm_b = keys["raw_b"] / max(full_scale_power, 1e-30)
    vmax_n = max(float(np.percentile(np.r_[norm_a.ravel(), norm_b.ravel()], 99.5)), 1e-12)
    _heat(axes[2, 0], norm_a, "Key A (power / full-scale)", 0.0, vmax_n, None, fig)
    _heat(axes[2, 1], norm_b, "Key B (power / full-scale)", 0.0, vmax_n, None, fig)

    d_tr = np.abs(keys["raw_a"] - keys["raw_b"]) / max(full_scale_power, 1e-30)
    d_ss = np.abs(keys["raw_a_ss"] - keys["raw_b_ss"]) / max(full_scale_power, 1e-30)
    vmax_d = max(float(np.percentile(np.r_[d_tr.ravel(), d_ss.ravel()], 99.5)), 1e-12)
    _heat(
        axes[2, 2],
        d_tr,
        f"|A-B| / full-scale, transient (RMS {np.sqrt((d_tr**2).mean()):.3f})",
        0.0,
        vmax_d,
        None,
        fig,
    )
    _heat(
        axes[2, 3],
        d_ss,
        f"|A-B| / full-scale, steady state (RMS {np.sqrt((d_ss**2).mean()):.3f})",
        0.0,
        vmax_d,
        None,
        fig,
    )

    fig.suptitle(
        f"Contra-DC PUF readout — device {puf.device_id}, L = {puf.L*1e3:.2f} mm, "
        rf"$\tau_{{max}}$ = {puf.tau_max*1e12:.1f} ps, {mod.symbol_rate/1e9:.0f} GBd "
        f"{mod.fmt.upper()}, T = {mod.symbol_period_s*1e12:.1f} ps, "
        f"{mod.p_launch/mod.grid.n_channels*1e3:.0f} mW/ch",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    # Same everything; only the bit sequence changes between A and B.
    grid = WDMGrid()
    sim = SimGrid(fs=4e12, n=16384)
    f0 = C_LIGHT / 1550e-9
    # Longer device so symbol period is not ≫ τ_max (otherwise Δkey ≈ 0).
    # Period retuned so the carriers land on the band edge, not the flat top.
    puf = ContraDCPUF(device_id=0, L=2.0e-3, n_sect=200, Lambda0=_LAMBDA0_EDGE)
    p_per_ch = 10e-3  # 10 mW per WDM channel
    mod = Modulator(
        grid=grid,
        symbol_rate=50e9,
        fmt="nrz",
        p_launch=p_per_ch * grid.n_channels,  # total; equal-split → ~10 mW/ch
    )

    n_bits = 32
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(1)
    challenge_a = Modulator.random_challenge(grid.n_channels, n_bits, rng_a)
    challenge_b = Modulator.random_challenge(grid.n_channels, n_bits, rng_b)
    assert not np.array_equal(challenge_a, challenge_b)

    # Same H(f) for both challenges — device/grid unchanged.
    H = puf.reflection_spectrum(f0 + sim.f)
    tx_a, rx_a = send_challenge(puf, mod, challenge_a, sim, f0, H=H)
    tx_b, rx_b = send_challenge(puf, mod, challenge_b, sim, f0, H=H)

    key_a, raw_a, fs_a = extract_key(rx_a, tx_a, mod, sim, f0)
    # Same full-scale for both so the difference is not a gain artifact.
    key_b, raw_b, _ = extract_key(rx_b, tx_b, mod, sim, f0, full_scale=fs_a)

    # Steady-state sampling (late in the symbol) on the same bursts, same scale.
    key_a_ss, raw_a_ss, _ = extract_key(
        rx_a, tx_a, mod, sim, f0, full_scale=fs_a, offsets=_STEADY_OFFSETS
    )
    key_b_ss, raw_b_ss, _ = extract_key(
        rx_b, tx_b, mod, sim, f0, full_scale=fs_a, offsets=_STEADY_OFFSETS
    )

    # CW control: no modulation at all, so only |r(f_i)| can matter.
    cw_measured, cw_theory = cw_reference(puf, mod, sim, f0, H=H)

    # full_scale in quantize_gray is on (power / p_launch_ref); convert back to watts.
    full_scale_power = fs_a * tx_a.p_launch_ref

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "key_difference.png"
    stats = key_difference_plot(
        key_a, key_b, raw_a, raw_b, out_path, full_scale_power=full_scale_power
    )

    carriers = mod.grid.frequencies()
    pow_a = np.abs(demux_channels(rx_a, sim, f0, carriers)) ** 2
    pow_b = np.abs(demux_channels(rx_b, sim, f0, carriers)) ** 2
    summary_path = out_dir / "puf_readout_summary.png"
    summary_figure(
        puf,
        mod,
        sim,
        f0,
        challenge_a,
        challenge_b,
        pow_a,
        pow_b,
        {
            "raw_a": raw_a,
            "raw_b": raw_b,
            "raw_a_ss": raw_a_ss,
            "raw_b_ss": raw_b_ss,
            "flips_transient": gray_bit_flips(key_a, key_b),
        },
        (cw_measured, cw_theory),
        full_scale_power,
        summary_path,
    )

    hd_bits = float(np.mean(challenge_a != challenge_b))
    sps = max(1, int(round(mod.symbol_period_s * sim.fs)))
    burst = n_bits * sps
    p_ch = [
        float(np.mean(np.abs(tx_a.channel_fields[i, :burst]) ** 2))
        for i in range(grid.n_channels)
    ]
    print(f"WDM λ/nm = {grid.wavelengths()*1e9}")
    print(f"p_launch total = {mod.p_launch*1e3:.1f} mW  ({p_per_ch*1e3:.1f} mW/ch target)")
    print(f"p_launch measured /ch (burst) = {[f'{p*1e3:.2f} mW' for p in p_ch]}")
    print(f"PUF L={puf.L*1e3:.2f} mm, tau_max={puf.tau_max*1e12:.1f} ps")
    print(f"symbol_rate={mod.symbol_rate:.3e} baud, T={mod.symbol_period_s*1e12:.1f} ps")
    print(f"challenge A/B Hamming frac (bits) = {hd_bits:.3f}")
    print(f"key shape = {key_a.shape}  (channels × samples)")
    print(f"analog RMS |Δ|           = {stats['rms_analog_diff']:.6e} W")
    print(f"RMS |Δ| / RMS signal     = {stats['rms_rel_to_signal']:.3f}")
    print(f"RMS |Δ| / ADC full-scale = {stats['rms_rel_to_full_scale']:.3f}")
    print(f"frac samples |Δ| > 1 LSB = {stats['frac_above_1_lsb']:.3f}")
    print(f"nibble mismatch          = {stats['nibble_mismatch_frac']:.3f}")
    print(f"mean bit flips/samp      = {stats['mean_bit_flips_per_sample']:.3f}")

    print("--- steady state (control) ---")
    dev_pct = 100.0 * np.abs(cw_measured - cw_theory) / np.maximum(cw_theory, 1e-30)
    print(f"CW theory /ch (mW)  = {[f'{p*1e3:.4f}' for p in cw_theory]}")
    print(f"CW measured /ch(mW) = {[f'{p*1e3:.4f}' for p in cw_measured]}")
    print(f"CW deviation        = {[f'{d:.2f}%' for d in dev_pct]} (demux leakage)")
    ss_flips = gray_bit_flips(key_a_ss, key_b_ss)
    print(f"steady-state nibble mismatch = {np.mean(key_a_ss != key_b_ss):.3f}")
    print(f"steady-state bit flips/samp  = {ss_flips.mean():.3f}")
    print(f"transient bit flips/samp     = {gray_bit_flips(key_a, key_b).mean():.3f}")

    print(f"wrote {stats['plot']}")
    print(f"wrote {stats['plot_bits']}")
    print(f"wrote {summary_path}")

    warns = mod.check_carriers_in_stopband(puf)
    print(f"stopband check: {warns if warns else '(none)'}")
