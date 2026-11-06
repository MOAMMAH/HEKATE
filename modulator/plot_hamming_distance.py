"""
Inter-challenge Hamming distance for the same PUF: single-mode vs 3-mode.

Same device_id, same WDM grid / challenges; only the modal model changes.
Writes two separate figures under modulator/out/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modulator.modulator_model import C_LIGHT, Modulator, WDMGrid
from modulator.send_to_puf import (
    _ADC_BITS,
    _GAP,
    _LAMBDA0_EDGE,
    _N_MODES_MAX,
    _W1,
    _W2,
    extract_key,
    gray_bit_flips,
    send_challenge,
)
from puf.modes import slab_modes
from puf.puf_model import ContraDCPUF, SimGrid


def _build_puf(mode: str) -> ContraDCPUF:
    kwargs = dict(device_id=0, L=2.0e-3, Lambda0=_LAMBDA0_EDGE)
    if mode == "single":
        return ContraDCPUF(mode_model="single", n_sect=200, **kwargs)
    m1 = slab_modes(_W1, 1550e-9, _N_MODES_MAX)
    m2 = slab_modes(_W2, 1550e-9, _N_MODES_MAX)
    lam_center = 1550e-9 / (float(m1.n_eff[0]) + float(m2.n_eff[0]))
    kwargs["Lambda0"] = lam_center * (_LAMBDA0_EDGE / 305.72e-9)
    return ContraDCPUF(
        mode_model="auto",
        w1=_W1,
        w2=_W2,
        gap=_GAP,
        n_modes_max=_N_MODES_MAX,
        n_sect=80,
        **kwargs,
    )


def _run_keys(puf: ContraDCPUF, sim: SimGrid, mod: Modulator, f0: float):
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(1)
    n_bits = 32
    challenge_a = Modulator.random_challenge(mod.n_channels, n_bits, rng_a)
    challenge_b = Modulator.random_challenge(mod.n_channels, n_bits, rng_b)

    H = None
    if puf.M2 <= 1:
        H = puf.reflection_spectrum(f0 + sim.f)
    tx_a, rx_a = send_challenge(puf, mod, challenge_a, sim, f0, H=H)
    tx_b, rx_b = send_challenge(puf, mod, challenge_b, sim, f0, H=H)
    key_a, _raw_a, fs_a = extract_key(rx_a, tx_a, mod, sim, f0)
    key_b, _raw_b, _ = extract_key(rx_b, tx_b, mod, sim, f0, full_scale=fs_a)
    return key_a, key_b, challenge_a, challenge_b


def hamming_distance_plot(
    key_a: np.ndarray,
    key_b: np.ndarray,
    *,
    title: str,
    out_path: Path,
    mode_label: str,
) -> dict:
    """
    Two-panel figure: per-sample HD heatmap + HD histogram.

    HD is the Gray-code bit Hamming distance in [0, ADC_BITS] per sample,
    also reported as a fractional bit HD over the full key.
    """
    flips = gray_bit_flips(key_a, key_b)  # (n_ch, n_samp), each in 0..ADC_BITS
    n_ch, n_samp = flips.shape
    total_bits = n_ch * n_samp * _ADC_BITS
    hd_bits = float(np.sum(flips))
    hd_frac = hd_bits / max(total_bits, 1)
    # Fractional HD per sample (normalize by ADC width).
    hd_samp = flips / _ADC_BITS

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    ax = axes[0]
    im = ax.imshow(
        hd_samp,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
    )
    ax.set_title(
        f"{mode_label}: per-sample fractional HD\n"
        f"mean HD = {hd_frac:.3f}  ({hd_bits:.0f}/{total_bits} bits)",
        fontsize=10,
    )
    ax.set_xlabel("sample index")
    ax.set_ylabel("WDM channel")
    ax.set_yticks(range(n_ch))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"HD / {_ADC_BITS} bits")

    ax = axes[1]
    # Histogram of per-sample bit flips (0..ADC_BITS)
    counts = np.bincount(flips.astype(int).ravel(), minlength=_ADC_BITS + 1)
    counts = counts / max(counts.sum(), 1)
    ax.bar(np.arange(_ADC_BITS + 1), counts, width=0.8, color="steelblue", edgecolor="k")
    ax.axvline(hd_frac * _ADC_BITS, color="crimson", ls="--", lw=1.5, label=f"mean = {hd_frac*_ADC_BITS:.2f} bits")
    ax.set_xlabel(f"Hamming distance (bits / sample, max {_ADC_BITS})")
    ax.set_ylabel("fraction of samples")
    ax.set_xticks(range(_ADC_BITS + 1))
    ax.set_ylim(0, max(0.05, float(counts.max()) * 1.15))
    ax.set_title(f"{mode_label}: HD distribution\noverall bit HD = {hd_frac:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=12)
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "hd_frac": hd_frac,
        "hd_bits": hd_bits,
        "total_bits": total_bits,
        "mean_flips_per_sample": float(np.mean(flips)),
        "plot": str(out_path),
    }


def main() -> None:
    grid = WDMGrid()
    f0 = C_LIGHT / 1550e-9
    p_per_ch = 10e-3
    mod = Modulator(
        grid=grid,
        symbol_rate=50e9,
        fmt="nrz",
        p_launch=p_per_ch * grid.n_channels,
    )
    out_dir = Path(__file__).resolve().parent / "out"

    cases = [
        ("single", "Single-mode (M=1)", 16384, "hamming_distance_single.png"),
        ("auto_m3", "3-mode (M1=M2=3)", 8192, "hamming_distance_auto_m3.png"),
    ]

    for tag, label, sim_n, fname in cases:
        print(f"=== {label} ===")
        mode = "single" if tag == "single" else "auto"
        puf = _build_puf(mode)
        sim = SimGrid(fs=4e12, n=sim_n)
        print(
            f"mode_model={puf.mode_model}, M1={puf.M1}, M2={puf.M2}, "
            f"sim.n={sim.n}, n_sect={puf.n_sect}"
        )
        key_a, key_b, ch_a, ch_b = _run_keys(puf, sim, mod, f0)
        challenge_hd = float(np.mean(ch_a != ch_b))
        stats = hamming_distance_plot(
            key_a,
            key_b,
            title=(
                f"Same PUF (device_id={puf.device_id}) — inter-challenge key HD\n"
                f"challenge bit HD = {challenge_hd:.3f}"
            ),
            out_path=out_dir / fname,
            mode_label=label,
        )
        print(
            f"key HD = {stats['hd_frac']:.3f}  "
            f"({stats['hd_bits']:.0f}/{stats['total_bits']} bits), "
            f"mean flips/sample = {stats['mean_flips_per_sample']:.3f}"
        )
        print(f"wrote {stats['plot']}")


if __name__ == "__main__":
    main()
