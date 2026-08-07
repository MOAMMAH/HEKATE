"""
PUF Design Parameter Exploration
=================================
Sweeps the key Contra-DC design parameters and shows how each one changes the
drop port transfer function H_drop(λ).  Only the PUF optical waveform is
modelled here — no WDM key extraction, no chiplet simulation.

Run:
    python puf_design_exploration.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.signal import lfilter

# ---------------------------------------------------------------------------
# Physical constants & wavelength sweep
# ---------------------------------------------------------------------------
C_LIGHT = 3e8

WL_NM = np.linspace(1490, 1560, 300)
WL_M = WL_NM * 1e-9
N_SECT = 80

# ---------------------------------------------------------------------------
# Baseline device parameters  (same as the paper figure)
# ---------------------------------------------------------------------------
P_BASE = dict(
    n1_0=2.580,
    n2_0=2.490,
    ng1=3.65,
    ng2=3.55,
    wl0=1.521e-6,
    kappa11=5000.0,
    kappa12=16300.0,
    kappa22=6200.0,
    loss_dBcm=5.0,
    Lambda0=300e-9,
    L=210e-6,
    ka1=0.99,
    ka2=0.14,
)

# Baseline technique flags — "all combined" from the paper
TECH_BASE = dict(
    kappa_mult=3.0,
    chirp=dict(on=True, rate=1.2),
    antiapo=dict(on=True, center_dip=0.90, edge_boost=2.0),
    sampled=dict(on=True, period=15.0, depth=1.0),
    phase_shifts=dict(on=True, n=4),
    cascade=dict(on=True, delta_Lambda=2.5),
)


# ---------------------------------------------------------------------------
# Transfer-matrix engine
# ---------------------------------------------------------------------------

def _alpha(p):
    return p["loss_dBcm"] * 100 / (20 * np.log10(np.e))


def _section_C(wl, n1, n2, k12, Lambda, dL, z_c, p):
    alpha = _alpha(p)
    b1 = 2 * np.pi * n1 / wl - 1j * alpha
    b2 = 2 * np.pi * n2 / wl - 1j * alpha
    pL = np.pi / Lambda
    d1 = b1 - pL
    d2 = b2 - pL
    r1 = np.real(b1)
    r2 = np.real(b2)
    p11 = np.exp(2j * r1 * z_c)
    p22 = np.exp(2j * r2 * z_c)
    p12 = np.exp(1j * (r1 + r2) * z_c)
    S1 = np.diag([1j * d1, 1j * d2, -1j * d1, -1j * d2])
    S2 = np.array(
        [
            [-1j * d1,            0,            -1j * p["kappa11"] * p11, -1j * k12 * p12],
            [0,                  -1j * d2,      -1j * k12 * p12,          -1j * p["kappa22"] * p22],
            [1j * p["kappa11"] * np.conj(p11), 1j * k12 * np.conj(p12),  1j * d1, 0],
            [1j * k12 * np.conj(p12), 1j * p["kappa22"] * np.conj(p22),  0,       1j * d2],
        ],
        dtype=complex,
    )
    return expm(S1 * dL) @ expm(S2 * dL)


def _phase_shift_mat():
    ph = np.pi / 2
    return np.diag(
        [np.exp(-1j * ph), np.exp(-1j * ph), np.exp(1j * ph), np.exp(1j * ph)]
    ).astype(complex)


def _bvp_solve(M, p):
    ka1 = p["ka1"]
    ka2 = p["ka2"]
    p_in = ka1 ** 2 + ka2 ** 2
    ein = np.array([ka1, ka2], dtype=complex)
    try:
        ef = np.linalg.solve(M[0:2, 0:2], ein)
        eb = M[2:4, 0:2] @ ef
        td = (abs(eb[0]) ** 2 + abs(eb[1]) ** 2) / p_in
        return float(np.clip(td, 0, 1)), float(np.angle(eb[0] + eb[1]))
    except np.linalg.LinAlgError:
        return 0.0, 0.0


def _build_profiles(tech, p):
    n = N_SECT
    l_tot = p["L"]
    d_l = l_tot / n
    z = np.linspace(0, l_tot, n, endpoint=False) + d_l / 2
    z_norm = (z - l_tot / 2) / (l_tot / 2)
    km = tech["kappa_mult"]

    if tech["antiapo"]["on"]:
        cd = tech["antiapo"]["center_dip"]
        eb = tech["antiapo"]["edge_boost"]
        apo = (1 - cd * np.cos(np.pi * z_norm / 2) ** 2) * (
            1 + eb * np.sin(np.pi * z_norm / 2) ** 2
        )
        apo = np.clip(apo, 0.0, 10.0)
    else:
        apo = np.ones(n)

    if tech["sampled"]["on"]:
        period_m = tech["sampled"]["period"] * 1e-6
        depth = tech["sampled"]["depth"]
        samp = np.abs(1.0 + depth * np.cos(2 * np.pi * z / period_m))
    else:
        samp = np.ones(n)

    kappa12_arr = p["kappa12"] * km * apo * samp

    if tech["chirp"]["on"]:
        rate_m = tech["chirp"]["rate"] * 1e-9
        lambda_arr = p["Lambda0"] + np.linspace(-rate_m / 2, rate_m / 2, n)
    else:
        lambda_arr = np.full(n, p["Lambda0"])

    if tech["phase_shifts"]["on"]:
        n_ps = max(1, int(round(tech["phase_shifts"]["n"])))
        ps_positions = list(np.linspace(0, l_tot, n_ps + 2)[1:-1])
    else:
        ps_positions = []

    return kappa12_arr, lambda_arr, ps_positions, z


def _transfer_one_wl(wl, tech, k12_arr, lam_arr, ps_pos, z_c, d_l, p):
    n1 = p["n1_0"] - (p["ng1"] - p["n1_0"]) / p["wl0"] * (wl - p["wl0"])
    n2 = p["n2_0"] - (p["ng2"] - p["n2_0"]) / p["wl0"] * (wl - p["wl0"])
    m = np.eye(4, dtype=complex)
    for i in range(len(z_c)):
        for z_ps in ps_pos:
            if (i == 0 and z_ps <= z_c[0]) or (i > 0 and z_c[i - 1] < z_ps <= z_c[i]):
                m = _phase_shift_mat() @ m
        m = _section_C(wl, n1, n2, k12_arr[i], lam_arr[i], d_l, z_c[i], p) @ m
    td, ph = _bvp_solve(m, p)
    return np.sqrt(td) * np.exp(1j * ph)


def simulate(tech, p=None):
    """Return H_drop in dB (normalized) over WL_M."""
    if p is None:
        p = P_BASE
    k12_arr, lam_arr, ps_pos, z = _build_profiles(tech, p)
    d_l = p["L"] / N_SECT

    h = np.zeros(len(WL_M), dtype=complex)
    for wi, wl in enumerate(WL_M):
        h[wi] = _transfer_one_wl(wl, tech, k12_arr, lam_arr, ps_pos, z, d_l, p)

    mx = float(np.max(np.abs(h)))
    if mx < 1e-9:
        mx = 1.0
    h /= mx
    hd_db = 10 * np.log10(np.clip(np.abs(h) ** 2, 1e-12, 1.0))
    return hd_db


# ---------------------------------------------------------------------------
# Helper: deep-copy + patch a nested key
# ---------------------------------------------------------------------------
import copy


def _patch_tech(key_path, value, base=None):
    """Return a copy of TECH_BASE with one leaf patched.

    key_path: tuple, e.g. ("chirp", "rate") or ("kappa_mult",)
    """
    if base is None:
        base = TECH_BASE
    t = copy.deepcopy(base)
    if len(key_path) == 1:
        t[key_path[0]] = value
    else:
        t[key_path[0]][key_path[1]] = value
    return t


def _patch_p(key, value):
    """Return a copy of P_BASE with one entry changed."""
    p = dict(P_BASE)
    p[key] = value
    return p


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------
# Each entry: (panel_title, sweep_values, label_fmt, simulate_fn)
# simulate_fn(val) → hd_db array

CMAP = plt.cm.viridis

sweeps = [
    # 1. Coupling coefficient multiplier  (κ_mult)
    dict(
        title="Coupling strength  κ_mult",
        xlabel="Wavelength (nm)",
        ylabel="H_drop (dB)",
        values=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        label_fn=lambda v: f"κ×{v:.1f}",
        sim_fn=lambda v: simulate(_patch_tech(("kappa_mult",), v)),
    ),
    # 2. Chirp rate
    dict(
        title="Grating chirp rate (nm/cm equivalent)",
        xlabel="Wavelength (nm)",
        ylabel="H_drop (dB)",
        values=[0.0, 0.3, 0.6, 1.0, 1.5, 2.5],
        label_fn=lambda v: f"rate={v:.1f}" if v > 0 else "no chirp",
        sim_fn=lambda v: simulate(
            _patch_tech(("chirp", "on"), v > 0,
                        _patch_tech(("chirp", "rate"), max(v, 0.01)))
        ) if v > 0 else simulate(_patch_tech(("chirp", "on"), False)),
    ),
    # 3. Anti-apodisation center dip
    dict(
        title="Anti-apodisation — centre dip depth",
        xlabel="Wavelength (nm)",
        ylabel="H_drop (dB)",
        values=[0.0, 0.3, 0.6, 0.9, 1.2, 1.5],
        label_fn=lambda v: f"cd={v:.1f}",
        sim_fn=lambda v: simulate(_patch_tech(("antiapo", "center_dip"), v)),
    ),
    # 4. Anti-apodisation edge boost
    dict(
        title="Anti-apodisation — edge boost",
        xlabel="Wavelength (nm)",
        ylabel="H_drop (dB)",
        values=[0.0, 0.5, 1.0, 2.0, 3.5, 5.0],
        label_fn=lambda v: f"eb={v:.1f}",
        sim_fn=lambda v: simulate(_patch_tech(("antiapo", "edge_boost"), v)),
    ),
    # 5. Sampled-grating period
    dict(
        title="Sampled-grating period (µm)",
        xlabel="Wavelength (nm)",
        ylabel="H_drop (dB)",
        values=[5.0, 10.0, 15.0, 25.0, 40.0, 60.0],
        label_fn=lambda v: f"T={v:.0f} µm",
        sim_fn=lambda v: simulate(_patch_tech(("sampled", "period"), v)),
    ),
    # 6. Number of π/2 phase shifts
    dict(
        title="Number of π/2 phase shifts",
        xlabel="Wavelength (nm)",
        ylabel="H_drop (dB)",
        values=[0, 1, 2, 4, 6, 8],
        label_fn=lambda v: f"n_ps={int(v)}",
        sim_fn=lambda v: simulate(
            _patch_tech(("phase_shifts", "on"), v > 0,
                        _patch_tech(("phase_shifts", "n"), max(int(v), 1)))
        ) if v > 0 else simulate(_patch_tech(("phase_shifts", "on"), False)),
    ),
    # 7. Device length L
    dict(
        title="Device length L (µm)",
        xlabel="Wavelength (nm)",
        ylabel="H_drop (dB)",
        values=[80, 130, 180, 210, 280, 360],
        label_fn=lambda v: f"L={int(v)} µm",
        sim_fn=lambda v: simulate(TECH_BASE, _patch_p("L", v * 1e-6)),
    ),
    # 8. Propagation loss
    dict(
        title="Propagation loss (dB/cm)",
        xlabel="Wavelength (nm)",
        ylabel="H_drop (dB)",
        values=[1.0, 2.5, 5.0, 8.0, 12.0, 20.0],
        label_fn=lambda v: f"{v:.0f} dB/cm",
        sim_fn=lambda v: simulate(TECH_BASE, _patch_p("loss_dBcm", v)),
    ),
    # 9. Cascade wavelength shift  Δλ
    dict(
        title="Cascade grating offset Δλ (nm)",
        xlabel="Wavelength (nm)",
        ylabel="H_drop (dB)",
        values=[0.5, 1.0, 2.0, 2.5, 4.0, 6.0],
        label_fn=lambda v: f"Δλ={v:.1f} nm",
        sim_fn=lambda v: simulate(_patch_tech(("cascade", "delta_Lambda"), v)),
    ),
]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def main():
    n_sweeps = len(sweeps)
    ncols = 3
    nrows = (n_sweeps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows))
    axes = axes.flatten()

    for ax_idx, sw in enumerate(sweeps):
        ax = axes[ax_idx]
        vals = sw["values"]
        colors = [CMAP(i / (len(vals) - 1)) for i in range(len(vals))]

        for vi, (val, color) in enumerate(zip(vals, colors)):
            print(f"  [{ax_idx+1}/{n_sweeps}] {sw['title']}  val={val}", flush=True)
            hd_db = sw["sim_fn"](val)
            lw = 1.8 if vi == len(vals) // 2 else 1.2
            ax.plot(WL_NM, hd_db, color=color, lw=lw, alpha=0.85,
                    label=sw["label_fn"](val))

        ax.set_title(sw["title"], fontsize=9, fontweight="bold")
        ax.set_xlabel(sw["xlabel"], fontsize=8)
        ax.set_ylabel(sw["ylabel"], fontsize=8)
        ax.set_xlim(float(WL_NM[0]), float(WL_NM[-1]))
        ax.set_ylim(-40, 3)
        ax.axhline(-10, color="0.5", lw=0.5, ls=":", alpha=0.5)
        ax.axhline(-20, color="0.5", lw=0.5, ls=":", alpha=0.5)
        ax.axhline(-30, color="0.5", lw=0.5, ls=":", alpha=0.5)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        leg = ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85,
                        handlelength=1.5, labelspacing=0.25)
        leg.get_frame().set_linewidth(0.5)

    # Hide unused axes
    for ax in axes[n_sweeps:]:
        ax.set_visible(False)

    fig.suptitle(
        "Contra-DC PUF — Drop Port Response vs. Design Parameters\n"
        "(baseline: κ×3, chirp=1.2, anti-apo cd=0.9/eb=2.0, sampled T=15 µm,"
        " n_ps=4, L=210 µm, 5 dB/cm, cascade Δλ=2.5 nm)",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()

    out_path = "outputs/puf_design_exploration.png"
    import pathlib
    pathlib.Path("outputs").mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
