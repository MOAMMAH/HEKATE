"""
1D slab / effective-index mode solver for multimode contra-DC.

Vertical EIM (220 nm SOI) → lateral symmetric slab of width w.
Mode count and n_eff follow from cutoff; n_g from finite-difference dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

C_LIGHT = 299792458.0

# Default 220 nm SOI stack at ~1550 nm
N_SI = 3.476
N_SIO2 = 1.444
H_SI = 220e-9


@dataclass(frozen=True)
class ModeSet:
    """Guided TE modes of one strip waveguide at a fixed wavelength."""

    width: float
    wavelength: float
    n_eff: np.ndarray  # shape (M,)
    n_g: np.ndarray  # shape (M,)
    n_core: float
    n_clad: float

    @property
    def n_modes(self) -> int:
        return int(self.n_eff.size)

    def field(self, m: int, x: np.ndarray, x0: float = 0.0) -> np.ndarray:
        """Normalized lateral E(x) for mode m, guide centered at x0."""
        return _slab_field(
            m,
            x - x0,
            self.width,
            float(self.n_eff[m]),
            self.n_core,
            self.n_clad,
            self.wavelength,
        )


def n_modes_supported(width: float, wavelength: float, n_core: float | None = None) -> int:
    """Number of guided TE modes in a symmetric slab of the given width."""
    if width <= 0.0 or wavelength <= 0.0:
        raise ValueError("width and wavelength must be > 0")
    n_clad = N_SIO2
    if n_core is None:
        n_core = _vertical_neff(wavelength)
    return len(_find_all_neff(width, wavelength, n_core, n_clad))


def slab_modes(
    width: float,
    wavelength: float,
    n_modes_max: int | None = None,
    *,
    n_core: float | None = None,
    n_clad: float = N_SIO2,
    d_lambda: float = 1e-9,
) -> ModeSet:
    """Solve guided TE modes of a lateral slab via the effective-index method."""
    if width <= 0.0 or wavelength <= 0.0:
        raise ValueError("width and wavelength must be > 0")
    if n_core is None:
        n_core = _vertical_neff(wavelength)
    if not (n_clad < n_core):
        raise ValueError("need n_clad < n_core for guided modes")

    n_eff = np.asarray(
        _find_all_neff(width, wavelength, n_core, n_clad), dtype=float
    )
    if n_modes_max is not None:
        n_eff = n_eff[: int(n_modes_max)]
    if n_eff.size < 1:
        raise ValueError(
            f"no guided modes at w={width*1e9:.1f} nm, λ={wavelength*1e9:.1f} nm"
        )

    # Group index by central finite difference of n_eff(λ).
    n_g = np.empty_like(n_eff)
    n_m = np.asarray(
        _find_all_neff(width, wavelength - d_lambda, n_core, n_clad), dtype=float
    )
    n_p = np.asarray(
        _find_all_neff(width, wavelength + d_lambda, n_core, n_clad), dtype=float
    )
    for m in range(n_eff.size):
        if m < n_m.size and m < n_p.size:
            dn_dl = (n_p[m] - n_m[m]) / (2.0 * d_lambda)
            n_g[m] = n_eff[m] - wavelength * dn_dl
        else:
            n_g[m] = n_eff[m]  # near cutoff

    return ModeSet(
        width=float(width),
        wavelength=float(wavelength),
        n_eff=n_eff,
        n_g=n_g,
        n_core=float(n_core),
        n_clad=float(n_clad),
    )


def coupling_matrices(
    modes1: ModeSet,
    modes2: ModeSet,
    kappa11: float,
    kappa12: float,
    kappa22: float,
    gap: float = 200e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build K11, K12, K22 from sidewall field products.

    Normalized so K12[0, 0] == kappa12 (and similarly for the diagonals of
    K11 / K22 when those modes exist). Higher-order overlaps follow the
    product of facing-sidewall field amplitudes.
    """
    m1, m2 = modes1.n_modes, modes2.n_modes
    # Facing walls: WG1 right edge, WG2 left edge.
    x1_wall = 0.5 * modes1.width
    x2_wall = -0.5 * modes2.width
    e1 = np.array(
        [float(modes1.field(i, np.array([x1_wall]))[0]) for i in range(m1)],
        dtype=float,
    )
    e2 = np.array(
        [float(modes2.field(j, np.array([x2_wall]))[0]) for j in range(m2)],
        dtype=float,
    )
    # Decay across the gap softens higher-order (more extended) modes slightly.
    k0 = 2.0 * np.pi / modes1.wavelength
    gamma1 = k0 * np.sqrt(np.maximum(modes1.n_eff**2 - modes1.n_clad**2, 1e-18))
    gamma2 = k0 * np.sqrt(np.maximum(modes2.n_eff**2 - modes2.n_clad**2, 1e-18))
    decay = np.exp(-0.5 * (gamma1[:, None] + gamma2[None, :]) * max(gap, 0.0))

    raw12 = np.abs(e1[:, None] * e2[None, :]) * decay
    raw11 = np.abs(e1[:, None] * e1[None, :])
    raw22 = np.abs(e2[:, None] * e2[None, :])

    def _scale(raw: np.ndarray, target: float) -> np.ndarray:
        if raw.size == 0:
            return raw
        ref = float(raw[0, 0]) if raw.shape[0] and raw.shape[1] else 0.0
        if abs(ref) < 1e-30:
            out = np.zeros_like(raw)
            if out.shape[0] and out.shape[1]:
                out[0, 0] = target
            return out
        return raw * (target / ref)

    k12 = _scale(raw12, float(kappa12))
    k11 = _scale(raw11, float(kappa11)) if m1 else np.zeros((0, 0))
    k22 = _scale(raw22, float(kappa22)) if m2 else np.zeros((0, 0))
    return k11, k12, k22


def synthetic_mode_indices(
    n0: float,
    ng0: float,
    n_modes: int,
    *,
    dn_per_mode: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Placeholder ladder when widths are not supplied but n_modes_max > 1.

    Higher-order modes sit at lower n_eff / n_g. Only used for mode_model='fixed'
    without geometry; prefer slab_modes when widths are available.
    """
    m = max(1, int(n_modes))
    n_eff = n0 - dn_per_mode * np.arange(m, dtype=float)
    n_g = ng0 - 0.5 * dn_per_mode * np.arange(m, dtype=float)
    return n_eff, n_g


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _vertical_neff(wavelength: float) -> float:
    """TE0 of the vertical Si slab (height H_SI) — EIM core index."""
    return _solve_neff(0, H_SI, wavelength, N_SI, N_SIO2)


def _find_all_neff(
    width: float,
    wavelength: float,
    n_core: float,
    n_clad: float,
) -> list[float]:
    """All guided TE n_eff, highest (fundamental) first."""
    k0 = 2.0 * np.pi / wavelength
    n_lo = n_clad + 1e-12
    n_hi = n_core - 1e-12
    if n_hi <= n_lo:
        return []

    def char(ne: float, parity: int) -> float:
        # parity 0 = even family, 1 = odd family (covers all mode orders).
        kx2 = n_core**2 - ne**2
        g2 = ne**2 - n_clad**2
        if kx2 <= 0.0 or g2 <= 0.0:
            return 1.0
        kx = k0 * np.sqrt(kx2)
        gamma = k0 * np.sqrt(g2)
        half = 0.5 * kx * width
        if parity == 0:
            return kx * np.sin(half) - gamma * np.cos(half)
        return kx * np.cos(half) + gamma * np.sin(half)

    roots: list[float] = []
    for parity in (0, 1):
        def f(ne: float, _p: int = parity) -> float:
            return char(ne, _p)

        n_scan = np.linspace(n_hi, n_lo, 2000)
        f_scan = np.array([f(n) for n in n_scan])
        for i in range(len(n_scan) - 1):
            a, b = f_scan[i], f_scan[i + 1]
            if a == 0.0 or a * b < 0.0:
                try:
                    lo_b, hi_b = (n_scan[i + 1], n_scan[i])
                    if lo_b > hi_b:
                        lo_b, hi_b = hi_b, lo_b
                    root = float(brentq(f, lo_b, hi_b))
                except ValueError:
                    continue
                if all(abs(root - r) > 1e-7 for r in roots):
                    roots.append(root)
    roots.sort(reverse=True)
    return roots


def _solve_neff(
    m: int,
    width: float,
    wavelength: float,
    n_core: float,
    n_clad: float,
) -> float:
    """Root of the even/odd TE slab characteristic equation for mode order m."""
    roots = _find_all_neff(width, wavelength, n_core, n_clad)
    if len(roots) <= m:
        raise ValueError(f"mode m={m} not guided at this width/wavelength")
    return float(roots[m])


def _slab_field(
    m: int,
    x: np.ndarray,
    width: float,
    n_eff: float,
    n_core: float,
    n_clad: float,
    wavelength: float,
) -> np.ndarray:
    """Unit-peak lateral TE field for mode m (guide centered at 0)."""
    k0 = 2.0 * np.pi / wavelength
    kx = k0 * np.sqrt(max(n_core**2 - n_eff**2, 0.0))
    gamma = k0 * np.sqrt(max(n_eff**2 - n_clad**2, 1e-30))
    half = 0.5 * width
    e = np.empty_like(x, dtype=float)
    core = np.abs(x) <= half
    if m % 2 == 0:
        e[core] = np.cos(kx * x[core])
        e[~core] = np.cos(kx * half) * np.exp(-gamma * (np.abs(x[~core]) - half))
    else:
        e[core] = np.sin(kx * x[core])
        sign = np.sign(x[~core])
        e[~core] = (
            sign * np.sin(kx * half) * np.exp(-gamma * (np.abs(x[~core]) - half))
        )
    peak = np.max(np.abs(e))
    if peak > 0.0:
        e /= peak
    return e
