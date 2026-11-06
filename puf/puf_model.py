"""
Contra-DC PUF model (coupled-mode transfer matrix).

Single-mode (default): 4x4 CMT, one mode per waveguide — bit-compatible with
the original calibrated device.

Multimode (opt-in): d = 2*(M1+M2) CMT with width-driven mode count from the
EIM slab solver. Drop-port modes are available as a vector; power-summed
spectra feed a single PD per wavelength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from puf.modes import (
    coupling_matrices,
    slab_modes,
    synthetic_mode_indices,
)

C_LIGHT = 299792458.0
DN_DT = 1.8e-4  # silicon thermo-optic coefficient /K


@dataclass(frozen=True)
class SimGrid:
    """Shared time / frequency grid for the physics chain."""

    fs: float = 4e12
    n: int = 16384

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n) / self.fs

    @property
    def f(self) -> np.ndarray:
        return np.fft.fftfreq(self.n, self.dt)


def _correlated_disorder(rng, n, d_l, corr_length, sigma):
    """White Gaussian -> Gaussian-smooth -> rescale to target std.

    Reflect-pads before convolution so edge variance is not tapered and so a
    kernel longer than ``n`` cannot change the output length (numpy's
    ``mode='same'`` returns ``max(len(signal), len(kernel))``).
    """
    white = rng.normal(0.0, 1.0, size=n)
    sigma_samp = max(corr_length / d_l, 1e-12)
    radius = max(1, int(np.ceil(3.0 * sigma_samp)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_samp) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(white, radius, mode="reflect")
    smooth = np.convolve(padded, kernel, mode="valid")
    s = float(smooth.std())
    if s < 1e-30:
        return np.zeros(n)
    return smooth * (sigma / s)


class ContraDCPUF:
    """One contra-DC PUF instance. Fabrication randomness seeded by device_id.

    Spectra are cached per frequency grid. ``temperature_offset_K`` is part of
    the cache key, so it may be reassigned after construction; changing any
    other geometry or coupling attribute requires ``clear_caches()``.

    Mode selection
    --------------
    mode_model:
      - ``"single"`` (default): one mode per guide, legacy 4x4 fast path.
      - ``"auto"``: mode count from waveguide widths via the slab solver.
      - ``"fixed"``: use ``n_modes_max`` (requires widths, or a synthetic ladder).
    """

    # Memory budget for one batch of section matrices (complex128 d×d each).
    _BATCH_BYTES = 64 << 20
    # Cached spectra per instance; keys are whole frequency grids, so cap them.
    _CACHE_MAX = 32

    def __init__(
        self,
        device_id: int = 0,
        # Geometry / discretization
        L: float = 210e-6,
        n_sect: int = 80,
        dz: float | None = None,
        # Period set so contra-DC drop peak lands on the WDM plan (~1550 nm):
        # Lambda0 = lambda_target / (n1_0 + n2_0) = 1550 nm / 5.070 ≈ 305.72 nm
        Lambda0: float = 305.72e-9,
        # Refractive indices & dispersion (single-mode defaults / overrides)
        n1_0: float = 2.580,
        n2_0: float = 2.490,
        ng1: float = 3.65,
        ng2: float = 3.55,
        wl0: float = 1.550e-6,
        # Coupling & loss
        kappa11: float = 5000.0,
        kappa12: float = 16300.0,
        kappa22: float = 6200.0,
        loss_dBcm: float = 5.0,
        # Anti-apodization (deterministic backbone)
        center_dip: float = 0.90,
        edge_boost: float = 2.0,
        kappa_mult: float = 3.0,
        # Disorder
        sigma_dn: float = 2e-3,
        sigma_kappa_rel: float = 0.05,
        sigma_dLambda: float = 0.5e-9,
        corr_length: float = 2e-6,
        # Temperature
        temperature_offset_K: float = 0.0,
        # Multimode (opt-in; default preserves single-mode behavior)
        mode_model: str = "single",
        w1: float | None = None,
        w2: float | None = None,
        gap: float = 200e-9,
        n_modes_max: int | None = None,
        sigma_w: float = 0.0,
        force_general_cmt: bool = False,
    ):
        self.device_id = int(device_id)
        self.L = float(L)
        if not self.L > 0.0:
            raise ValueError("L must be > 0")
        if dz is not None:
            if not float(dz) > 0.0:
                raise ValueError("dz must be > 0")
            self.n_sect = max(1, int(round(self.L / float(dz))))
        else:
            if int(n_sect) < 1:
                raise ValueError("n_sect must be >= 1")
            self.n_sect = int(n_sect)
        self.dz = self.L / self.n_sect
        self.Lambda0 = float(Lambda0)
        if not self.Lambda0 > 0.0:
            raise ValueError("Lambda0 must be > 0")

        self.n1_0 = float(n1_0)
        self.n2_0 = float(n2_0)
        self.ng1 = float(ng1)
        self.ng2 = float(ng2)
        self.wl0 = float(wl0)
        if not self.wl0 > 0.0:
            raise ValueError("wl0 must be > 0")

        self.kappa11 = float(kappa11)
        self.kappa12 = float(kappa12)
        self.kappa22 = float(kappa22)
        self.loss_dBcm = float(loss_dBcm)
        if self.loss_dBcm < 0.0:
            raise ValueError("loss_dBcm must be >= 0 (passive device)")

        self.center_dip = float(center_dip)
        self.edge_boost = float(edge_boost)
        self.kappa_mult = float(kappa_mult)

        self.sigma_dn = float(sigma_dn)
        self.sigma_kappa_rel = float(sigma_kappa_rel)
        self.sigma_dLambda = float(sigma_dLambda)
        self.corr_length = float(corr_length)
        if not self.corr_length > 0.0:
            raise ValueError("corr_length must be > 0")
        self.temperature_offset_K = float(temperature_offset_K)

        self.mode_model = str(mode_model).lower()
        if self.mode_model not in ("single", "auto", "fixed"):
            raise ValueError('mode_model must be "single", "auto", or "fixed"')
        self.w1 = None if w1 is None else float(w1)
        self.w2 = None if w2 is None else float(w2)
        self.gap = float(gap)
        self.n_modes_max = None if n_modes_max is None else int(n_modes_max)
        self.sigma_w = float(sigma_w)
        self.force_general_cmt = bool(force_general_cmt)

        self._z = np.linspace(0.0, self.L, self.n_sect, endpoint=False) + self.dz / 2.0
        z_norm = (self._z - self.L / 2.0) / (self.L / 2.0)
        apo = self.anti_apodization(z_norm)

        rng = np.random.default_rng(self.device_id)
        self.dn1 = _correlated_disorder(
            rng, self.n_sect, self.dz, self.corr_length, self.sigma_dn
        )
        self.dn2 = _correlated_disorder(
            rng, self.n_sect, self.dz, self.corr_length, self.sigma_dn
        )
        self.dkappa = _correlated_disorder(
            rng, self.n_sect, self.dz, self.corr_length, self.sigma_kappa_rel
        )
        self.dLambda = _correlated_disorder(
            rng, self.n_sect, self.dz, self.corr_length, self.sigma_dLambda
        )
        self.dw1 = _correlated_disorder(
            rng, self.n_sect, self.dz, self.corr_length, self.sigma_w
        )
        self.dw2 = _correlated_disorder(
            rng, self.n_sect, self.dz, self.corr_length, self.sigma_w
        )

        self.k12_arr = self.kappa12 * self.kappa_mult * apo * (1.0 + self.dkappa)
        self.lam_arr = self.Lambda0 + self.dLambda
        if np.any(self.lam_arr <= 0.0):
            raise ValueError(
                "sigma_dLambda drove a section period to <= 0; reduce it or raise Lambda0"
            )

        # Mode geometry / coupling matrices
        self._init_modes()

        self._r_cache: dict = {}
        self._h_cache: dict = {}
        self._rm_cache: dict = {}

    def _init_modes(self) -> None:
        """Resolve M1/M2, per-mode indices, and K matrices from mode_model."""
        if self.mode_model == "single" or (
            self.mode_model == "fixed" and self.n_modes_max == 1
        ):
            self.n_eff1 = np.array([self.n1_0], dtype=float)
            self.n_eff2 = np.array([self.n2_0], dtype=float)
            self.n_g1 = np.array([self.ng1], dtype=float)
            self.n_g2 = np.array([self.ng2], dtype=float)
            self.modes1 = None
            self.modes2 = None
            self.K11 = np.array([[self.kappa11]], dtype=float)
            self.K12 = np.array([[self.kappa12]], dtype=float)
            self.K22 = np.array([[self.kappa22]], dtype=float)
        elif self.mode_model == "auto":
            if self.w1 is None or self.w2 is None:
                raise ValueError('mode_model="auto" requires w1 and w2')
            self.modes1 = slab_modes(self.w1, self.wl0, self.n_modes_max)
            self.modes2 = slab_modes(self.w2, self.wl0, self.n_modes_max)
            self.n_eff1 = self.modes1.n_eff.copy()
            self.n_eff2 = self.modes2.n_eff.copy()
            self.n_g1 = self.modes1.n_g.copy()
            self.n_g2 = self.modes2.n_g.copy()
            # Keep scalar aliases as fundamental for Bragg / tau helpers.
            self.n1_0 = float(self.n_eff1[0])
            self.n2_0 = float(self.n_eff2[0])
            self.ng1 = float(self.n_g1[0])
            self.ng2 = float(self.n_g2[0])
            self.K11, self.K12, self.K22 = coupling_matrices(
                self.modes1,
                self.modes2,
                self.kappa11,
                self.kappa12,
                self.kappa22,
                gap=self.gap,
            )
        else:  # fixed
            if self.n_modes_max is None or self.n_modes_max < 1:
                raise ValueError('mode_model="fixed" requires n_modes_max >= 1')
            if self.w1 is not None and self.w2 is not None:
                self.modes1 = slab_modes(self.w1, self.wl0, self.n_modes_max)
                self.modes2 = slab_modes(self.w2, self.wl0, self.n_modes_max)
                self.n_eff1 = self.modes1.n_eff.copy()
                self.n_eff2 = self.modes2.n_eff.copy()
                self.n_g1 = self.modes1.n_g.copy()
                self.n_g2 = self.modes2.n_g.copy()
                self.n1_0 = float(self.n_eff1[0])
                self.n2_0 = float(self.n_eff2[0])
                self.ng1 = float(self.n_g1[0])
                self.ng2 = float(self.n_g2[0])
                self.K11, self.K12, self.K22 = coupling_matrices(
                    self.modes1,
                    self.modes2,
                    self.kappa11,
                    self.kappa12,
                    self.kappa22,
                    gap=self.gap,
                )
            else:
                self.modes1 = None
                self.modes2 = None
                self.n_eff1, self.n_g1 = synthetic_mode_indices(
                    self.n1_0, self.ng1, self.n_modes_max
                )
                self.n_eff2, self.n_g2 = synthetic_mode_indices(
                    self.n2_0, self.ng2, self.n_modes_max
                )
                m1, m2 = self.n_eff1.size, self.n_eff2.size
                # Diagonal-dominant placeholder; cross terms decay with |i-j|.
                ii = np.arange(m1)[:, None]
                jj = np.arange(m2)[None, :]
                decay = np.exp(-0.5 * np.abs(ii - jj))
                self.K12 = self.kappa12 * decay
                self.K11 = self.kappa11 * np.exp(
                    -0.5 * np.abs(np.arange(m1)[:, None] - np.arange(m1)[None, :])
                )
                self.K22 = self.kappa22 * np.exp(
                    -0.5 * np.abs(np.arange(m2)[:, None] - np.arange(m2)[None, :])
                )

        self.M1 = int(self.n_eff1.size)
        self.M2 = int(self.n_eff2.size)
        self.n_fwd = self.M1 + self.M2
        self.d = 2 * self.n_fwd
        # Legacy 4x4 when truly single-mode and not forced otherwise.
        self.use_legacy_4x4 = (
            self.M1 == 1 and self.M2 == 1 and not self.force_general_cmt
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def tau_max(self) -> float:
        """Delay spread: 2 * n_g_avg * L / c (seconds), using fundamental n_g."""
        return 2.0 * 0.5 * (self.ng1 + self.ng2) * self.L / C_LIGHT

    @property
    def tau_max_modes(self) -> np.ndarray:
        """Per drop-mode delay spread using WG2 group indices."""
        return 2.0 * self.n_g2 * self.L / C_LIGHT

    def anti_apodization(self, z_norm):
        """Anti-apodization profile along the grating (z_norm in [-1, 1])."""
        apo = (1.0 - self.center_dip * np.cos(np.pi * z_norm / 2.0) ** 2) * (
            1.0 + self.edge_boost * np.sin(np.pi * z_norm / 2.0) ** 2
        )
        return np.clip(apo, 0.0, 10.0)

    def clear_caches(self) -> None:
        """Drop cached spectra. Required after mutating geometry or coupling."""
        self._r_cache.clear()
        self._h_cache.clear()
        self._rm_cache.clear()

    def _cache_store(self, cache: dict, key, value: np.ndarray) -> None:
        """Insert with a FIFO cap; keys are whole frequency grids."""
        if len(cache) >= self._CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = value

    def bragg_wavelengths(self) -> np.ndarray:
        """Phase-match wavelengths λ_ij = Λ · (n1_i + n2_j) for all mode pairs."""
        return self.Lambda0 * (
            self.n_eff1[:, None] + self.n_eff2[None, :]
        )

    def reflection_spectrum(self, f_abs: np.ndarray) -> np.ndarray:
        """Complex drop-port reflection of the fundamental WG2 mode (Hz)."""
        f_abs = np.asarray(f_abs, dtype=float).ravel()
        key = (f_abs.tobytes(), self.temperature_offset_K, "r0")
        if key in self._r_cache:
            return self._r_cache[key].copy()

        wl = C_LIGHT / f_abs
        r = self._transfer_spectrum(wl)
        self._cache_store(self._r_cache, key, r)
        return r.copy()

    def drop_amplitudes(self, f_abs: np.ndarray) -> np.ndarray:
        """Drop-port amplitudes for all WG2 modes; shape (n_freq, M2)."""
        f_abs = np.asarray(f_abs, dtype=float).ravel()
        key = (f_abs.tobytes(), self.temperature_offset_K, "rm")
        if key in self._rm_cache:
            return self._rm_cache[key].copy()

        wl = C_LIGHT / f_abs
        ports = self._transfer_spectrum(wl, return_ports=True)
        rm = ports["B2_modes"]
        self._cache_store(self._rm_cache, key, rm)
        return rm.copy()

    def reflection_power_spectrum(self, f_abs: np.ndarray) -> np.ndarray:
        """Mode-summed drop power Σ_j |r_j|² (single-PD observable)."""
        rm = self.drop_amplitudes(f_abs)
        return np.sum(np.abs(rm) ** 2, axis=1)

    def transmission_spectrum(self, f_abs: np.ndarray) -> np.ndarray:
        """Complex through-port transmission t at absolute optical frequencies (Hz)."""
        f_abs = np.asarray(f_abs, dtype=float).ravel()
        ports = self.port_amplitudes(f_abs)
        return ports["F1_L"]

    def port_amplitudes(self, f_abs: np.ndarray) -> dict[str, np.ndarray]:
        """Port amplitudes for unit launch into F1 mode 0.

        Keys B1_0, B2_0, F1_L, F2_L are the *fundamental* components (compat).
        Also includes B2_modes with shape (n_freq, M2) and total_power.
        """
        f_abs = np.asarray(f_abs, dtype=float).ravel()
        wl = C_LIGHT / f_abs
        return self._transfer_spectrum(wl, return_ports=True)

    def impulse_response(self, grid: SimGrid, f0: float) -> np.ndarray:
        """Baseband-equivalent causal h(t) from fundamental r(f0 + grid.f)."""
        key = (float(grid.fs), int(grid.n), float(f0), self.temperature_offset_K, "h0")
        if key in self._h_cache:
            return self._h_cache[key].copy()

        f_abs = f0 + grid.f
        r = self.reflection_spectrum(f_abs)
        h = np.fft.fft(r) * (grid.fs / grid.n)
        self._cache_store(self._h_cache, key, h)
        return h.copy()

    def impulse_responses(self, grid: SimGrid, f0: float) -> np.ndarray:
        """Per-mode baseband h_j(t); shape (M2, grid.n)."""
        key = (float(grid.fs), int(grid.n), float(f0), self.temperature_offset_K, "hm")
        if key in self._h_cache:
            return self._h_cache[key].copy()

        f_abs = f0 + grid.f
        rm = self.drop_amplitudes(f_abs)
        # fft along frequency axis for each mode
        hm = np.fft.fft(rm, axis=0).T * (grid.fs / grid.n)
        self._cache_store(self._h_cache, key, hm)
        return hm.copy()

    def drop_port_response(self, wl_nm=None):
        """Plotting helper: wavelength (nm) and mode-summed |r|^2 in dB."""
        if wl_nm is None:
            wl_nm = np.linspace(1520.0, 1580.0, 300)
        wl_nm = np.asarray(wl_nm, dtype=float)
        f_abs = C_LIGHT / (wl_nm * 1e-9)
        p = self.reflection_power_spectrum(f_abs)
        hd_db = 10.0 * np.log10(np.clip(p, 1e-12, None))
        return wl_nm, hd_db

    def peak_reflectivity(self, wl_nm=None) -> float:
        """Global max sqrt(Σ|r_j|²) over the device response band."""
        if wl_nm is None:
            wl_nm = np.linspace(1520.0, 1580.0, 601)
        wl_nm = np.asarray(wl_nm, dtype=float)
        f_abs = C_LIGHT / (wl_nm * 1e-9)
        return float(np.sqrt(np.max(self.reflection_power_spectrum(f_abs))))

    # ------------------------------------------------------------------
    # Transfer-matrix engine
    # ------------------------------------------------------------------

    def _alpha(self) -> float:
        return self.loss_dBcm * 100.0 / (20.0 * np.log10(np.e))

    def _section_matrices(self, wl: np.ndarray, dn_th: float) -> np.ndarray:
        """Exact transfer matrices for every (wavelength, section) pair.

        Returns shape ``(n_wl, n_sect, 4, 4)`` on the legacy single-mode path.
        """
        w = wl[:, None]  # (n_wl, 1) against (1, n_sect) section arrays
        # Width disorder as an effective index shift (dn/dw ~ 1/μm scale).
        dn_w1 = 0.0
        dn_w2 = 0.0
        if self.sigma_w > 0.0 and self.w1 is not None:
            dn_w1 = (1.0 / max(self.w1, 1e-9)) * self.dw1[None, :]
        if self.sigma_w > 0.0 and self.w2 is not None:
            dn_w2 = (1.0 / max(self.w2, 1e-9)) * self.dw2[None, :]

        n1 = (
            self.n1_0
            - (self.ng1 - self.n1_0) / self.wl0 * (w - self.wl0)
            + dn_th
            + self.dn1[None, :]
            + dn_w1
        )
        n2 = (
            self.n2_0
            - (self.ng2 - self.n2_0) / self.wl0 * (w - self.wl0)
            + dn_th
            + self.dn2[None, :]
            + dn_w2
        )
        pL = np.pi / self.lam_arr[None, :]
        d1 = 2.0 * np.pi * n1 / w - pL
        d2 = 2.0 * np.pi * n2 / w - pL
        k12 = np.broadcast_to(self.k12_arr[None, :], d1.shape)
        alpha = self._alpha()

        S = np.zeros(d1.shape + (4, 4), dtype=complex)
        S[..., 0, 0] = 1j * d1 - alpha
        S[..., 0, 2] = 1j * self.kappa11
        S[..., 0, 3] = 1j * k12
        S[..., 1, 1] = 1j * d2 - alpha
        S[..., 1, 2] = 1j * k12
        S[..., 1, 3] = 1j * self.kappa22
        S[..., 2, 0] = -1j * self.kappa11
        S[..., 2, 1] = -1j * k12
        S[..., 2, 2] = -1j * d1 + alpha
        S[..., 3, 0] = -1j * k12
        S[..., 3, 1] = -1j * self.kappa22
        S[..., 3, 3] = -1j * d2 + alpha
        return expm(S * self.dz)

    def _section_matrices_mm(self, wl: np.ndarray, dn_th: float) -> np.ndarray:
        """Multimode section matrices, shape (n_wl, n_sect, d, d) — vectorized."""
        m1, m2 = self.M1, self.M2
        n_fwd = self.n_fwd
        d = self.d
        alpha = self._alpha()
        apo_k = (
            self.kappa_mult
            * self.anti_apodization((self._z - self.L / 2.0) / (self.L / 2.0))
            * (1.0 + self.dkappa)
        )

        # Shapes: (n_wl, n_sect, M)
        n_wl = int(wl.size)
        w = wl[:, None, None]
        n1 = np.broadcast_to(
            self.n_eff1[None, None, :]
            - (self.n_g1 - self.n_eff1)[None, None, :]
            / self.wl0
            * (w - self.wl0)
            + dn_th,
            (n_wl, self.n_sect, m1),
        ).copy()
        n2 = np.broadcast_to(
            self.n_eff2[None, None, :]
            - (self.n_g2 - self.n_eff2)[None, None, :]
            / self.wl0
            * (w - self.wl0)
            + dn_th,
            (n_wl, self.n_sect, m2),
        ).copy()
        n1[:, :, 0] += self.dn1[None, :]
        n2[:, :, 0] += self.dn2[None, :]
        if self.sigma_w > 0.0 and self.w1 is not None:
            n1 += (
                (np.arange(m1) + 1.0)[None, None, :]
                * self.dw1[None, :, None]
                / max(self.w1, 1e-9)
            )
        if self.sigma_w > 0.0 and self.w2 is not None:
            n2 += (
                (np.arange(m2) + 1.0)[None, None, :]
                * self.dw2[None, :, None]
                / max(self.w2, 1e-9)
            )

        pL = (np.pi / self.lam_arr)[None, :, None]
        d1 = 2.0 * np.pi * n1 / w - pL  # (n_wl, n_sect, M1)
        d2 = 2.0 * np.pi * n2 / w - pL

        S = np.zeros(d1.shape[:2] + (d, d), dtype=complex)
        # Forward / backward diagonals
        i1 = np.arange(m1)
        j2 = np.arange(m2)
        S[..., i1, i1] = 1j * d1 - alpha
        S[..., m1 + j2, m1 + j2] = 1j * d2 - alpha
        S[..., n_fwd + i1, n_fwd + i1] = -1j * d1 + alpha
        S[..., n_fwd + m1 + j2, n_fwd + m1 + j2] = -1j * d2 + alpha

        # Couplings: broadcast over wavelength; κ12 scaled per section.
        k11 = self.K11  # (m1, m1)
        k22 = self.K22
        k12 = self.K12[None, :, :] * apo_k[:, None, None]  # (n_sect, m1, m2)

        # F1 <-> B1
        S[..., 0:m1, n_fwd : n_fwd + m1] = 1j * k11
        S[..., n_fwd : n_fwd + m1, 0:m1] = -1j * k11.T
        # F2 <-> B2
        S[..., m1:n_fwd, n_fwd + m1 : d] = 1j * k22
        S[..., n_fwd + m1 : d, m1:n_fwd] = -1j * k22.T
        # F1 <-> B2  and  F2 <-> B1  (κ12)
        S[..., 0:m1, n_fwd + m1 : d] = 1j * k12
        S[..., n_fwd + m1 : d, 0:m1] = -1j * np.swapaxes(k12, -2, -1)
        S[..., m1:n_fwd, n_fwd : n_fwd + m1] = 1j * np.swapaxes(k12, -2, -1)
        S[..., n_fwd : n_fwd + m1, m1:n_fwd] = -1j * k12

        return expm(S * self.dz)

    def _bvp_solve(self, M: np.ndarray):
        """
        Boundary-value solve for unit launch into WG1, batched over frequency.

        M has shape (n_wl, 4, 4) and maps fields at z=0 to z=L.
        F(0)=[1,0], B(L)=0. Returns (B1(0), B2(0), F1(L), F2(L)) arrays.
        """
        Mff = M[:, 0:2, 0:2]
        Mfb = M[:, 0:2, 2:4]
        Mbf = M[:, 2:4, 0:2]
        Mbb = M[:, 2:4, 2:4]
        # F(0) = [1, 0], so M?f @ F(0) is just the first column of M?f.
        try:
            B0 = np.linalg.solve(Mbb, -Mbf[:, :, 0, None])[:, :, 0]
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"ContraDCPUF BVP singular (device_id={self.device_id})"
            ) from exc
        FL = Mff[:, :, 0] + np.einsum("nij,nj->ni", Mfb, B0)
        return B0[:, 0], B0[:, 1], FL[:, 0], FL[:, 1]

    def _bvp_solve_mm(self, M: np.ndarray):
        """General BVP: F(0)=e_in (WG1 mode 0), B(L)=0 for all backward modes."""
        n_fwd = self.n_fwd
        Mff = M[:, 0:n_fwd, 0:n_fwd]
        Mfb = M[:, 0:n_fwd, n_fwd:]
        Mbf = M[:, n_fwd:, 0:n_fwd]
        Mbb = M[:, n_fwd:, n_fwd:]
        rhs = -Mbf[:, :, 0]
        try:
            B0 = np.linalg.solve(Mbb, rhs[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            # Deep stopband / long device: TMM cascade ill-conditioned.
            B0 = np.einsum("nij,nj->ni", np.linalg.pinv(Mbb), rhs)
        FL = Mff[:, :, 0] + np.einsum("nij,nj->ni", Mfb, B0)
        return B0, FL

    def _transfer_spectrum(self, wl: np.ndarray, return_ports: bool = False):
        """Evaluate drop-port r, or port amplitudes, on wavelengths (m)."""
        wl = np.asarray(wl, dtype=float).ravel()
        if np.any(~np.isfinite(wl)) or np.any(wl <= 0.0):
            raise ValueError("wavelengths must be finite and > 0")

        if self.use_legacy_4x4:
            return self._transfer_spectrum_legacy(wl, return_ports=return_ports)
        return self._transfer_spectrum_mm(wl, return_ports=return_ports)

    def _transfer_spectrum_legacy(self, wl: np.ndarray, return_ports: bool = False):
        n_freq = wl.size
        dn_th = DN_DT * self.temperature_offset_K

        B1 = np.empty(n_freq, dtype=complex)
        B2 = np.empty(n_freq, dtype=complex)
        F1 = np.empty(n_freq, dtype=complex)
        F2 = np.empty(n_freq, dtype=complex)

        bytes_per_wl = self.n_sect * 16 * np.dtype(complex).itemsize
        chunk = max(1, self._BATCH_BYTES // max(bytes_per_wl, 1))

        for lo in range(0, n_freq, chunk):
            hi = min(lo + chunk, n_freq)
            C = self._section_matrices(wl[lo:hi], dn_th)
            M = np.broadcast_to(np.eye(4, dtype=complex), (hi - lo, 4, 4)).copy()
            for i in range(self.n_sect):
                M = C[:, i] @ M
            B1[lo:hi], B2[lo:hi], F1[lo:hi], F2[lo:hi] = self._bvp_solve(M)

        if not (
            np.isfinite(B1).all()
            and np.isfinite(B2).all()
            and np.isfinite(F1).all()
            and np.isfinite(F2).all()
        ):
            raise FloatingPointError(
                f"ContraDCPUF spectrum is not finite (device_id={self.device_id}); "
                "cascade overflowed — reduce kappa/L or switch to an S-matrix cascade"
            )

        if return_ports:
            return {
                "B1_0": B1,
                "B2_0": B2,
                "F1_L": F1,
                "F2_L": F2,
                "B2_modes": B2[:, None],
                "total_power": (
                    np.abs(B1) ** 2
                    + np.abs(B2) ** 2
                    + np.abs(F1) ** 2
                    + np.abs(F2) ** 2
                ),
            }
        return B2

    def _transfer_spectrum_mm(self, wl: np.ndarray, return_ports: bool = False):
        n_freq = wl.size
        dn_th = DN_DT * self.temperature_offset_K
        d = self.d
        m1, m2 = self.M1, self.M2

        B_all = np.empty((n_freq, self.n_fwd), dtype=complex)
        F_all = np.empty((n_freq, self.n_fwd), dtype=complex)

        bytes_per_wl = self.n_sect * d * d * np.dtype(complex).itemsize
        chunk = max(1, self._BATCH_BYTES // max(bytes_per_wl, 1))

        for lo in range(0, n_freq, chunk):
            hi = min(lo + chunk, n_freq)
            C = self._section_matrices_mm(wl[lo:hi], dn_th)
            M = np.broadcast_to(np.eye(d, dtype=complex), (hi - lo, d, d)).copy()
            for i in range(self.n_sect):
                M = C[:, i] @ M
            B0, FL = self._bvp_solve_mm(M)
            B_all[lo:hi] = B0
            F_all[lo:hi] = FL

        if not (np.isfinite(B_all).all() and np.isfinite(F_all).all()):
            raise FloatingPointError(
                f"ContraDCPUF multimode spectrum is not finite "
                f"(device_id={self.device_id}); cascade overflowed"
            )

        B1 = B_all[:, 0]
        B2 = B_all[:, m1] if m2 >= 1 else np.zeros(n_freq, dtype=complex)
        F1 = F_all[:, 0]
        F2 = F_all[:, m1] if m2 >= 1 else np.zeros(n_freq, dtype=complex)
        B2_modes = B_all[:, m1 : m1 + m2]

        if return_ports:
            total = np.sum(np.abs(B_all) ** 2, axis=1) + np.sum(
                np.abs(F_all) ** 2, axis=1
            )
            return {
                "B1_0": B1,
                "B2_0": B2,
                "F1_L": F1,
                "F2_L": F2,
                "B2_modes": B2_modes,
                "total_power": total,
            }
        return B2


# Back-compat alias
ContraDCPuf = ContraDCPUF


if __name__ == "__main__":
    puf = ContraDCPUF(device_id=1)
    print(
        f"device_id={puf.device_id}, n_sect={puf.n_sect}, "
        f"M1={puf.M1}, M2={puf.M2}, legacy={puf.use_legacy_4x4}, "
        f"tau_max={puf.tau_max*1e12:.2f} ps"
    )

    wl_nm, hd_db = puf.drop_port_response()
    print(f"λ: {wl_nm[0]:.1f}–{wl_nm[-1]:.1f} nm")
    print(f"|r|^2: min={hd_db.min():.1f} dB, max={hd_db.max():.1f} dB")

    f0 = C_LIGHT / 1550e-9
    f_abs = C_LIGHT / (wl_nm * 1e-9)
    ports = puf.port_amplitudes(f_abs)
    passive = float(np.max(ports["total_power"]))
    print(f"max port power = {passive:.6f}")
    if passive > 1.0 + 1e-9:
        print("  WARNING: passivity violated — check transfer-matrix sign conventions")

    puf2 = ContraDCPUF(device_id=2)
    r = ports["B2_0"]
    r2 = puf2.reflection_spectrum(f_abs)
    corr = np.corrcoef(np.abs(r), np.abs(r2))[0, 1]
    print(f"|r| corr (id1 vs id2) = {corr:.3f}")

    grid = SimGrid(fs=4e12, n=2048)
    h = puf.impulse_response(grid, f0)
    print(f"h(t): {len(h)} samples, max|h|={np.max(np.abs(h)):.3e}")

    # Multimode smoke: wider guides
    puf_mm = ContraDCPUF(
        device_id=1,
        mode_model="auto",
        w1=1.2e-6,
        w2=1.0e-6,
        n_modes_max=3,
        n_sect=20,
        L=210e-6,
    )
    print(
        f"multimode: M1={puf_mm.M1}, M2={puf_mm.M2}, "
        f"Bragg(nm)=\n{puf_mm.bragg_wavelengths()*1e9}"
    )
    rm = puf_mm.drop_amplitudes(f_abs[::10])
    print(f"drop_amplitudes shape={rm.shape}")
