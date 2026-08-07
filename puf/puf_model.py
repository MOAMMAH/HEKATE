"""
Contra-DC PUF model (4x4 coupled-mode transfer matrix).

Primary API: reflection_spectrum(f_abs) -> complex r(omega).
Impulse response via IFFT on a uniform-frequency SimGrid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

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
    """

    # Memory budget for one batch of section matrices (complex128 4x4 each).
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
        # Refractive indices & dispersion
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

        self.k12_arr = self.kappa12 * self.kappa_mult * apo * (1.0 + self.dkappa)
        self.lam_arr = self.Lambda0 + self.dLambda
        if np.any(self.lam_arr <= 0.0):
            raise ValueError(
                "sigma_dLambda drove a section period to <= 0; reduce it or raise Lambda0"
            )

        self._r_cache: dict = {}
        self._h_cache: dict = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def tau_max(self) -> float:
        """Delay spread: 2 * n_g_avg * L / c (seconds)."""
        return 2.0 * 0.5 * (self.ng1 + self.ng2) * self.L / C_LIGHT

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

    def _cache_store(self, cache: dict, key, value: np.ndarray) -> None:
        """Insert with a FIFO cap; keys are whole frequency grids."""
        if len(cache) >= self._CACHE_MAX:
            cache.pop(next(iter(cache)))
        cache[key] = value

    def reflection_spectrum(self, f_abs: np.ndarray) -> np.ndarray:
        """Complex drop-port reflection r at absolute optical frequencies (Hz)."""
        f_abs = np.asarray(f_abs, dtype=float).ravel()
        key = (f_abs.tobytes(), self.temperature_offset_K)
        if key in self._r_cache:
            return self._r_cache[key].copy()

        wl = C_LIGHT / f_abs
        r = self._transfer_spectrum(wl)
        self._cache_store(self._r_cache, key, r)
        return r.copy()

    def transmission_spectrum(self, f_abs: np.ndarray) -> np.ndarray:
        """Complex through-port transmission t at absolute optical frequencies (Hz)."""
        f_abs = np.asarray(f_abs, dtype=float).ravel()
        ports = self.port_amplitudes(f_abs)
        return ports["F1_L"]

    def port_amplitudes(self, f_abs: np.ndarray) -> dict[str, np.ndarray]:
        """Four output amplitudes for unit launch into F1(0).

        Keys: B1_0, B2_0 (drop), F1_L (through), F2_L.
        """
        f_abs = np.asarray(f_abs, dtype=float).ravel()
        wl = C_LIGHT / f_abs
        return self._transfer_spectrum(wl, return_ports=True)

    def impulse_response(self, grid: SimGrid, f0: float) -> np.ndarray:
        """Baseband-equivalent causal h(t) from r(f0 + grid.f).

        Uses the e^{+jωt} Fourier sign so positive group delay maps to t > 0.
        """
        # Key on grid *values*: id() is reused after a temporary SimGrid is
        # collected, which returned another grid's h (wrong length included).
        key = (float(grid.fs), int(grid.n), float(f0), self.temperature_offset_K)
        if key in self._h_cache:
            return self._h_cache[key].copy()

        f_abs = f0 + grid.f
        r = self.reflection_spectrum(f_abs)
        h = np.fft.fft(r) * (grid.fs / grid.n)
        self._cache_store(self._h_cache, key, h)
        return h.copy()

    def drop_port_response(self, wl_nm=None):
        """Plotting helper: wavelength (nm) and |r|^2 in dB (not normalized)."""
        if wl_nm is None:
            wl_nm = np.linspace(1520.0, 1580.0, 300)
        wl_nm = np.asarray(wl_nm, dtype=float)
        f_abs = C_LIGHT / (wl_nm * 1e-9)
        r = self.reflection_spectrum(f_abs)
        hd_db = 10.0 * np.log10(np.clip(np.abs(r) ** 2, 1e-12, None))
        return wl_nm, hd_db

    def peak_reflectivity(self, wl_nm=None) -> float:
        """Global max |r| over the device response band (not a local grid window)."""
        if wl_nm is None:
            wl_nm = np.linspace(1520.0, 1580.0, 601)
        wl_nm = np.asarray(wl_nm, dtype=float)
        f_abs = C_LIGHT / (wl_nm * 1e-9)
        return float(np.max(np.abs(self.reflection_spectrum(f_abs))))

    # ------------------------------------------------------------------
    # Transfer-matrix engine
    # ------------------------------------------------------------------

    def _alpha(self) -> float:
        return self.loss_dBcm * 100.0 / (20.0 * np.log10(np.e))

    def _section_matrices(self, wl: np.ndarray, dn_th: float) -> np.ndarray:
        """Exact transfer matrices for every (wavelength, section) pair.

        Returns shape ``(n_wl, n_sect, 4, 4)``. One batched ``expm`` per call
        replaces ``n_wl * n_sect`` scalar calls, which dominated the runtime.
        """
        w = wl[:, None]  # (n_wl, 1) against (1, n_sect) section arrays
        n1 = (
            self.n1_0
            - (self.ng1 - self.n1_0) / self.wl0 * (w - self.wl0)
            + dn_th
            + self.dn1[None, :]
        )
        n2 = (
            self.n2_0
            - (self.ng2 - self.n2_0) / self.wl0 * (w - self.wl0)
            + dn_th
            + self.dn2[None, :]
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

    def _transfer_spectrum(self, wl: np.ndarray, return_ports: bool = False):
        """Evaluate drop-port r, or all four port amplitudes, on wavelengths (m)."""
        wl = np.asarray(wl, dtype=float).ravel()
        if np.any(~np.isfinite(wl)) or np.any(wl <= 0.0):
            raise ValueError("wavelengths must be finite and > 0")
        n_freq = wl.size
        dn_th = DN_DT * self.temperature_offset_K

        B1 = np.empty(n_freq, dtype=complex)
        B2 = np.empty(n_freq, dtype=complex)
        F1 = np.empty(n_freq, dtype=complex)
        F2 = np.empty(n_freq, dtype=complex)

        # Chunk the frequency axis so the section-matrix batch stays bounded.
        bytes_per_wl = self.n_sect * 16 * np.dtype(complex).itemsize
        chunk = max(1, self._BATCH_BYTES // max(bytes_per_wl, 1))

        for lo in range(0, n_freq, chunk):
            hi = min(lo + chunk, n_freq)
            C = self._section_matrices(wl[lo:hi], dn_th)
            M = np.broadcast_to(np.eye(4, dtype=complex), (hi - lo, 4, 4)).copy()
            for i in range(self.n_sect):
                M = C[:, i] @ M
            B1[lo:hi], B2[lo:hi], F1[lo:hi], F2[lo:hi] = self._bvp_solve(M)

        # Non-finite output means the cascade overflowed (very strong or long
        # grating). Fail loudly rather than leaking NaN into the fingerprint.
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
            return {"B1_0": B1, "B2_0": B2, "F1_L": F1, "F2_L": F2}
        return B2


# Back-compat alias
ContraDCPuf = ContraDCPUF


if __name__ == "__main__":
    puf = ContraDCPUF(device_id=1)
    print(f"device_id={puf.device_id}, n_sect={puf.n_sect}, tau_max={puf.tau_max*1e12:.2f} ps")

    wl_nm, hd_db = puf.drop_port_response()
    print(f"λ: {wl_nm[0]:.1f}–{wl_nm[-1]:.1f} nm")
    print(f"|r|^2: min={hd_db.min():.1f} dB, max={hd_db.max():.1f} dB")

    f0 = C_LIGHT / 1550e-9
    f_abs = C_LIGHT / (wl_nm * 1e-9)
    ports = puf.port_amplitudes(f_abs)
    passive = np.max(
        np.abs(ports["B1_0"]) ** 2
        + np.abs(ports["B2_0"]) ** 2
        + np.abs(ports["F1_L"]) ** 2
        + np.abs(ports["F2_L"]) ** 2
    )
    print(f"max four-port power = {passive:.6f}")
    if passive > 1.0 + 1e-9:
        print("  WARNING: passivity violated — check transfer-matrix sign conventions")

    puf2 = ContraDCPUF(device_id=2)
    r = ports["B2_0"]
    r2 = puf2.reflection_spectrum(f_abs)
    corr = np.corrcoef(np.abs(r), np.abs(r2))[0, 1]
    print(f"|r| corr (id1 vs id2) = {corr:.3f}")

    # Small grid for a quick h(t) smoke check (full SimGrid is heavier)
    grid = SimGrid(fs=4e12, n=2048)
    h = puf.impulse_response(grid, f0)
    print(f"h(t): {len(h)} samples, max|h|={np.max(np.abs(h)):.3e}")
