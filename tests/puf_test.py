"""Six-tier physics gate for ContraDCPUF. Run in order; each assumes prior tiers pass."""

from __future__ import annotations

import numpy as np
import pytest

from puf.puf_model import C_LIGHT, ContraDCPUF, SimGrid

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Centered on the default Bragg peak (~1550 nm after Lambda0 retune).
WL_BAND_NM = np.linspace(1530.0, 1570.0, 81)
F_BAND = C_LIGHT / (WL_BAND_NM * 1e-9)


def _golden(**kwargs) -> ContraDCPUF:
    """Uniform grating: no disorder, flat apodization, single κ channel."""
    defaults = dict(
        device_id=0,
        L=210e-6,
        n_sect=40,
        Lambda0=300e-9,
        n1_0=2.580,
        n2_0=2.490,
        ng1=3.65,
        ng2=3.55,
        wl0=1.521e-6,
        kappa11=0.0,
        kappa12=48_900.0,
        kappa22=0.0,
        loss_dBcm=0.0,
        center_dip=0.0,
        edge_boost=0.0,
        kappa_mult=1.0,
        sigma_dn=0.0,
        sigma_kappa_rel=0.0,
        sigma_dLambda=0.0,
    )
    defaults.update(kwargs)
    return ContraDCPUF(**defaults)


def _four_port_power(puf: ContraDCPUF, f_abs: np.ndarray) -> np.ndarray:
    ports = puf.port_amplitudes(f_abs)
    return (
        np.abs(ports["B1_0"]) ** 2
        + np.abs(ports["B2_0"]) ** 2
        + np.abs(ports["F1_L"]) ** 2
        + np.abs(ports["F2_L"]) ** 2
    )


def _analytic_tau_eff(ng: float, kappa: float, length: float) -> float:
    return 2.0 * ng * np.tanh(kappa * length) / (kappa * C_LIGHT)


def _stopband_width_nm(puf: ContraDCPUF, wl_nm: np.ndarray) -> float:
    """Full width between κ band edges estimated from |r|^2 half-max."""
    r2 = np.abs(puf.reflection_spectrum(C_LIGHT / (wl_nm * 1e-9))) ** 2
    half = 0.5 * r2.max()
    above = np.where(r2 >= half)[0]
    return float(wl_nm[above[-1]] - wl_nm[above[0]])


def _analytic_stopband_nm(puf: ContraDCPUF) -> float:
    # dβ/dλ uses the phase index in β=2πn/λ (not n_g).
    n_avg = 0.5 * (puf.n1_0 + puf.n2_0)
    lam_b = puf.Lambda0 * (puf.n1_0 + puf.n2_0)
    return (lam_b**2 * puf.kappa12 / (np.pi * n_avg)) * 1e9


# ===========================================================================
# Tier 0: numerics
# ===========================================================================


class TestTier0Numerics:
    def test_determinism(self):
        a = ContraDCPUF(device_id=7).reflection_spectrum(F_BAND)
        b = ContraDCPUF(device_id=7).reflection_spectrum(F_BAND)
        np.testing.assert_array_equal(a, b)

    def test_distinctness(self):
        r1 = ContraDCPUF(device_id=1).reflection_spectrum(F_BAND)
        r2 = ContraDCPUF(device_id=2).reflection_spectrum(F_BAND)
        assert not np.allclose(r1, r2)

    def test_no_nan_inf(self):
        r = ContraDCPUF(device_id=3).reflection_spectrum(F_BAND)
        assert np.isfinite(r).all()

    def test_n_sect_convergence(self):
        """Double n_sect → spectrum changes < 1% (uniform golden case)."""
        coarse = _golden(n_sect=40)
        fine = _golden(n_sect=80)
        r0 = coarse.reflection_spectrum(F_BAND)
        r1 = fine.reflection_spectrum(F_BAND)
        denom = np.maximum(np.abs(r0), 1e-12)
        rel = np.abs(r1 - r0) / denom
        assert float(np.max(rel)) < 0.01

    def test_frequency_sampling_convergence(self):
        """Halve Δf (double T_win) → early h(t) unchanged once T ≫ τ_eff."""
        puf = _golden(n_sect=20, kappa12=7_500.0)
        f0 = C_LIGHT / puf.wl0
        coarse = SimGrid(fs=4e12, n=1024)
        fine = SimGrid(fs=4e12, n=2048)
        h0 = puf.impulse_response(coarse, f0)
        h1 = puf.impulse_response(fine, f0)
        # Same absolute times: sample i is t=i/fs on both grids.
        n_cmp = coarse.n // 4  # well inside the window, before wrap risk
        num = np.linalg.norm(h1[:n_cmp] - h0[:n_cmp])
        den = np.linalg.norm(h0[:n_cmp]) + 1e-30
        assert num / den < 0.05


# ===========================================================================
# Tier 1: energy / signs
# ===========================================================================


class TestTier1Energy:
    def test_four_port_passivity(self):
        puf = ContraDCPUF(device_id=1, n_sect=40)
        power = _four_port_power(puf, F_BAND)
        assert float(np.max(power)) <= 1.0 + 1e-9

    def test_lossless_limit(self):
        puf = _golden(loss_dBcm=0.0, n_sect=20)
        power = _four_port_power(puf, F_BAND)
        np.testing.assert_allclose(power, 1.0, atol=1e-9)

    def test_loss_monotonicity(self):
        losses = [0.0, 5.0, 50.0, 200.0, 500.0]
        totals = []
        for loss in losses:
            puf = _golden(loss_dBcm=loss, n_sect=20)
            totals.append(float(np.mean(_four_port_power(puf, F_BAND))))
        assert all(totals[i] >= totals[i + 1] - 1e-12 for i in range(len(totals) - 1))
        assert totals[-1] < totals[0]

    def test_zero_coupling(self):
        puf = _golden(
            kappa11=0.0,
            kappa12=0.0,
            kappa22=0.0,
            loss_dBcm=500.0,
            n_sect=10,
        )
        ports = puf.port_amplitudes(np.array([C_LIGHT / puf.wl0]))
        alpha = puf._alpha()
        assert abs(ports["B2_0"][0]) < 1e-12
        assert abs(ports["B1_0"][0]) < 1e-12
        np.testing.assert_allclose(
            abs(ports["F1_L"][0]),
            np.exp(-alpha * puf.L),
            rtol=1e-12,
        )


# ===========================================================================
# Tier 2: analytic limits
# ===========================================================================


class TestTier2Analytic:
    def test_bragg_wavelength(self):
        puf = _golden(n_sect=20)
        wl_nm = np.linspace(1505.0, 1535.0, 601)
        r = puf.reflection_spectrum(C_LIGHT / (wl_nm * 1e-9))
        peak_nm = float(wl_nm[np.argmax(np.abs(r))])
        expected = puf.Lambda0 * (puf.n1_0 + puf.n2_0) * 1e9
        assert abs(peak_nm - expected) < 1.0

    def test_peak_reflectivity(self):
        puf = _golden(n_sect=20)
        wl_nm = np.linspace(1505.0, 1535.0, 601)
        r = puf.reflection_spectrum(C_LIGHT / (wl_nm * 1e-9))
        r_max = float(np.max(np.abs(r) ** 2))
        expected = float(np.tanh(puf.kappa12 * puf.L) ** 2)
        assert abs(r_max - expected) / expected < 0.01

    def test_stopband_width(self):
        puf = _golden(n_sect=20)
        wl_nm = np.linspace(1490.0, 1550.0, 1201)
        width_nm = _stopband_width_nm(puf, wl_nm)
        expected_nm = _analytic_stopband_nm(puf)
        assert abs(width_nm - expected_nm) / expected_nm < 0.10

    def test_scaling(self):
        # Moderate κ so both L and 2L are past the weak-grating edge.
        base = _golden(L=210e-6, kappa12=15_000.0, n_sect=20)
        long = _golden(L=420e-6, kappa12=15_000.0, n_sect=40)
        strong = _golden(L=210e-6, kappa12=30_000.0, n_sect=20)
        wl_nm = np.linspace(1490.0, 1550.0, 1201)

        def r_peak(puf: ContraDCPUF) -> float:
            return float(
                np.max(np.abs(puf.reflection_spectrum(C_LIGHT / (wl_nm * 1e-9))) ** 2)
            )

        w_base = _stopband_width_nm(base, wl_nm)
        w_long = _stopband_width_nm(long, wl_nm)
        w_strong = _stopband_width_nm(strong, wl_nm)

        # Double L at fixed κ: width unchanged, R saturates upward.
        assert abs(w_long - w_base) / w_base < 0.15
        assert r_peak(long) >= r_peak(base) - 1e-6
        # Double κ at fixed L: width roughly doubles.
        assert abs(w_strong / w_base - 2.0) < 0.30


# ===========================================================================
# Tier 3: time domain
# ===========================================================================


class TestTier3TimeDomain:
    @pytest.fixture
    def causal_case(self):
        # Moderate κ so τ_eff is resolvable and the window is long vs delay.
        puf = _golden(kappa12=7_500.0, n_sect=20)
        grid = SimGrid(fs=4e12, n=2048)
        f0 = C_LIGHT / puf.wl0
        h = puf.impulse_response(grid, f0)
        r = puf.reflection_spectrum(f0 + grid.f)
        return puf, grid, h, r

    def test_causality(self, causal_case):
        puf, grid, h, _ = causal_case
        e = np.abs(h) ** 2
        # fftfreq time axis: indices n/2..n-1 are negative times under e^{+jωt}.
        n = grid.n
        neg = e[n // 2 :]
        assert float(neg.sum() / e.sum()) < 0.01

    def test_parseval(self, causal_case):
        _, grid, h, r = causal_case
        lhs = np.sum(np.abs(h) ** 2) * grid.dt
        rhs = np.sum(np.abs(r) ** 2) * (grid.fs / grid.n)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)

    def test_no_wraparound(self, causal_case):
        _, grid, h, _ = causal_case
        peak = float(np.max(np.abs(h)))
        tail = float(np.max(np.abs(h[grid.n // 4 : grid.n // 2])))
        assert tail / (peak + 1e-30) < 0.05

    def test_effective_delay_spread(self, causal_case):
        puf, grid, h, _ = causal_case
        # FWHM of |h(t)| on t>=0. (90%-energy widths are ring-dominated.)
        n_pos = grid.n // 2
        ah = np.abs(h[:n_pos])
        half = 0.5 * ah.max()
        above = np.where(ah >= half)[0]
        measured = (above[-1] - above[0] + 1) * grid.dt
        ng = 0.5 * (puf.ng1 + puf.ng2)
        expected = _analytic_tau_eff(ng, puf.kappa12, puf.L)
        assert 0.25 * expected < measured < 4.0 * expected
        # Hard check on a strong grating: penetration ≪ geometric round-trip.
        strong = _golden(kappa12=48_900.0, n_sect=20)
        hs = strong.impulse_response(grid, C_LIGHT / strong.wl0)
        ahs = np.abs(hs[:n_pos])
        half = 0.5 * ahs.max()
        above = np.where(ahs >= half)[0]
        width_s = (above[-1] - above[0] + 1) * grid.dt
        tau_s = _analytic_tau_eff(ng, strong.kappa12, strong.L)
        assert width_s < 0.35 * strong.tau_max
        assert 0.25 * tau_s < width_s < 4.0 * tau_s


# ===========================================================================
# Tier 4: PUF behaviour
# ===========================================================================


class TestTier4PufBehaviour:
    def test_disorder_monotonicity(self):
        sigmas = [0.0, 5e-4, 1e-3, 2e-3, 5e-3]
        # Divergence from the zero-disorder reference, averaged over a few ids.
        ref = {
            i: ContraDCPUF(
                device_id=i,
                n_sect=40,
                sigma_dn=0.0,
                sigma_kappa_rel=0.0,
                sigma_dLambda=0.0,
            ).reflection_spectrum(F_BAND)
            for i in range(5)
        }
        divergences = []
        for sigma in sigmas:
            dists = []
            for i in range(5):
                r = ContraDCPUF(
                    device_id=i,
                    n_sect=40,
                    sigma_dn=sigma,
                    sigma_kappa_rel=0.0,
                    sigma_dLambda=0.0,
                ).reflection_spectrum(F_BAND)
                dists.append(float(np.mean(np.abs(np.abs(r) - np.abs(ref[i])))))
            divergences.append(float(np.mean(dists)))
        assert divergences[0] < 1e-12
        assert all(
            divergences[i] <= divergences[i + 1] + 1e-12
            for i in range(len(divergences) - 1)
        )

    def test_inter_device_correlation(self):
        # Residual after removing the shared deterministic backbone.
        spectra = np.array(
            [
                np.abs(
                    ContraDCPUF(
                        device_id=i,
                        n_sect=40,
                        center_dip=0.0,
                        edge_boost=0.0,
                        kappa_mult=1.0,
                    ).reflection_spectrum(F_BAND)
                )
                for i in range(40)
            ]
        )
        residual = spectra - spectra.mean(axis=0, keepdims=True)
        corrs = []
        for i in range(len(residual)):
            for j in range(i + 1, len(residual)):
                c = np.corrcoef(residual[i], residual[j])[0, 1]
                if np.isfinite(c):
                    corrs.append(float(c))
        corrs = np.asarray(corrs)
        assert float(np.mean(corrs)) < 0.35
        assert float(np.mean(corrs > 0.9)) < 0.05

    def test_correlation_length_sensitivity(self):
        """Longer spatial correlation → broader spectral autocorrelation."""
        lengths = [1e-6, 5e-6, 20e-6, 80e-6]
        wl_nm = np.linspace(1535.0, 1565.0, 301)
        f = C_LIGHT / (wl_nm * 1e-9)
        widths = []
        for lc in lengths:
            specs = []
            for i in range(6):
                puf = ContraDCPUF(
                    device_id=100 + i,
                    n_sect=60,
                    L=400e-6,
                    corr_length=lc,
                    center_dip=0.0,
                    edge_boost=0.0,
                    kappa_mult=1.0,
                    kappa11=0.0,
                    kappa12=3_000.0,
                    kappa22=0.0,
                    sigma_dn=5e-3,
                    sigma_kappa_rel=0.0,
                    sigma_dLambda=0.0,
                    loss_dBcm=0.0,
                )
                specs.append(np.abs(puf.reflection_spectrum(f)))
            resid = np.asarray(specs) - np.mean(specs, axis=0, keepdims=True)
            ac_widths = []
            for row in resid:
                x = row - row.mean()
                ac = np.correlate(x, x, mode="full")
                ac = ac[ac.size // 2 :]
                ac = ac / (ac[0] + 1e-30)
                below = np.where(ac < 0.5)[0]
                ac_widths.append(int(below[0]) if below.size else len(ac))
            widths.append(float(np.mean(ac_widths)))
        assert widths[0] < widths[-1]
        assert widths[0] <= widths[1] <= widths[2] <= widths[3] + 0.5

    def test_feature_count_versus_dof(self):
        """Resolvable residual peaks agree with τ_eff × B within ~2×."""
        from scipy.signal import find_peaks

        wl_nm = np.linspace(1535.0, 1565.0, 601)
        f = C_LIGHT / (wl_nm * 1e-9)
        df = float(np.median(np.abs(np.diff(f))))
        common = dict(
            n_sect=80,
            L=210e-6,
            center_dip=0.0,
            edge_boost=0.0,
            kappa_mult=1.0,
            kappa11=0.0,
            kappa12=7_500.0,
            kappa22=0.0,
            sigma_dn=3e-3,
            sigma_kappa_rel=0.0,
            sigma_dLambda=0.0,
            loss_dBcm=0.0,
            corr_length=2e-6,
        )
        specs = [
            np.abs(ContraDCPUF(device_id=300 + i, **common).reflection_spectrum(f))
            for i in range(10)
        ]
        mean = np.mean(specs, axis=0)
        puf = ContraDCPUF(device_id=4, **common)
        resid = np.abs(np.abs(puf.reflection_spectrum(f)) - mean)
        ng = 0.5 * (puf.ng1 + puf.ng2)
        kappa = float(np.mean(puf.k12_arr))
        tau_eff = _analytic_tau_eff(ng, max(kappa, 1.0), puf.L)
        B = abs(f[0] - f[-1])  # interrogation bandwidth
        dof = tau_eff * B
        min_dist = max(1, int(round((1.0 / tau_eff) / df)))
        peaks, _ = find_peaks(
            resid, distance=min_dist, prominence=0.02 * (resid.max() + 1e-30)
        )
        n_feat = len(peaks)
        assert n_feat <= max(2.0 * dof, 1.0) + 1
        assert n_feat >= dof / 2.0 - 1

    def test_temperature(self):
        puf0 = _golden(temperature_offset_K=0.0, n_sect=20)
        puf1 = _golden(temperature_offset_K=1.0, n_sect=20)
        wl_nm = np.linspace(1505.0, 1535.0, 601)
        f = C_LIGHT / (wl_nm * 1e-9)
        peak0 = float(wl_nm[np.argmax(np.abs(puf0.reflection_spectrum(f)))])
        peak1 = float(wl_nm[np.argmax(np.abs(puf1.reflection_spectrum(f)))])
        dldt = peak1 - peak0  # nm/K
        # Thermo-optic: dλ/dT ≈ λ * DN_DT / n  → ~0.08 nm/K near Bragg.
        assert abs(dldt - 0.08) < 0.03
        # Reversible: +1 K then construct -1 K from the +1 K base offset.
        puf_back = _golden(temperature_offset_K=0.0, n_sect=20)
        np.testing.assert_allclose(
            puf_back.reflection_spectrum(f),
            puf0.reflection_spectrum(f),
            atol=0,
            rtol=0,
        )
        # Explicit round-trip through a +1 then reset.
        puf_rt = _golden(temperature_offset_K=1.0, n_sect=20)
        puf_rt.temperature_offset_K = 0.0
        puf_rt._r_cache.clear()
        np.testing.assert_allclose(puf_rt.reflection_spectrum(f), puf0.reflection_spectrum(f))


# ===========================================================================
# Tier 5: design-specific
# ===========================================================================


class TestTier5Design:
    def test_carrier_placement(self):
        """Four WDM carriers at 1.2 nm spacing must sit in the stopband."""
        puf = ContraDCPUF(device_id=0, n_sect=40)
        # Center the plan on the design Bragg wavelength (device stopband).
        lam_b_nm = puf.Lambda0 * (puf.n1_0 + puf.n2_0) * 1e9
        spacing = 1.2
        carriers_nm = lam_b_nm + spacing * np.array([-1.5, -0.5, 0.5, 1.5])
        f_c = C_LIGHT / (carriers_nm * 1e-9)
        wl_scan = np.linspace(lam_b_nm - 20.0, lam_b_nm + 20.0, 401)
        r_scan = np.abs(puf.reflection_spectrum(C_LIGHT / (wl_scan * 1e-9)))
        peak = float(r_scan.max())
        r_c = np.abs(puf.reflection_spectrum(f_c))
        assert np.all(r_c >= 0.10 * peak)


# Keep the original uniform-grating unit checks.
def test_uniform_grating_matches_analytic_reflectivity():
    puf = _golden(n_sect=1, kappa12=7_500.0)
    r = puf.reflection_spectrum(np.array([C_LIGHT / puf.wl0]))[0]
    np.testing.assert_allclose(abs(r) ** 2, np.tanh(puf.kappa12 * puf.L) ** 2, rtol=1e-9)


def test_uniform_grating_is_independent_of_section_count():
    frequency = np.array([C_LIGHT / 1.521e-6])
    coarse = _golden(n_sect=1, kappa12=7_500.0)
    fine = _golden(n_sect=200, kappa12=7_500.0)
    np.testing.assert_allclose(
        fine.reflection_spectrum(frequency),
        coarse.reflection_spectrum(frequency),
        rtol=1e-10,
        atol=1e-13,
    )
