"""
WDM modulator for the PUF readout chain.

Challenge bits -> per-channel power envelopes -> chirped complex fields
-> single complex baseband sum on SimGrid.

Levels are optical *power*; the field is sqrt(power) * exp(j*phi).
Chirp uses the alpha-parameter MRM model (first-cut, not a full ring model).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

C_LIGHT = 299792458.0

# Optical power levels (normalized 0…1). Field = sqrt(power).
NRZ_POWER = np.array([0.0, 1.0])
PAM4_POWER = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])

# 2-bit gray code word -> PAM4 level (adjacent levels differ by one bit)
_GRAY_TO_LEVEL = np.empty(4, dtype=int)
_GRAY_TO_LEVEL[0b00] = 0
_GRAY_TO_LEVEL[0b01] = 1
_GRAY_TO_LEVEL[0b11] = 2
_GRAY_TO_LEVEL[0b10] = 3

# Rectangular hold needs enough samples/symbol to stay meaningful.
_MIN_SPS = 16


@dataclass(frozen=True)
class WDMGrid:
    """WDM channel plan. SI units (m). Fixed system input; retune the grating to it."""

    lambda_start: float = 1548e-9
    delta_lambda: float = 1.2e-9
    n_channels: int = 4

    @classmethod
    def from_nm(cls, wl0_nm: float, spacing_nm: float, n_channels: int) -> WDMGrid:
        return cls(wl0_nm * 1e-9, spacing_nm * 1e-9, int(n_channels))

    def wavelengths(self) -> np.ndarray:
        return self.lambda_start + self.delta_lambda * np.arange(self.n_channels)

    def frequencies(self) -> np.ndarray:
        return C_LIGHT / self.wavelengths()

    def offsets(self, f0: float) -> np.ndarray:
        return self.frequencies() - float(f0)


@dataclass
class ModulatedField:
    """Complex baseband field entering the PUF, plus ADC timing metadata."""

    field: np.ndarray  # complex, length simgrid.n, units sqrt(W)
    channel_fields: np.ndarray  # complex, shape (n_channels, simgrid.n), pre-sum
    p_launch_ref: float  # W, tap normalization reference
    symbol_edges: np.ndarray  # sample indices where symbols start


class Modulator:
    """Challenge-driven WDM modulator (NRZ / PAM4) over a carrier grid."""

    def __init__(
        self,
        grid: WDMGrid,
        symbol_rate: float = 50e9,  # baud; bitrate = symbol_rate * bits/symbol
        fmt: str = "nrz",
        gray_map: bool = True,
        extinction_ratio_dB: float = 6.0,
        chirp_alpha: float = 3.0,
        mod_bw_factor: float = 0.75,
        p_launch: float = 1e-3,
        p_idle: float | None = None,
    ):
        self.grid = grid
        self.symbol_rate = float(symbol_rate)
        if self.symbol_rate <= 0.0:
            raise ValueError("symbol_rate must be > 0")

        self.fmt = fmt.lower()
        if self.fmt not in ("nrz", "pam4"):
            raise ValueError('fmt must be "nrz" or "pam4"')

        self.gray_map = bool(gray_map)
        self.extinction_ratio_dB = float(extinction_ratio_dB)
        self.er_linear = 10.0 ** (self.extinction_ratio_dB / 10.0)
        self.chirp_alpha = float(chirp_alpha)
        self.mod_bw_factor = float(mod_bw_factor)
        self.p_launch = float(p_launch)
        # Explicit idle fill (default = off level). Do not reuse power[0]: that
        # only keeps the FFT wrap continuous by accident.
        self.p_idle = (
            1.0 / self.er_linear if p_idle is None else float(p_idle)
        )
        if self.p_idle <= 0.0:
            raise ValueError("p_idle must be > 0")

        self.symbol_period_s = 1.0 / self.symbol_rate

    @property
    def bitrate(self) -> float:
        """Bit rate (bit/s). Twice symbol_rate for PAM4, equal for NRZ."""
        return self.symbol_rate * (1 if self.fmt == "nrz" else 2)

    @property
    def n_channels(self) -> int:
        return self.grid.n_channels

    @staticmethod
    def random_challenge(
        n_channels: int, n_bits: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Independent per-channel challenge bits, shape (n_channels, n_bits)."""
        return rng.integers(0, 2, size=(n_channels, n_bits), dtype=np.uint8)

    def modulate(self, challenge: np.ndarray, simgrid, f0: float) -> ModulatedField:
        """
        challenge : uint8 array, shape (n_channels, n_bits), values 0/1
                    NRZ: 1 bit/symbol. PAM4: 2 bits/symbol (even n_bits).
        simgrid   : SimGrid-like (fs, n, t, dt, f)
        f0        : reference optical frequency (Hz) for complex baseband
        """
        challenge = np.asarray(challenge, dtype=np.uint8)
        if challenge.ndim != 2:
            raise ValueError("challenge must have shape (n_channels, n_bits)")
        if challenge.shape[0] != self.n_channels:
            raise ValueError(
                f"challenge has {challenge.shape[0]} channels, grid has {self.n_channels}"
            )
        if not np.all(np.isin(challenge, (0, 1))):
            raise ValueError("challenge bits must be 0 or 1")

        sps = max(1, int(round(self.symbol_period_s * simgrid.fs)))
        if sps < _MIN_SPS:
            warnings.warn(
                f"sps={sps} < {_MIN_SPS}; rectangular hold is poorly resolved",
                stacklevel=2,
            )

        offsets = self.grid.offsets(f0)
        max_offset = float(np.max(np.abs(offsets))) if offsets.size else 0.0
        nyquist = 0.5 * float(simgrid.fs)
        if max_offset >= nyquist:
            raise ValueError(
                f"carrier offset {max_offset:.6e} Hz exceeds Nyquist "
                f"{nyquist:.6e} Hz (fs/2); reduce delta_lambda or raise fs"
            )

        t = simgrid.t
        channel_fields = np.zeros((self.n_channels, simgrid.n), dtype=complex)
        symbol_edges = None
        burst_end = 0

        for ch in range(self.n_channels):
            symbols = self._bits_to_symbols(challenge[ch])
            power = self._symbols_to_power(symbols)
            n_sym = power.size

            n_needed = n_sym * sps
            if n_needed > simgrid.n:
                raise ValueError(
                    f"burst needs {n_needed} samples "
                    f"(n_sym={n_sym} * sps={sps}) but simgrid.n={simgrid.n}; "
                    f"raise n, shorten challenge, or increase symbol_rate"
                )
            burst_end = max(burst_end, n_needed)

            edges = np.arange(n_sym, dtype=int) * sps
            if symbol_edges is None:
                symbol_edges = edges.copy()

            # Rectangular hold onto the sim grid; idle tail uses p_idle
            p = np.full(simgrid.n, self.p_idle)
            for k in range(n_sym):
                start = int(edges[k])
                end = start + sps
                p[start:end] = power[k]

            # Bandwidth limit acts on the modulator *drive*; filtering optical
            # power is an approximation. The Gaussian kernel is strictly
            # positive, so there is no undershoot — do not "fix" this into a
            # Bessel on power (that would produce negative powers).
            p = self._gaussian_lpf(p, simgrid.fs, self.mod_bw_factor * self.symbol_rate)
            p = np.clip(p, 1e-18, None)

            # Alpha-parameter chirp on the power envelope (first-cut MRM model)
            p_mean = float(np.mean(p[:burst_end])) if burst_end else float(np.mean(p))
            phi = 0.5 * self.chirp_alpha * np.log(p / p_mean)
            e_ch = np.sqrt(p) * np.exp(1j * phi)

            channel_fields[ch] = e_ch * np.exp(1j * 2.0 * np.pi * offsets[ch] * t)

        x = channel_fields.sum(axis=0)

        # Normalize launch power over the burst only — idle tail is not launch.
        if burst_end <= 0:
            raise RuntimeError("modulated field has empty burst")
        p_avg = float(np.mean(np.abs(x[:burst_end]) ** 2))
        if p_avg < 1e-30:
            raise RuntimeError("modulated field has near-zero power")
        scale = np.sqrt(self.p_launch / p_avg)
        x *= scale
        channel_fields *= scale

        return ModulatedField(
            field=x,
            channel_fields=channel_fields,
            p_launch_ref=self.p_launch,
            symbol_edges=symbol_edges if symbol_edges is not None else np.array([], dtype=int),
        )

    def check_carriers_in_stopband(self, puf, threshold: float = 0.1) -> list[str]:
        """
        Warn if any WDM carrier sits below `threshold` * global peak |r|.

        Peak reference must be the device's full response band, not a window
        around the grid — a grid parked outside the stopband would otherwise
        pass with a near-1 local ratio while every carrier is dark.
        """
        peak = float(puf.peak_reflectivity())
        if peak < 1e-12:
            return ["PUF |r| peak is ~0; cannot place carriers"]

        f_ch = self.grid.frequencies()
        # Mode-summed power (single-PD); falls back to |r0|^2 when M=1.
        if hasattr(puf, "reflection_power_spectrum"):
            p_ch = puf.reflection_power_spectrum(f_ch)
            amp_ch = np.sqrt(np.clip(p_ch, 0.0, None))
        else:
            amp_ch = np.abs(puf.reflection_spectrum(f_ch))
        warnings_out = []
        for i, (fi, ai) in enumerate(zip(f_ch, amp_ch)):
            if ai < threshold * peak:
                wl_nm = C_LIGHT / fi * 1e9
                warnings_out.append(
                    f"channel {i} @ {wl_nm:.2f} nm: |r|/|r|_peak = {ai/peak:.3f} < {threshold}"
                )
        return warnings_out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=int).ravel()
        if self.fmt == "nrz":
            return bits.copy()

        if bits.size % 2 != 0:
            raise ValueError("PAM4 challenge length per channel must be even (2 bits/symbol)")
        pairs = bits.reshape(-1, 2)
        natural = (pairs[:, 0] << 1) | pairs[:, 1]
        if self.gray_map:
            return _GRAY_TO_LEVEL[natural]
        return natural

    def _symbols_to_power(self, symbols: np.ndarray) -> np.ndarray:
        """Map symbols to optical power in [1/ER, 1]."""
        base = NRZ_POWER if self.fmt == "nrz" else PAM4_POWER
        if np.any(symbols < 0) or np.any(symbols >= len(base)):
            raise ValueError(f"symbol out of range for {self.fmt}")
        # Finite extinction: floor at 1/ER instead of perfect zero
        p_off = 1.0 / self.er_linear
        return p_off + (1.0 - p_off) * base[symbols]

    @staticmethod
    def _gaussian_lpf(x: np.ndarray, fs: float, f3db: float) -> np.ndarray:
        """Gaussian low-pass; |H(f3db)| = 1/sqrt(2)."""
        f = np.fft.fftfreq(x.size, d=1.0 / fs)
        f0 = f3db / np.sqrt(np.log(2.0))
        H = np.exp(-0.5 * (f / max(f0, 1.0)) ** 2)
        return np.fft.ifft(np.fft.fft(x) * H).real


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from puf.puf_model import ContraDCPUF, SimGrid

    grid = WDMGrid()  # 1548 nm, 1.2 nm, 4 ch — plan is fixed; grating retuned to it
    sim = SimGrid(fs=4e12, n=16384)
    f0 = C_LIGHT / 1550e-9
    rng = np.random.default_rng(0)
    puf = ContraDCPUF(device_id=0)

    for fmt in ("nrz", "pam4"):
        mod = Modulator(grid=grid, symbol_rate=50e9, fmt=fmt)
        n_bits = 32 if fmt == "nrz" else 64
        challenge = Modulator.random_challenge(grid.n_channels, n_bits, rng)
        out = mod.modulate(challenge, sim, f0)
        burst = int(out.symbol_edges[-1]) + int(round(mod.symbol_period_s * sim.fs))
        p_avg = float(np.mean(np.abs(out.field[:burst]) ** 2))
        print(f"--- {fmt.upper()} ---")
        print(f"grid λ/nm = {grid.wavelengths()*1e9}")
        print(f"symbol_rate={mod.symbol_rate:.3e} baud, bitrate={mod.bitrate:.3e} bit/s")
        print(f"challenge shape={challenge.shape}, symbols/ch≈{n_bits // (1 if fmt=='nrz' else 2)}")
        print(f"|field|={out.field.size}, channel_fields={out.channel_fields.shape}")
        print(f"p_avg(burst)={p_avg:.6e} W (target {mod.p_launch})")
        print(f"symbol_edges[:5]={out.symbol_edges[:5]}, sps≈{int(round(mod.symbol_period_s*sim.fs))}")
        print()

    warns = Modulator(grid=grid).check_carriers_in_stopband(puf)
    print(f"stopband check warnings: {warns if warns else '(none)'}")
    print(
        f"PUF Bragg ≈ {puf.Lambda0 * (puf.n1_0 + puf.n2_0) * 1e9:.2f} nm, "
        f"|r|_peak={puf.peak_reflectivity():.4f}"
    )

    # Symbol-rate sweep must change the field (A2); both bursts must fit simgrid.n
    ch = Modulator.random_challenge(4, 16, np.random.default_rng(1))
    a = Modulator(grid, symbol_rate=25e9).modulate(ch, sim, f0).field
    b = Modulator(grid, symbol_rate=50e9).modulate(ch, sim, f0).field
    print(f"symbol_rate sweep field diff RMS = {np.sqrt(np.mean(np.abs(a-b)**2)):.3e}")
