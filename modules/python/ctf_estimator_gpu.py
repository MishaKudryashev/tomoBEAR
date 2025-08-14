"""GPU-accelerated CTF estimation in Python.

This module replaces the external GCTF program used by TomoBEAR for
contrast transfer function (CTF) estimation.  It uses CuPy for GPU
acceleration when available, falling back to NumPy otherwise.  The core
routine estimates the defocus of a micrograph by comparing its radial
power spectrum against a set of theoretical CTF curves.

The implementation is intentionally lightweight; it supports only a
subset of the features of the original GCTF tool but provides a pure
Python/CUDA solution that can be extended further.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Union

try:  # pragma: no cover - exercised only when GPU libraries present
    import cupy as xp  # type: ignore
    _GPU_AVAILABLE = True
except Exception:  # pragma: no cover - CPU fall back used in tests
    import numpy as xp  # type: ignore
    _GPU_AVAILABLE = False

import numpy as np

try:  # pragma: no cover - optional dependency
    import mrcfile
except Exception:  # pragma: no cover
    mrcfile = None


@dataclass
class CTFResult:
    """Container for CTF estimation results."""

    defocus_angstrom: float
    best_frequency: float
    correlation: float


def _electron_wavelength(voltage_kv: float) -> float:
    """Return electron wavelength in Ångström for the given accelerating voltage."""

    # Relativistic electron wavelength in meters
    m0c2 = 510.99895000  # keV
    lamb_ang = 12.3986 / math.sqrt(voltage_kv * 1000 * (1 + voltage_kv / (2 * m0c2)))
    return lamb_ang


def _radial_average(img: xp.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial average of the 2‑D power spectrum."""

    ny, nx = img.shape
    cy, cx = ny // 2, nx // 2
    y, x = xp.ogrid[:ny, :nx]
    r = xp.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(xp.int32)
    tbin = xp.bincount(r.ravel(), img.ravel())
    nr = xp.bincount(r.ravel())
    radial_mean = tbin / xp.maximum(nr, 1)
    freqs = xp.fft.fftfreq(ny)[: radial_mean.size]
    return np.asarray(freqs.get()), np.asarray(radial_mean.get())


def _ctf_1d(freqs: np.ndarray, defocus_ang: float, pixel_size_ang: float, voltage_kv: float) -> np.ndarray:
    """Compute 1‑D CTF curve squared for correlation comparison."""

    lamb = _electron_wavelength(voltage_kv)
    spatial = freqs / pixel_size_ang
    gamma = math.pi * lamb * defocus_ang * (spatial ** 2)
    ctf = np.sin(gamma)
    return ctf ** 2


def estimate_ctf(micrograph: Union[str, np.ndarray], pixel_size_ang: float, voltage_kv: float = 300.0,
                 defocus_range_ang: Tuple[float, float] = (5000.0, 50000.0),
                 defocus_steps: int = 100) -> CTFResult:
    """Estimate defocus for a micrograph.

    Parameters
    ----------
    micrograph: Union[str, np.ndarray]
        Path to an MRC micrograph file or a 2‑D NumPy array containing the
        micrograph data.
    pixel_size_ang: float
        Pixel size in Ångström.
    voltage_kv: float, optional
        Microscope accelerating voltage in kV.
    defocus_range_ang: tuple, optional
        Search range for defocus in Ångström.
    defocus_steps: int, optional
        Number of defocus values to evaluate inside the range.

    Returns
    -------
    CTFResult
        Estimated defocus and accompanying diagnostic values.
    """

    if isinstance(micrograph, str):
        if mrcfile is None:
            raise ImportError("mrcfile library is required to read MRC files")
        with mrcfile.open(micrograph, permissive=True) as mrc:
            data = mrc.data.astype(np.float32)
    else:
        data = np.asarray(micrograph, dtype=np.float32)

    arr = xp.asarray(data)
    fft = xp.fft.fftshift(xp.fft.fft2(arr))
    power = xp.abs(fft) ** 2
    freqs, radial = _radial_average(power)

    defocus_values = np.linspace(defocus_range_ang[0], defocus_range_ang[1], defocus_steps)
    best_defocus = defocus_values[0]
    best_corr = -np.inf
    best_freq = 0.0
    for d in defocus_values:
        model = _ctf_1d(freqs, d, pixel_size_ang, voltage_kv)
        corr = np.corrcoef(model, radial)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_defocus = d
            best_freq = freqs[np.argmax(model)]
    return CTFResult(best_defocus, best_freq, best_corr)


def main():  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Estimate CTF defocus using GPU acceleration")
    parser.add_argument("mrc", help="Input micrograph in MRC format")
    parser.add_argument("pixel_size", type=float, help="Pixel size in Ångström")
    parser.add_argument("--voltage", type=float, default=300.0, help="Accelerating voltage in kV")
    args = parser.parse_args()

    result = estimate_ctf(args.mrc, args.pixel_size, args.voltage)
    print(f"Estimated defocus: {result.defocus_angstrom/10000:.2f} µm")
    if not _GPU_AVAILABLE:
        print("Warning: CuPy not available, computation ran on CPU")


if __name__ == "__main__":  # pragma: no cover
    main()
