"""The 11 normalised spectral indices, TOA conversion, and pan-sharpening.

Purpose
-------
Turn raw sensor data into the 11-channel feature stack the U-Net consumes,
exactly as specified in the paper:

* **TOA reflectance** (paper Eq. 2) for Landsat, ``DN / 10000`` for Sentinel-2
  L1C, clipped to [0, 1].
* **Brovey pan-sharpening** (paper Eq. 3) of the RGB composite using the 15 m
  panchromatic band, for Landsat 7/8/9. Used for annotation display only -- it
  does **not** feed the indices, which are computed from 30 m bands.
* **The 11 indices** of paper Table 2: eight normalised differences in [-1, 1]
  and three relative chromaticity values in [0, 1].

Inputs
------
- A band stack ``(6, H, W)`` ordered ``[Blue, Green, Red, NIR, SWIR1, SWIR2]``,
  as TOA reflectance.

Outputs
-------
- An index stack ``(11, H, W)`` in :data:`INDEX_NAMES` order.

Vectorization
-------------
Every function here is a whole-array numpy expression over the full scene.
There are no Python loops over pixels, and each helper accepts an arbitrarily
shaped array so the same code serves a single scene, a patch, or a stacked
time series.

Index formulas (paper Table 2)
------------------------------
=============================  ==========================================
Index                          Formula
=============================  ==========================================
NDSI                           (G - SWIR1) / (G + SWIR1)
NDWI_GAO                       (NIR - SWIR1) / (NIR + SWIR1)
NBR                            (NIR - SWIR2) / (NIR + SWIR2)
NDVI                           (NIR - R) / (NIR + R)
ND_SWIR1_SWIR2                 (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
ND_BLUE_RED                    (B - R) / (B + R)
ND_BLUE_NIR                    (B - NIR) / (B + NIR)
ND_GREEN_RED                   (G - R) / (G + R)
r_norm                         R / (R + G + B)
g_norm                         G / (R + G + B)
b_norm                         B / (R + G + B)
=============================  ==========================================
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

#: Stabiliser added to every denominator to avoid division by zero over
#: no-data/masked pixels, without perturbing normal reflectance ratios.
EPS = 1e-6

#: Canonical band order of the 6-band input stack.
BAND_ORDER = ("blue", "green", "red", "nir", "swir1", "swir2")

#: Canonical order of the 11 output channels. Must stay in lockstep with
#: ``features.spectral_indices`` in config.yaml and with the model's input
#: channel order -- reordering this silently invalidates a trained model.
INDEX_NAMES = (
    "NDVI",
    "NDSI",
    "NDWI_GAO",
    "NBR",
    "ND_SWIR1_SWIR2",
    "ND_BLUE_RED",
    "ND_BLUE_NIR",
    "ND_GREEN_RED",
    "r",
    "g",
    "b",
)

#: Indices 1-8 are normalised differences in [-1, 1].
NORMALIZED_DIFFERENCE_INDICES = INDEX_NAMES[:8]
#: Indices 9-11 are relative chromaticities in [0, 1].
CHROMATICITY_INDICES = INDEX_NAMES[8:]


def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Return ``(a - b) / (a + b + eps)``, elementwise over whole arrays.

    The ``eps`` in the denominator keeps the result finite where both bands are
    zero (shadow, nodata fill), where the mathematical form is undefined.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return ((a - b) / (a + b + eps)).astype(np.float32)


def compute_indices(
    bands: np.ndarray,
    eps: float = EPS,
    names: Sequence[str] = INDEX_NAMES,
) -> np.ndarray:
    """Compute the 11 spectral indices from a band stack.

    Parameters
    ----------
    bands
        Array shaped ``(6, ...)`` ordered ``[Blue, Green, Red, NIR, SWIR1,
        SWIR2]``, as TOA reflectance. Any trailing shape is allowed, so this
        works on ``(6, H, W)`` scenes and ``(6, N)`` pixel lists alike.
    eps
        Denominator stabiliser.
    names
        Which indices to return, and in what order.

    Returns
    -------
    np.ndarray
        Float32 array shaped ``(len(names), ...)``.
    """
    bands = np.asarray(bands, dtype=np.float32)
    if bands.shape[0] != 6:
        raise ValueError(
            f"Expected 6 bands ordered {BAND_ORDER}, got first axis = {bands.shape[0]}"
        )

    computed = compute_indices_dict(bands, eps=eps)
    missing = [name for name in names if name not in computed]
    if missing:
        raise KeyError(f"Unknown spectral index/indices: {missing}")

    return np.stack([computed[name] for name in names], axis=0).astype(np.float32)


def compute_indices_dict(bands: np.ndarray, eps: float = EPS) -> dict[str, np.ndarray]:
    """Compute all 11 indices and return them keyed by name."""
    bands = np.asarray(bands, dtype=np.float32)
    blue, green, red, nir, swir1, swir2 = (bands[i] for i in range(6))

    out: dict[str, np.ndarray] = {
        "NDVI": normalized_difference(nir, red, eps),
        "NDSI": normalized_difference(green, swir1, eps),
        "NDWI_GAO": normalized_difference(nir, swir1, eps),
        "NBR": normalized_difference(nir, swir2, eps),
        "ND_SWIR1_SWIR2": normalized_difference(swir1, swir2, eps),
        "ND_BLUE_RED": normalized_difference(blue, red, eps),
        "ND_BLUE_NIR": normalized_difference(blue, nir, eps),
        "ND_GREEN_RED": normalized_difference(green, red, eps),
    }

    rgb_sum = (red + green + blue + eps).astype(np.float32)
    out["r"] = (red / rgb_sum).astype(np.float32)
    out["g"] = (green / rgb_sum).astype(np.float32)
    out["b"] = (blue / rgb_sum).astype(np.float32)
    return out


def landsat_toa_reflectance(
    dn: np.ndarray,
    mult: float,
    add: float,
    solar_zenith_deg: Optional[float] = None,
    solar_elevation_deg: Optional[float] = None,
    clip: bool = True,
) -> np.ndarray:
    """Convert Landsat DN to TOA reflectance (paper Eq. 2).

    .. math::
        \\rho_\\lambda = \\frac{M_\\rho \\cdot \\mathrm{DN} + A_\\rho}
                              {\\cos\\theta_{SZ}}

    Parameters
    ----------
    dn
        Raw digital numbers.
    mult, add
        ``REFLECTANCE_MULT_BAND_x`` and ``REFLECTANCE_ADD_BAND_x`` from the
        scene's MTL metadata.
    solar_zenith_deg
        Solar zenith angle in degrees. Give this *or* ``solar_elevation_deg``
        (Landsat MTL files record elevation; zenith = 90 - elevation).
    solar_elevation_deg
        Solar elevation angle in degrees.
    clip
        Clip the result to [0, 1], as the paper specifies.
    """
    if solar_zenith_deg is None and solar_elevation_deg is None:
        raise ValueError("Provide either solar_zenith_deg or solar_elevation_deg.")
    if solar_zenith_deg is None:
        solar_zenith_deg = 90.0 - float(solar_elevation_deg)

    cos_sz = float(np.cos(np.deg2rad(solar_zenith_deg)))
    if cos_sz <= 0:
        raise ValueError(
            f"Non-positive cos(solar zenith) ({cos_sz:.4f}); the sun is at or "
            f"below the horizon for solar_zenith_deg={solar_zenith_deg}."
        )

    rho = (np.asarray(dn, dtype=np.float32) * np.float32(mult) + np.float32(add)) / np.float32(cos_sz)
    return np.clip(rho, 0.0, 1.0).astype(np.float32) if clip else rho.astype(np.float32)


def sentinel2_toa_reflectance(
    dn: np.ndarray, scale: float = 10000.0, clip: bool = True
) -> np.ndarray:
    """Convert Sentinel-2 L1C DN to TOA reflectance: ``rho = DN / 10000``."""
    rho = np.asarray(dn, dtype=np.float32) / np.float32(scale)
    return np.clip(rho, 0.0, 1.0).astype(np.float32) if clip else rho.astype(np.float32)


def brovey_pansharpen(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    pan: np.ndarray,
    eps: float = EPS,
    clip: bool = True,
) -> np.ndarray:
    """Brovey pan-sharpening of an RGB composite (paper Eq. 3).

    .. math::
        B_k^{15} = B_k^{30} \\cdot
                   \\frac{\\mathrm{PAN}_{15}}{\\sum_j B_j^{30} + \\varepsilon}

    The denominator sums only the three visible bands, per the standard Brovey
    formulation stated in the paper. Every input must already be on the same
    (15 m) grid: upsample the 30 m bands before calling this.

    Applies to Landsat 7/8/9, which carry a 15 m panchromatic band. Landsat 5
    and Sentinel-2 have none.

    Returns
    -------
    np.ndarray
        Stack shaped ``(3, ...)`` ordered ``[R, G, B]``, clipped to [0, 1].

    Notes
    -----
    This sharpens the RGB composites shown to the annotator. It deliberately
    does **not** feed :func:`compute_indices`, which the paper specifies are
    computed from the 30 m bands.
    """
    red = np.asarray(red, dtype=np.float32)
    green = np.asarray(green, dtype=np.float32)
    blue = np.asarray(blue, dtype=np.float32)
    pan = np.asarray(pan, dtype=np.float32)

    if not (red.shape == green.shape == blue.shape == pan.shape):
        raise ValueError(
            f"All bands must share a shape; got R{red.shape} G{green.shape} "
            f"B{blue.shape} PAN{pan.shape}. Resample to the 15 m grid first."
        )

    ratio = pan / (red + green + blue + eps)
    out = np.stack([red * ratio, green * ratio, blue * ratio], axis=0)
    return np.clip(out, 0.0, 1.0).astype(np.float32) if clip else out.astype(np.float32)


def bands_from_mapping(
    bands: Mapping[str, np.ndarray], order: Sequence[str] = BAND_ORDER
) -> np.ndarray:
    """Stack a ``{band_name: array}`` mapping into the canonical 6-band order."""
    missing = [name for name in order if name not in bands]
    if missing:
        raise KeyError(f"Missing band(s) {missing}; expected all of {tuple(order)}")
    return np.stack([np.asarray(bands[name], dtype=np.float32) for name in order], axis=0)


def valid_pixel_mask(bands: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels where every band is finite."""
    return np.all(np.isfinite(np.asarray(bands, dtype=np.float32)), axis=0)
