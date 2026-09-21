"""Visible Glacier Surface (VGS) reference mask.

Purpose
-------
Select, for each glacier, the reference mask of its visible surface -- the
spatial support over which F_snow is computed for every year. Implements
Section 4.7 of the paper.

Definition (paper, verbatim criteria)
-------------------------------------
"For each glacier, a reference VGS mask is identified from the **earliest year**
satisfying four simultaneous criteria:

1. the primary 8-connected component is the largest component intersecting the
   RGI polygon;
2. Snow+Ice coverage within the RGI polygon >= 70%;
3. primary component area <= 1.5 x S_RGI (excludes years where transient,
   out-of-place fresh snow cover or connections to neighbouring glaciers
   inflate the primary component beyond the glacier's true extent);
4. cloud fraction <= 10%.

The mask excludes pixels from adjacent RGI polygons, applies morphological
closing to fill internal gaps, and includes all pixels within the reference RGI
boundary."

Inputs
------
- Per-year annual worst-state composites ``(H, W)`` of class codes.
- The glacier's rasterized RGI polygon mask, and optionally a mask of
  neighbouring RGI polygons to exclude.

Outputs
-------
- A boolean VGS mask, the year it came from, and the per-year diagnostics
  explaining why each candidate year passed or failed.

Vectorization
-------------
Connected-component labelling uses ``scipy.ndimage.label`` with an 8-connected
structuring element; every subsequent test is a whole-array boolean reduction.
There are no Python loops over pixels -- the only loop is over candidate years,
which is inherently sequential because the paper asks for the *earliest*
qualifying year.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

import numpy as np

from .classes import CLOUD, ICE, NODATA, OTHER, SNOW

#: Minimum Snow+Ice coverage of the RGI polygon (criterion 2), as a fraction.
DEFAULT_MIN_SNOW_ICE_COVERAGE = 0.70
#: Maximum primary-component area as a multiple of the RGI area (criterion 3).
DEFAULT_MAX_AREA_RATIO = 1.5
#: Maximum cloud fraction (criterion 4), as a fraction.
DEFAULT_MAX_CLOUD_FRACTION = 0.10
#: Radius, in pixels, of the morphological closing that fills internal gaps.
DEFAULT_CLOSING_RADIUS = 1

#: 8-connectivity structuring element for connected-component labelling.
_STRUCT_8 = np.ones((3, 3), dtype=bool)


@dataclass
class VgsYearDiagnostics:
    """Why one candidate year passed or failed the four VGS criteria."""

    year: int
    snow_ice_coverage: float = 0.0
    area_ratio: float = 0.0
    cloud_fraction: float = 1.0
    primary_component_px: int = 0
    rgi_px: int = 0
    has_primary_component: bool = False
    passes_coverage: bool = False
    passes_area_ratio: bool = False
    passes_cloud: bool = False

    @property
    def passes(self) -> bool:
        """True when all four criteria hold simultaneously."""
        return (
            self.has_primary_component
            and self.passes_coverage
            and self.passes_area_ratio
            and self.passes_cloud
        )

    def reason(self) -> str:
        """Short human-readable explanation of the outcome."""
        if self.passes:
            return "ok"
        failures = []
        if not self.has_primary_component:
            failures.append("no Snow+Ice component intersecting the RGI polygon")
        if not self.passes_coverage:
            failures.append(f"Snow+Ice coverage {self.snow_ice_coverage:.1%} below threshold")
        if not self.passes_area_ratio:
            failures.append(f"primary component {self.area_ratio:.2f}x the RGI area")
        if not self.passes_cloud:
            failures.append(f"cloud fraction {self.cloud_fraction:.1%} above threshold")
        return "; ".join(failures)


@dataclass
class VgsResult:
    """The selected VGS mask and the trail of how it was chosen."""

    mask: Optional[np.ndarray]
    year: Optional[int]
    diagnostics: list[VgsYearDiagnostics] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.mask is not None

    @property
    def area_px(self) -> int:
        return int(self.mask.sum()) if self.mask is not None else 0


def primary_component(
    snow_ice: np.ndarray,
    rgi_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return the largest 8-connected Snow+Ice component intersecting the RGI polygon.

    Criterion 1. Components are labelled over the whole Snow+Ice field (not just
    inside the polygon), so a glacier extending slightly beyond its RGI outline
    is kept whole; only components that touch the polygon are considered.

    Returns ``(component_mask, size_px)``; an all-False mask and 0 when no
    component intersects the polygon.
    """
    from scipy import ndimage

    snow_ice = np.asarray(snow_ice, dtype=bool)
    rgi_mask = np.asarray(rgi_mask, dtype=bool)

    if not snow_ice.any():
        return np.zeros_like(snow_ice), 0

    labels, n_labels = ndimage.label(snow_ice, structure=_STRUCT_8)
    if n_labels == 0:
        return np.zeros_like(snow_ice), 0

    # Labels present inside the RGI polygon (vectorized; excludes background 0).
    inside_labels = np.unique(labels[rgi_mask])
    inside_labels = inside_labels[inside_labels > 0]
    if inside_labels.size == 0:
        return np.zeros_like(snow_ice), 0

    # Size of every label in one pass, then pick the biggest intersecting one.
    sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
    best_label = int(inside_labels[np.argmax(sizes[inside_labels])])
    component = labels == best_label
    return component, int(sizes[best_label])


def evaluate_year(
    state_map: np.ndarray,
    rgi_mask: np.ndarray,
    year: int,
    min_snow_ice_coverage: float = DEFAULT_MIN_SNOW_ICE_COVERAGE,
    max_area_ratio: float = DEFAULT_MAX_AREA_RATIO,
    max_cloud_fraction: float = DEFAULT_MAX_CLOUD_FRACTION,
) -> tuple[VgsYearDiagnostics, np.ndarray]:
    """Test one year's annual composite against the four VGS criteria.

    Returns ``(diagnostics, primary_component_mask)``.
    """
    state = np.asarray(state_map, dtype=np.uint8)
    rgi = np.asarray(rgi_mask, dtype=bool)

    diag = VgsYearDiagnostics(year=year, rgi_px=int(rgi.sum()))
    if diag.rgi_px == 0:
        return diag, np.zeros_like(rgi)

    snow_ice = (state == SNOW) | (state == ICE)

    # Criterion 1: the primary 8-connected component.
    component, component_px = primary_component(snow_ice, rgi)
    diag.primary_component_px = component_px
    diag.has_primary_component = component_px > 0

    # Criterion 2: Snow+Ice coverage of the RGI polygon.
    diag.snow_ice_coverage = float((snow_ice & rgi).sum()) / diag.rgi_px
    diag.passes_coverage = diag.snow_ice_coverage >= min_snow_ice_coverage

    # Criterion 3: the component must not balloon past 1.5x the RGI area.
    diag.area_ratio = component_px / diag.rgi_px
    diag.passes_area_ratio = diag.area_ratio <= max_area_ratio

    # Criterion 4: cloud fraction inside the RGI polygon.
    observed = rgi & (state != NODATA)
    n_observed = int(observed.sum())
    diag.cloud_fraction = (
        float(((state == CLOUD) & observed).sum()) / n_observed if n_observed else 1.0
    )
    diag.passes_cloud = diag.cloud_fraction <= max_cloud_fraction

    return diag, component


def build_vgs_mask(
    component: np.ndarray,
    rgi_mask: np.ndarray,
    neighbour_mask: Optional[np.ndarray] = None,
    closing_radius: int = DEFAULT_CLOSING_RADIUS,
) -> np.ndarray:
    """Assemble the final VGS mask from a qualifying year's primary component.

    Per the paper, the mask:

    * takes the primary component **union** every pixel inside the reference RGI
      boundary ("includes all pixels within the reference RGI boundary");
    * **excludes** pixels belonging to adjacent RGI polygons;
    * has **morphological closing** applied to fill internal gaps.

    Closing is applied before the neighbour exclusion, so dilation cannot pull
    a neighbouring glacier's pixels back in.
    """
    from scipy import ndimage

    component = np.asarray(component, dtype=bool)
    rgi = np.asarray(rgi_mask, dtype=bool)

    mask = component | rgi

    if closing_radius and closing_radius > 0:
        size = 2 * int(closing_radius) + 1
        mask = ndimage.binary_closing(mask, structure=np.ones((size, size), dtype=bool))

    if neighbour_mask is not None:
        neighbours = np.asarray(neighbour_mask, dtype=bool)
        # A pixel inside this glacier's own polygon always belongs to it, even
        # where polygons are recorded as touching.
        mask &= ~(neighbours & ~rgi)

    return mask.astype(bool)


def select_vgs(
    annual_states: Mapping[int, np.ndarray] | Iterable[tuple[int, np.ndarray]],
    rgi_mask: np.ndarray,
    neighbour_mask: Optional[np.ndarray] = None,
    min_snow_ice_coverage: float = DEFAULT_MIN_SNOW_ICE_COVERAGE,
    max_area_ratio: float = DEFAULT_MAX_AREA_RATIO,
    max_cloud_fraction: float = DEFAULT_MAX_CLOUD_FRACTION,
    closing_radius: int = DEFAULT_CLOSING_RADIUS,
) -> VgsResult:
    """Select the VGS mask from the **earliest** year meeting all four criteria.

    Parameters
    ----------
    annual_states
        ``{year: annual_state_map}``, or an iterable of ``(year, map)`` pairs.
    rgi_mask
        Rasterized RGI polygon of this glacier.
    neighbour_mask
        Rasterized union of adjacent RGI polygons, excluded from the mask.

    Returns
    -------
    VgsResult
        With ``found=False`` and full per-year diagnostics if no year qualifies.
    """
    items = (
        sorted(annual_states.items())
        if isinstance(annual_states, Mapping)
        else sorted(annual_states, key=lambda kv: kv[0])
    )

    diagnostics: list[VgsYearDiagnostics] = []
    for year, state_map in items:
        diag, component = evaluate_year(
            state_map,
            rgi_mask,
            year,
            min_snow_ice_coverage=min_snow_ice_coverage,
            max_area_ratio=max_area_ratio,
            max_cloud_fraction=max_cloud_fraction,
        )
        diagnostics.append(diag)

        if diag.passes:
            mask = build_vgs_mask(
                component, rgi_mask, neighbour_mask, closing_radius=closing_radius
            )
            return VgsResult(mask=mask, year=year, diagnostics=diagnostics)

    return VgsResult(mask=None, year=None, diagnostics=diagnostics)
