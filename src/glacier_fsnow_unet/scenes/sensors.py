"""Sensor specifications for the Google Earth Engine scene download.

Purpose
-------
One declarative table describing the five sensors of the paper (Table 1),
so per-sensor constants (collection id, bands, year range) live in a single
place rather than being restated inline once per sensor-specific module.

Inputs / outputs
----------------
Pure data plus small helpers; no I/O and no Earth Engine calls.

Sensors (paper Table 1)
-----------------------
==============  ==========  ======  ====
Sensor          Period      Res.    PAN
==============  ==========  ======  ====
Landsat 5 TM    1984-2011   30 m    no
Landsat 7 ETM+  1999-2002   30 m    15 m
Landsat 8 OLI   2013-       30 m    15 m
Landsat 9 OLI-2 2021-       30 m    15 m
Sentinel-2 MSI  2015-       10 m    no
==============  ==========  ======  ====

Landsat 7 note
--------------
The paper restricts Landsat 7 to **SLC-on only (1999-2002)**: after the 2003
Scan Line Corrector failure, every L7 scene has striping data gaps unsuitable
for pixel-level classification. This encodes the paper's range directly in
:data:`SENSORS`, and additionally keeps the SLC-off years as an explicit
second guard (:func:`is_year_allowed`) so that period cannot be reintroduced
by overriding the year range on the command line.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

#: Ablation-season window applied to every sensor (paper Section 3.2).
SEASON_START_MMDD = "07-01"
SEASON_END_MMDD = "09-30"

#: Landsat 7 SLC failure: scenes from 2003 onwards have striping gaps.
SLC_OFF_FIRST_YEAR = 2003
#: Last year of the SLC-off guard window; the paper's own L7 range simply ends
#: at 2002, so this guard is never reached when using the paper's range, only
#: when a caller overrides the year range on the command line.
SLC_OFF_LAST_YEAR = 2012


@dataclass(frozen=True)
class SensorSpec:
    """Everything the downloader needs to know about one sensor."""

    key: str
    name: str
    collection: str
    #: GEE band ids in the pipeline's canonical order
    #: [Blue, Green, Red, NIR, SWIR1, SWIR2].
    bands: tuple[str, ...]
    first_year: int
    last_year: Optional[int]  # None -> up to the current year
    native_resolution_m: float
    pan_band: Optional[str] = None
    pan_resolution_m: Optional[float] = None
    #: True for Sentinel-2 L1C (DN/10000); False for Landsat (Eq. 2 + MTL gains).
    is_sentinel: bool = False

    @property
    def has_pan(self) -> bool:
        return self.pan_band is not None

    def years(self, until: Optional[int] = None) -> list[int]:
        """Inclusive list of acquisition years for this sensor."""
        end = self.last_year if self.last_year is not None else (until or date.today().year)
        if until is not None:
            end = min(end, until)
        return list(range(self.first_year, end + 1))


#: Landsat Collection 2 Tier 1 TOA, and Sentinel-2 L1C (harmonized).
SENSORS: dict[str, SensorSpec] = {
    "L5": SensorSpec(
        key="L5",
        name="Landsat 5 TM",
        collection="LANDSAT/LT05/C02/T1_TOA",
        bands=("B1", "B2", "B3", "B4", "B5", "B7"),
        first_year=1984,
        last_year=2011,
        native_resolution_m=30.0,
    ),
    "L7": SensorSpec(
        key="L7",
        name="Landsat 7 ETM+",
        collection="LANDSAT/LE07/C02/T1_TOA",
        bands=("B1", "B2", "B3", "B4", "B5", "B7"),
        first_year=1999,
        last_year=2002,  # SLC-on only (paper Table 1)
        native_resolution_m=30.0,
        pan_band="B8",
        pan_resolution_m=15.0,
    ),
    "L8": SensorSpec(
        key="L8",
        name="Landsat 8 OLI",
        collection="LANDSAT/LC08/C02/T1_TOA",
        bands=("B2", "B3", "B4", "B5", "B6", "B7"),
        first_year=2013,
        last_year=None,
        native_resolution_m=30.0,
        pan_band="B8",
        pan_resolution_m=15.0,
    ),
    "L9": SensorSpec(
        key="L9",
        name="Landsat 9 OLI-2",
        collection="LANDSAT/LC09/C02/T1_TOA",
        bands=("B2", "B3", "B4", "B5", "B6", "B7"),
        first_year=2021,
        last_year=None,
        native_resolution_m=30.0,
        pan_band="B8",
        pan_resolution_m=15.0,
    ),
    "S2": SensorSpec(
        key="S2",
        name="Sentinel-2 MSI",
        collection="COPERNICUS/S2_HARMONIZED",
        bands=("B2", "B3", "B4", "B8", "B11", "B12"),
        first_year=2015,
        last_year=None,
        native_resolution_m=10.0,
        is_sentinel=True,
    ),
}

#: Processing order, oldest sensor first, so a resumed download processes
#: sensors in the same predictable order every time.
SENSOR_ORDER: tuple[str, ...] = ("L5", "L7", "L8", "L9", "S2")


def get_sensor(key: str) -> SensorSpec:
    """Look up a sensor spec by key, with a helpful error listing valid keys."""
    try:
        return SENSORS[key.upper()]
    except KeyError:
        raise KeyError(
            f"Unknown sensor '{key}'. Valid sensors: {', '.join(SENSOR_ORDER)}"
        ) from None


def is_year_allowed(sensor_key: str, year: int, skip_landsat7_slc_off: bool = True) -> bool:
    """Whether ``year`` should be downloaded for ``sensor_key``.

    Guards the Landsat 7 SLC-off period explicitly, so it stays excluded even
    if a caller widens the year range.
    """
    spec = get_sensor(sensor_key)
    if year < spec.first_year:
        return False
    if spec.last_year is not None and year > spec.last_year:
        return False
    if (
        skip_landsat7_slc_off
        and spec.key == "L7"
        and SLC_OFF_FIRST_YEAR <= year <= SLC_OFF_LAST_YEAR
    ):
        return False
    return True


def season_bounds(year: int) -> tuple[str, str]:
    """Return the ``(start, end)`` ISO dates of the ablation window for a year.

    The end date is exclusive, as Earth Engine's ``filterDate`` expects, so
    30 September is included.
    """
    return f"{year}-{SEASON_START_MMDD}", f"{year}-10-01"


def sensors_for_years(
    years: Sequence[int], skip_landsat7_slc_off: bool = True
) -> dict[str, list[int]]:
    """Map each sensor to the subset of ``years`` it can actually provide."""
    return {
        key: [y for y in years if is_year_allowed(key, y, skip_landsat7_slc_off)]
        for key in SENSOR_ORDER
    }
