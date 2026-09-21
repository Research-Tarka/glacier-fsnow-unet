"""Declarative definitions of the five active climate/environmental sources.

Purpose
-------
Describe each source as **data** -- collection id, band-to-variable requests,
temporal windows, reducers, unit conversions -- so one generic runner can drive
them all, instead of one bespoke script per source. This keeps the Earth
Engine query logic in a single place (``gee_runner.py``) and makes adding or
retiring a source a data change, not a code change.

Scope (ROADMAP decision #4)
---------------------------
Only five sources are supported, chosen because each contributes a distinct,
non-redundant physical signal (surface energy balance, atmospheric reanalysis,
monthly climate normals, daily meteorology, aerosol optical depth) at
resolutions and latencies that suit glacier-year features over 1984-2025:

============================  ========  ==============================
Source                        Backend   Coverage
============================  ========  ==============================
ERA5-Land                     GEE       1984-2025
ERA5 Reanalysis (CDS/hourly)  GEE/CDS   1984-2025
TerraClimate                  GEE       1984-2024
Daymet V4                     GEE       1984-2024
MERRA-2 Aerosols              GEE       1984-2025
============================  ========  ==============================

Additional candidate sources exist in Earth Engine's catalog but are
deliberately **not** included here, rather than added then disabled, to keep
the feature set to sources with a clear, justified rationale.

Inputs / outputs
----------------
Pure data plus small helpers; no I/O and no Earth Engine calls.

See ``docs/decisions/climate_features.md`` for the rationale behind each
source and the unit conversions applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

Window = Literal["annual", "summer", "winter", "spring"]
Reducer = Literal["mean", "sum", "min", "max"]

#: Reference climatology window (DESIGN_FEATURES.md section 1.3).
CLIMATOLOGY_REFERENCE = (1991, 2020)

#: Target output window for the final temporal parquets (section 1.1).
TARGET_YEARS = (1984, 2025)

#: Months making up each named seasonal window. Only two seasons matter for
#: this pipeline: JAS (the ablation season the satellite scenes are already
#: restricted to, see ``scenes.sensors.season_bounds``) and DJF (the
#: accumulation season). "summer"/"winter" are kept as the internal window
#: names throughout this module (matching every delivered `_summer`/`_winter`
#: column in DESIGN_FEATURES.md) but now mean JAS/DJF, not JJA/DJF -- no
#: monthly (12-value) expansion is fetched for any source: a single
#: ``reduceRegions`` call with a `_m01..m12`-per-variable band count (as this
#: module used to build) is expensive enough server-side to intermittently
#: stall or hit Earth Engine's "User memory limit exceeded", and no `_m01`
#: .. `_m12` column is part of the delivered feature set.
SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "annual": tuple(range(1, 13)),
    "summer": (7, 8, 9),  # JAS
    "winter": (12, 1, 2),  # DJF
    "spring": (3, 4, 5),
}


@dataclass(frozen=True)
class BandRequest:
    """One output variable derived from one band over one temporal window."""

    out: str
    band: str
    window: Window
    reducer: Reducer = "mean"
    #: Multiplicative unit conversion applied after reduction.
    scale: float = 1.0
    #: Additive offset applied after scaling (e.g. -273.15 for K -> degC).
    offset: float = 0.0


@dataclass(frozen=True)
class SourceSpec:
    """Everything needed to download and aggregate one climate source."""

    slug: str
    description: str
    collection: str
    requests: tuple[BandRequest, ...]
    #: Nominal reduction scale in metres (the collection's native pixel size).
    scale_m: float
    first_year: int = 1984
    last_year: int = 2025
    backend: Literal["gee", "cds"] = "gee"
    #: Columns derived arithmetically after the per-band reduction.
    derived: tuple[str, ...] = ()

    @property
    def value_columns(self) -> tuple[str, ...]:
        """Every output column this source produces, reductions then derived."""
        return tuple(r.out for r in self.requests) + self.derived

    def years(self) -> list[int]:
        return list(range(self.first_year, self.last_year + 1))


# ---------------------------------------------------------------------------
# ERA5-Land -- principal land-surface source (temperature, snow, radiation,
# humidity, wind). GEE monthly aggregates, ~11 km.
# ---------------------------------------------------------------------------

_ERA5_LAND_REQUESTS: tuple[BandRequest, ...] = (
    BandRequest("t2m_annual", "temperature_2m", "annual", "mean", offset=-273.15),
    BandRequest("t2m_summer", "temperature_2m", "summer", "mean", offset=-273.15),
    BandRequest("t2m_winter", "temperature_2m", "winter", "mean", offset=-273.15),
    BandRequest("t2m_max_annual", "temperature_2m_max", "annual", "mean", offset=-273.15),
    BandRequest("t2m_min_annual", "temperature_2m_min", "annual", "mean", offset=-273.15),
    BandRequest("skin_temperature_summer", "skin_temperature", "summer", "mean", offset=-273.15),
    BandRequest("dewpoint_summer", "dewpoint_temperature_2m", "summer", "mean", offset=-273.15),
    BandRequest("precip_annual_mm", "total_precipitation_sum", "annual", "sum", scale=1000.0),
    BandRequest("precip_summer_mm", "total_precipitation_sum", "summer", "sum", scale=1000.0),
    BandRequest("precip_winter_mm", "total_precipitation_sum", "winter", "sum", scale=1000.0),
    BandRequest("snowfall_annual_mm", "snowfall_sum", "annual", "sum", scale=1000.0),
    BandRequest("snowfall_winter_mm", "snowfall_sum", "winter", "sum", scale=1000.0),
    BandRequest("snowmelt_annual_mm", "snowmelt_sum", "annual", "sum", scale=1000.0),
    BandRequest("snowmelt_summer_mm", "snowmelt_sum", "summer", "sum", scale=1000.0),
    BandRequest("snow_depth_we_annual", "snow_depth_water_equivalent", "annual", "mean"),
    BandRequest("snow_depth_we_winter", "snow_depth_water_equivalent", "winter", "mean"),
    BandRequest("snow_cover_annual", "snow_cover", "annual", "mean"),
    BandRequest("snow_cover_winter", "snow_cover", "winter", "mean"),
    BandRequest("snow_albedo_winter", "snow_albedo", "winter", "mean"),
    BandRequest("sw_down_summer", "surface_solar_radiation_downwards_sum", "summer", "sum"),
    BandRequest("lw_down_summer", "surface_thermal_radiation_downwards_sum", "summer", "sum"),
    BandRequest("sw_net_summer", "surface_net_solar_radiation_sum", "summer", "sum"),
    BandRequest("lw_net_summer", "surface_net_thermal_radiation_sum", "summer", "sum"),
    BandRequest("surface_pressure_annual", "surface_pressure", "annual", "mean", scale=0.01),
    BandRequest("surface_pressure_summer", "surface_pressure", "summer", "mean", scale=0.01),
    BandRequest("u10_summer", "u_component_of_wind_10m", "summer", "mean"),
    BandRequest("v10_summer", "v_component_of_wind_10m", "summer", "mean"),
    BandRequest("runoff_annual", "runoff_sum", "annual", "sum", scale=1000.0),
)

ERA5_LAND = SourceSpec(
    slug="era5_land",
    description="ERA5-Land monthly aggregates via Google Earth Engine.",
    collection="ECMWF/ERA5_LAND/MONTHLY_AGGR",
    requests=_ERA5_LAND_REQUESTS,
    scale_m=11132.0,
    first_year=1984,
    last_year=2025,
    derived=(
        "wind_speed_summer",
        "vpd_summer",
        "dewpoint_gap_summer",
        "diurnal_range_annual",
    ),
)


# ---------------------------------------------------------------------------
# ERA5 Reanalysis -- global fallback and 2025 extension.
# ---------------------------------------------------------------------------

#: Band names on ECMWF/ERA5/MONTHLY, not ECMWF/ERA5/HOURLY -- see the module
#: docstring note on this source for why.
_ERA5_REANALYSIS_REQUESTS: tuple[BandRequest, ...] = (
    BandRequest("era5_t2m_annual", "mean_2m_air_temperature", "annual", "mean", offset=-273.15),
    BandRequest("era5_t2m_summer", "mean_2m_air_temperature", "summer", "mean", offset=-273.15),
    BandRequest("era5_dewpoint_summer", "dewpoint_2m_temperature", "summer", "mean", offset=-273.15),
    BandRequest("era5_surface_pressure_annual", "surface_pressure", "annual", "mean", scale=0.01),
    BandRequest("era5_precip_annual_mm", "total_precipitation", "annual", "sum", scale=1000.0),
    BandRequest("era5_precip_summer_mm", "total_precipitation", "summer", "sum", scale=1000.0),
    BandRequest("era5_u10_summer", "u_component_of_wind_10m", "summer", "mean"),
    BandRequest("era5_v10_summer", "v_component_of_wind_10m", "summer", "mean"),
)

ERA5_REANALYSIS = SourceSpec(
    slug="era5_reanalysis_cds",
    description="ERA5 reanalysis (monthly-aggregated) via Google Earth Engine.",
    # ECMWF/ERA5/HOURLY (the collection this source originally used)
    # reproducibly fails "User memory limit exceeded" reducing a full
    # calendar year (~8760 hourly images) in
    # one call -- confirmed live, independent of band count, GEE project, or
    # transient service load (see docs/decisions/climate_features_scope_v2.md).
    # A client-side month-chunked workaround made this source correct but
    # ~90x slower (up to ~90 real network requests per year instead of 1).
    # ECMWF/ERA5/MONTHLY is Earth Engine's own pre-aggregated monthly product
    # of the same underlying ERA5 reanalysis, with matching band semantics
    # for every variable this source needs (mean/min/max 2m air temperature,
    # dewpoint, precipitation, surface pressure, 10 m wind) -- reducing 12
    # monthly images instead of ~8760 hourly ones in one call succeeds in
    # well under a second with no chunking needed, confirmed live.
    collection="ECMWF/ERA5/MONTHLY",
    requests=_ERA5_REANALYSIS_REQUESTS,
    scale_m=27830.0,
    first_year=1984,
    last_year=2025,
    derived=("era5_wind_speed_summer",),
)


# ---------------------------------------------------------------------------
# TerraClimate -- high-resolution mountain complement, ~4 km, to 2024.
# ---------------------------------------------------------------------------

_TERRACLIMATE_REQUESTS: tuple[BandRequest, ...] = (
    BandRequest("terraclimate_tmin_annual", "tmmn", "annual", "mean", scale=0.1),
    BandRequest("terraclimate_tmin_summer", "tmmn", "summer", "mean", scale=0.1),
    BandRequest("terraclimate_tmax_annual", "tmmx", "annual", "mean", scale=0.1),
    BandRequest("terraclimate_tmax_summer", "tmmx", "summer", "mean", scale=0.1),
    BandRequest("terraclimate_precip_annual_mm", "pr", "annual", "sum"),
    BandRequest("terraclimate_precip_summer_mm", "pr", "summer", "sum"),
    BandRequest("terraclimate_srad_summer", "srad", "summer", "mean"),
    BandRequest("terraclimate_vap_summer", "vap", "summer", "mean"),
    BandRequest("terraclimate_vpd_summer", "vpd", "summer", "mean"),
    BandRequest("terraclimate_ws_summer", "vs", "summer", "mean"),
    BandRequest("terraclimate_aet_annual", "aet", "annual", "sum"),
    BandRequest("terraclimate_pet_annual", "pet", "annual", "sum"),
    BandRequest("terraclimate_def_annual", "def", "annual", "sum"),
    BandRequest("terraclimate_pdsi_annual", "pdsi", "annual", "mean"),
    BandRequest("terraclimate_runoff_annual", "ro", "annual", "sum"),
    BandRequest("terraclimate_soil_annual", "soil", "annual", "mean"),
    BandRequest("terraclimate_swe_spring", "swe", "spring", "mean"),
)

TERRACLIMATE = SourceSpec(
    slug="terraclimate",
    description="TerraClimate monthly climate via Google Earth Engine.",
    collection="IDAHO_EPSCOR/TERRACLIMATE",
    requests=_TERRACLIMATE_REQUESTS,
    scale_m=4638.0,
    first_year=1984,
    last_year=2024,
    derived=("terraclimate_tavg_annual", "terraclimate_diurnal_range_annual"),
)


# ---------------------------------------------------------------------------
# Daymet V4 -- highest-resolution North American source, 1 km, to 2024.
# ---------------------------------------------------------------------------

_DAYMET_REQUESTS: tuple[BandRequest, ...] = (
    BandRequest("daymet_tmin_annual", "tmin", "annual", "mean"),
    BandRequest("daymet_tmin_summer", "tmin", "summer", "mean"),
    BandRequest("daymet_tmax_annual", "tmax", "annual", "mean"),
    BandRequest("daymet_tmax_summer", "tmax", "summer", "mean"),
    BandRequest("daymet_precip_annual_mm", "prcp", "annual", "sum"),
    BandRequest("daymet_precip_summer_mm", "prcp", "summer", "sum"),
    BandRequest("daymet_srad_summer", "srad", "summer", "mean"),
    BandRequest("daymet_swe_spring", "swe", "spring", "mean"),
    BandRequest("daymet_vp_summer", "vp", "summer", "mean"),
    BandRequest("daymet_dayl_summer", "dayl", "summer", "mean"),
)

DAYMET_V4 = SourceSpec(
    slug="daymet_v4",
    description="Daymet V4 daily surface weather via Google Earth Engine.",
    collection="NASA/ORNL/DAYMET_V4",
    requests=_DAYMET_REQUESTS,
    scale_m=1000.0,
    first_year=1984,
    last_year=2024,
    derived=("daymet_tavg_annual", "daymet_diurnal_range_annual"),
)


# ---------------------------------------------------------------------------
# MERRA-2 Aerosols -- long continuous aerosol record with speciation.
# ---------------------------------------------------------------------------

_MERRA2_AEROSOL_REQUESTS: tuple[BandRequest, ...] = (
    BandRequest("aod_total_annual", "TOTEXTTAU", "annual", "mean"),
    BandRequest("aod_total_summer", "TOTEXTTAU", "summer", "mean"),
    BandRequest("aod_bc_summer", "BCEXTTAU", "summer", "mean"),
    BandRequest("aod_dust_summer", "DUEXTTAU", "summer", "mean"),
    BandRequest("aod_sulfate_summer", "SUEXTTAU", "summer", "mean"),
    BandRequest("aod_oc_summer", "OCEXTTAU", "summer", "mean"),
)

MERRA2_AEROSOLS = SourceSpec(
    slug="merra2_aerosols",
    description="MERRA-2 aerosol optical depth via Google Earth Engine.",
    collection="NASA/GSFC/MERRA/aer/2",
    requests=_MERRA2_AEROSOL_REQUESTS,
    scale_m=50000.0,
    first_year=1984,
    last_year=2025,
)


#: The five active sources, keyed by the config's source names.
SOURCES: dict[str, SourceSpec] = {
    "ERA5_Land": ERA5_LAND,
    "ERA5_Reanalysis_CDS": ERA5_REANALYSIS,
    "TerraClimate": TERRACLIMATE,
    "Daymet_V4": DAYMET_V4,
    "MERRA2_Aerosols": MERRA2_AEROSOLS,
}

#: Source priority for gap-filling merged variables, best first.
#: DESIGN_FEATURES.md sections 5.1-5.2: "TerraClimate / Daymet when spatially
#: relevant, then ERA5-Land, then ERA5 CDS". Daymet is ranked above TerraClimate
#: because it is 1 km against TerraClimate's ~4 km and is purpose-built for
#: North America, which is the entire study domain (RGI 01 + 02).
SOURCE_PRIORITY: tuple[str, ...] = (
    "Daymet_V4",
    "TerraClimate",
    "ERA5_Land",
    "ERA5_Reanalysis_CDS",
)


def get_source(name: str) -> SourceSpec:
    """Look up a source spec by its config name."""
    try:
        return SOURCES[name]
    except KeyError:
        raise KeyError(
            f"Unknown feature source '{name}'. Active sources: {', '.join(SOURCES)}"
        ) from None
