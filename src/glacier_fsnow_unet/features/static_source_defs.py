"""Declarative definitions of the static (non-temporal) feature sources.

Purpose
-------
Complement ``source_defs.py``'s five per-year GEE sources with four sources
that describe one fixed value per glacier rather than a time series: RGI's
own structural glacier attributes, Koppen-Geiger climate classification,
WorldClim's 1970-2000 climatology, and circum-Arctic permafrost extent. Each
is fetched once (an HTTP/FTP download for three of them, a local file read
for RGI) and sampled at every glacier's centroid.

Scope
-----
Only these four static sources are in scope, the same way ROADMAP decision
#4 keeps only 5 of ~30 candidate temporal sources: a deliberately curated
set with a clear physical rationale (glacier connectivity/surge behaviour,
climate classification, climatology normals, permafrost extent), not every
static layer technically available. Other candidates (soil properties,
alternative climatologies, lithology, lake proximity) are left out for the
same reason.

Inputs / outputs
-----------------
Pure data (specs) plus the fetch/build functions each spec points at; the
actual network and filesystem I/O lives in ``remote_fetch.py`` and the point
sampling in ``static_sampling.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from .remote_fetch import fetch_and_extract, first_match
from .static_sampling import sample_rasters_at_centroids, sample_vector_attributes_at_centroids

Builder = Callable[[pd.DataFrame, Path], pd.DataFrame]


@dataclass(frozen=True)
class StaticSourceSpec:
    """Everything needed to fetch and sample one static source.

    ``builder`` always receives ``(glaciers, cache_dir)`` and is responsible
    for fetching ``urls`` into ``cache_dir`` (via ``fetch_and_extract``) and
    sampling the result at each glacier's centroid.
    """

    slug: str
    description: str
    builder: Builder
    #: Root the source's raw downloads/extracted files are cached under
    #: (relative to a configured static-sources cache directory).
    cache_subdir: str
    #: Remote URLs to fetch.
    urls: tuple[str, ...] = ()


KOPPEN_GEIGER_CLASS_MAP: dict[int, str] = {
    1: "Af", 2: "Am", 3: "Aw", 4: "BWh", 5: "BWk", 6: "BSh", 7: "BSk",
    8: "Csa", 9: "Csb", 10: "Csc", 11: "Cwa", 12: "Cwb", 13: "Cwc",
    14: "Cfa", 15: "Cfb", 16: "Cfc", 17: "Dsa", 18: "Dsb", 19: "Dsc",
    20: "Dsd", 21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd", 25: "Dfa",
    26: "Dfb", 27: "Dfc", 28: "Dfd", 29: "ET", 30: "EF",
}
KOPPEN_GEIGER_GROUP_MAP = {"A": "tropical", "B": "arid", "C": "temperate", "D": "cold", "E": "polar"}


#: Structural attribute columns pulled from each RGI region's own
#: ``*-attributes.csv`` (already shipped alongside its geometry -- no
#: download needed, unlike the other three static sources).
_RGI_ATTRIBUTE_COLUMNS = ("rgi_id", "glims_id", "primeclass", "conn_lvl", "surge_type", "term_type")


def build_rgi_structural(glaciers: pd.DataFrame, region_roots: Sequence[Path]) -> pd.DataFrame:
    """Read RGI 7.0's own per-glacier structural attributes CSV.

    ``region_roots`` are local RGI region folders (one per RGI O1 region,
    e.g. ``RGI2000-v7.0-G-01_alaska``); each already ships an
    ``*-attributes.csv`` next to its geometry, so this is a local read, not a
    network fetch. Exposed as a public function (unlike the other three
    builders) because it takes region roots rather than a cache directory --
    ``run_static_source`` in ``scripts/13_download_static_sources.py`` calls
    it directly instead of through the ``StaticSourceSpec.builder`` slot.
    """
    frames = []
    for root in region_roots:
        csv_path = first_match(root, "*-attributes.csv")
        if csv_path is None:
            raise FileNotFoundError(f"rgi_structural_static: no attributes CSV found under {root}")
        frames.append(pd.read_csv(csv_path, usecols=list(_RGI_ATTRIBUTE_COLUMNS)))
    if not frames:
        raise FileNotFoundError("rgi_structural_static: no RGI region roots configured.")

    combined = pd.concat(frames, ignore_index=True)
    combined["glims_id"] = combined["glims_id"].astype(str)
    target_ids = set(glaciers["glims_id"].astype(str))
    out = combined.loc[combined["glims_id"].isin(target_ids)].copy()
    out = out.drop_duplicates(subset=["glims_id"], keep="first").sort_values("glims_id").reset_index(drop=True)
    return out


#: Beck et al. 2018/2023 Koppen-Geiger maps, Figshare article 21789074 --
#: verified live against the Figshare API (file ids are per-upload, not
#: stable across article revisions, so this is checked, not guessed).
#: ``koppen_geiger_tif.zip`` contains one present-day raster plus several
#: future-projection rasters; only the present-day one is sampled.
_KOPPEN_GEIGER_ZIP_URL = "https://ndownloader.figshare.com/files/61012822"


def _build_koppen_geiger(glaciers: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    extract_dir = fetch_and_extract((_KOPPEN_GEIGER_ZIP_URL,), cache_dir, force=False)
    raster = first_match(
        extract_dir,
        "*1901_1930*.tif", "*present*.tif", "*koppen_geiger_0p00833333.tif", "*.tif",
    )
    if raster is None:
        raise FileNotFoundError(f"koppen_geiger_static: no raster found under {extract_dir}")
    out = sample_rasters_at_centroids(glaciers, {"koppen_class": raster})
    codes = pd.to_numeric(out["koppen_class"], errors="coerce").round().astype("Int64")
    zones = codes.map(KOPPEN_GEIGER_CLASS_MAP)
    out["koppen_class"] = codes
    out["koppen_geiger_zone"] = zones.astype("string")
    out["koppen_geiger_group"] = zones.str.slice(0, 1).map(KOPPEN_GEIGER_GROUP_MAP).astype("string")
    return out


def _build_worldclim(glaciers: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    base = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"
    urls = (f"{base}/wc2.1_2.5m_bio.zip",)
    extract_dir = fetch_and_extract(urls, cache_dir, force=False)
    patterns = {f"worldclim_bio{i}": f"*bio_{i}.tif" for i in range(1, 20)}
    variable_to_raster = {}
    for name, pattern in patterns.items():
        match = first_match(extract_dir, pattern)
        if match is not None:
            variable_to_raster[name] = match
    if not variable_to_raster:
        raise FileNotFoundError(f"worldclim_static: no bioclim rasters found under {extract_dir}")
    return sample_rasters_at_centroids(glaciers, variable_to_raster)


#: The circum-Arctic permafrost/ground-ice shapefile (Brown et al. 1997,
#: NSIDC ggd318) is published as loose shapefile components on this FTP
#: directory, not a single archive -- verified live via an FTP listing, not
#: guessed. Every ``permaice.*`` sidecar is needed for ``geopandas`` to read
#: the shapefile.
_PERMAFROST_BASE_URL = "ftp://sidads.colorado.edu/pub/DATASETS/fgdc/ggd318_map_circumarctic"
_PERMAFROST_FILES = ("permaice.shp", "permaice.shx", "permaice.dbf", "permaice.prj", "permaice.avl")


#: Brown et al. 1997/1998's ``COMBO`` field packs extent + terrain origin
#: into one string, first letter only relevant here (verified against the
#: shapefile's own attribute table: C/D/S/I are real permafrost-extent
#: classes; g/l/o/r are non-permafrost cover -- glacier, lake, ocean, relict
#: permafrost -- so they map to "not permafrost", not to a missing value).
_PERMAFROST_EXTENT_MAP = {
    "C": "continuous",  # 90-100%
    "D": "discontinuous",  # 50-90%
    "S": "sporadic",  # 10-50%
    "I": "isolated",  # 0-10%
}
_PERMAFROST_NON_PERMAFROST_COVER = {"g": "glacier", "l": "lake", "o": "ocean", "r": "relict_permafrost"}
#: ``CONTENT`` is the ground-ice content class, coded directly (l/m/h).
_PERMAFROST_ICE_CONTENT_MAP = {"l": "low", "m": "medium", "h": "high"}


def _decode_permafrost_combo(combo: object) -> tuple[object, object]:
    text = str(combo).strip() if pd.notna(combo) else ""
    if not text:
        return pd.NA, pd.NA
    first = text[0]
    if first in _PERMAFROST_EXTENT_MAP:
        return first, _PERMAFROST_EXTENT_MAP[first]
    if first in _PERMAFROST_NON_PERMAFROST_COVER:
        return "none", _PERMAFROST_NON_PERMAFROST_COVER[first]
    return pd.NA, pd.NA


def _build_permafrost(glaciers: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    urls = tuple(f"{_PERMAFROST_BASE_URL}/{name}" for name in _PERMAFROST_FILES)
    extract_dir = fetch_and_extract(urls, cache_dir, force=False)
    shp = first_match(extract_dir, "*.shp")
    if shp is None:
        raise FileNotFoundError(f"permafrost_static: no shapefile found under {extract_dir}")
    out = sample_vector_attributes_at_centroids(
        glaciers,
        shp,
        column_map={
            "permafrost_combo": ("COMBO",),
            "permafrost_content": ("CONTENT",),
        },
    )
    decoded = out["permafrost_combo"].map(_decode_permafrost_combo)
    out["permafrost_extent_code"] = decoded.map(lambda pair: pair[0])
    out["permafrost_extent_name"] = decoded.map(lambda pair: pair[1])
    out["permafrost_ground_ice_content"] = out["permafrost_content"].map(_PERMAFROST_ICE_CONTENT_MAP)
    return out.drop(columns=["permafrost_combo", "permafrost_content"])


KOPPEN_GEIGER = StaticSourceSpec(
    slug="koppen_geiger_static",
    description="Koppen-Geiger climate classification raster (Beck et al. 2018, present-day 0.1 deg).",
    builder=_build_koppen_geiger,
    cache_subdir="koppen_geiger_static",
    urls=(_KOPPEN_GEIGER_ZIP_URL,),
)

WORLDCLIM = StaticSourceSpec(
    slug="worldclim_static",
    description="WorldClim v2.1 1970-2000 bioclimatic variables (19 BIO rasters, 2.5 arc-min).",
    builder=_build_worldclim,
    cache_subdir="worldclim_static",
    urls=("https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip",),
)

PERMAFROST = StaticSourceSpec(
    slug="permafrost_static",
    description="Circum-Arctic permafrost extent and ground-ice content (IPA/NSIDC).",
    builder=_build_permafrost,
    cache_subdir="permafrost_static",
    urls=tuple(f"{_PERMAFROST_BASE_URL}/{name}" for name in _PERMAFROST_FILES),
)

#: RGI structural attributes is deliberately not in this dict: it has no
#: remote fetch and takes local RGI region roots rather than a cache
#: directory, so ``scripts/13_download_static_sources.py`` calls
#: ``build_rgi_structural`` directly instead of going through the uniform
#: ``StaticSourceSpec.builder(glaciers, cache_dir)`` interface.
STATIC_SOURCES: dict[str, StaticSourceSpec] = {
    "KoppenGeiger": KOPPEN_GEIGER,
    "WorldClim": WORLDCLIM,
    "Permafrost": PERMAFROST,
}


def get_static_source(name: str) -> StaticSourceSpec:
    try:
        return STATIC_SOURCES[name]
    except KeyError:
        raise KeyError(
            f"Unknown static feature source '{name}'. Active sources: {', '.join(STATIC_SOURCES)}"
        ) from None
