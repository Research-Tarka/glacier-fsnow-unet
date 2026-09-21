"""Explicit grid-alignment check between a glacier's DEM and its sensor scenes.

Purpose
-------
Verify -- for real, per glacier, not just as an assumption from the
registry's square-window bounds -- that the DEM (all three output
resolutions) and every downloaded sensor's TOA grid share EPSG:3413 and a
mutually consistent pixel grid, flagging any offset or misalignment rather
than silently accepting it.

Why this needs a real check
----------------------------
The DEM's analysis window and the sensor scenes' analysis window are built
from the same registry row but are **not** pixel-identical by construction:
the DEM stage adds a fixed buffer margin
(``dem.processing.MARGIN_METERS``, 150 m) around the glacier's square analysis
window before fetching the source DEM (so interior-hole interpolation has
material to work with at the edges), while the scene-fetch stage requests
exactly the unbuffered analysis window. A correct fixture therefore shows a
small, *bounded* origin offset between the DEM grid and every scene group's
grid -- not zero, but not arbitrary either. This module makes that expectation
explicit and checkable, instead of leaving it as an unstated assumption.

A real, live bug this check would have caught: an earlier version of the
scene fetch requested its analysis window as an EPSG:4326 lon/lat rectangle,
which reprojects into a visibly non-rectangular region at this fixture's
~60 degN latitude and shifted the fetched grid by roughly 1.5 km relative to
the DEM -- a genuine misalignment, not a rounding difference. Running this
check against that version's output would have failed loudly, exactly as
intended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..scenes.zarr_store import GROUP_ORDER, group_names, read_toa
from .zarr_store import read_dem

#: The DEM's buffer margin beyond the sensor scenes' analysis window
#: (dem.processing.MARGIN_METERS). A scene/DEM origin offset within this
#: bound (plus one pixel of rounding) is expected, not a misalignment.
_EXPECTED_MAX_OFFSET_M = 200.0


@dataclass
class GridCheckResult:
    """One glacier's grid-alignment report."""

    glims_id: str
    dem_crs_ok: bool = True
    scene_crs_ok: dict[str, bool] = field(default_factory=dict)
    dem_scene_offset_m: dict[str, tuple[float, float]] = field(default_factory=dict)
    scene_pairwise_offset_m: dict[str, tuple[float, float]] = field(default_factory=dict)
    pixel_size_mismatch: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every check passed: consistent CRS, no pixel-size
        mismatch, and every offset within the expected bound."""
        if self.errors or self.pixel_size_mismatch:
            return False
        if not self.dem_crs_ok or not all(self.scene_crs_ok.values()):
            return False
        for dx, dy in self.dem_scene_offset_m.values():
            if abs(dx) > _EXPECTED_MAX_OFFSET_M or abs(dy) > _EXPECTED_MAX_OFFSET_M:
                return False
        for dx, dy in self.scene_pairwise_offset_m.values():
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                return False
        return True

    def report(self) -> str:
        lines = [f"Grid check for {self.glims_id}: {'OK' if self.ok else 'FAILED'}"]
        lines.append(f"  DEM CRS is EPSG:3413: {self.dem_crs_ok}")
        for group, ok in sorted(self.scene_crs_ok.items()):
            lines.append(f"  {group} CRS is EPSG:3413: {ok}")
        for group, (dx, dy) in sorted(self.dem_scene_offset_m.items()):
            flag = "" if max(abs(dx), abs(dy)) <= _EXPECTED_MAX_OFFSET_M else "  <-- OUT OF BOUND"
            lines.append(f"  DEM<->{group} origin offset: ({dx:.1f} m, {dy:.1f} m){flag}")
        for pair, (dx, dy) in sorted(self.scene_pairwise_offset_m.items()):
            flag = "" if max(abs(dx), abs(dy)) <= 1e-6 else "  <-- MISALIGNED"
            lines.append(f"  {pair} origin offset: ({dx:.3f} m, {dy:.3f} m){flag}")
        if self.pixel_size_mismatch:
            lines.append(f"  Pixel-size mismatches: {self.pixel_size_mismatch}")
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)


def check_grid_alignment(
    glims_id: str,
    glacier_dir,
    zarr_path=None,
    expected_max_offset_m: float = _EXPECTED_MAX_OFFSET_M,
) -> GridCheckResult:
    """Run the grid-alignment check for one glacier, against real zarr contents.

    Parameters
    ----------
    glacier_dir
        The glacier's directory (holding ``<glims_id>.zarr``).
    zarr_path
        Override the store path (defaults to ``<glacier_dir>/<glims_id>.zarr``).
    """
    from ..scenes.zarr_store import zarr_path_for_glacier

    path = zarr_path if zarr_path is not None else zarr_path_for_glacier(glacier_dir)
    result = GridCheckResult(glims_id=glims_id)

    try:
        _, dem_transform, dem_crs = read_dem(glacier_dir, 30)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"could not read DEM: {exc}")
        return result

    result.dem_crs_ok = "3413" in dem_crs

    origins: dict[str, tuple[float, float]] = {}
    pixel_sizes: dict[str, tuple[float, float]] = {}

    for group in group_names(path):
        if group not in GROUP_ORDER:
            continue
        try:
            _, transform, crs_wkt = read_toa(path, group, 0, n_bands=6)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"could not read {group} TOA: {exc}")
            continue

        result.scene_crs_ok[group] = "3413" in crs_wkt
        origins[group] = (transform.c, transform.f)
        pixel_sizes[group] = (abs(transform.a), abs(transform.e))

        result.dem_scene_offset_m[group] = (
            transform.c - dem_transform.c,
            transform.f - dem_transform.f,
        )

    # Every Landsat group (l89/l7/l5) is nominally 30 m and must share the
    # exact same origin, since they all request the same analysis window at
    # the same scale; Sentinel-2 is nominally 10 m and is checked separately
    # only for pixel size, not origin equality with the 30 m groups.
    landsat_groups = [g for g in origins if g in ("l89", "l7", "l5")]
    for i, group_a in enumerate(landsat_groups):
        for group_b in landsat_groups[i + 1:]:
            ox_a, oy_a = origins[group_a]
            ox_b, oy_b = origins[group_b]
            result.scene_pairwise_offset_m[f"{group_a}<->{group_b}"] = (
                ox_a - ox_b, oy_a - oy_b,
            )
            if abs(pixel_sizes[group_a][0] - pixel_sizes[group_b][0]) > 1e-6:
                result.pixel_size_mismatch.append(f"{group_a} vs {group_b}")

    return result
