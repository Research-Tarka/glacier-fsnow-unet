"""glacier-fsnow-unet: U-Net glacier surface segmentation and snow-fraction pipeline.

Import ordering workaround (Windows)
-------------------------------------
Several modules in this package (``dem.engine``, ``dem.sources``,
``glaciers.registry``, ``inference.coastline``) import both a
GDAL/rasterio-family package and, indirectly through pandas, pyarrow's parquet
engine. On Windows, conda-forge's GDAL/rasterio and pip's pyarrow wheel both
ship their own copies of several shared DLLs; whichever loads first wins the
process-wide DLL search order. If rasterio loads first, pyarrow's own native
extensions can fail later with an ``ImportError`` ("DLL load failed") the
first time any code calls ``pandas.DataFrame.to_parquet``/``read_parquet`` --
even though pyarrow imports fine on its own, and even though nothing in the
failing call touches rasterio at all. Importing pyarrow's parquet engine here,
before this package's own submodules get a chance to import rasterio, fixes
the DLL search order for every entry point (``scripts/*.py``, tests) that
imports ``glacier_fsnow_unet`` first, regardless of which of its submodules
happens to import rasterio.
"""

try:
    import pyarrow.dataset  # noqa: F401
    import pyarrow.parquet  # noqa: F401
except ImportError:
    # pyarrow is a documented dependency (environment.yml); if it is missing
    # entirely, parquet I/O will fail later with its own clear error, which is
    # more informative here than masking the absence.
    pass

__version__ = "0.1.0"
