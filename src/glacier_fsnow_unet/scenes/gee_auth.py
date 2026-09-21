"""Google Earth Engine authentication with fail-fast, actionable errors.

Purpose
-------
Initialize the Earth Engine client, and fail immediately with an instructive
message when credentials or a project id are missing -- rather than hanging,
or failing much later inside a download loop.

Inputs
------
- A GEE project id, from (in order): the explicit argument, the ``EE_PROJECT``
  environment variable / ``.env``, or ``config.scene_download.ee_project_default``.
- Earth Engine credentials, from the standard credential store created by
  ``earthengine authenticate``, or a service-account key file.

Outputs
-------
- An initialized ``ee`` module, or :class:`GeeAuthError`.

Why fail fast
-------------
Stage 4 cannot run without credentials, and a partially-authenticated client
produces confusing errors deep inside per-scene calls. :func:`initialize_ee`
therefore validates the project id up front, then performs a **round-trip call**
(``ee.Number(1).add(1).getInfo()``) with a timeout so a misconfigured or
unauthorized project surfaces here, at startup, with a message naming the fix.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: Seconds to wait for the startup round-trip before declaring the auth broken.
DEFAULT_AUTH_TIMEOUT_S = 60


class GeeAuthError(RuntimeError):
    """Raised when Earth Engine cannot be initialized. Message names the fix."""


def resolve_project(
    explicit: Optional[str] = None, config_default: Optional[str] = None
) -> str:
    """Resolve the GEE project id, or raise with instructions.

    Order: explicit argument, ``EE_PROJECT`` env var, config default.
    """
    for candidate in (explicit, os.environ.get("EE_PROJECT"), config_default):
        if candidate and str(candidate).strip() and str(candidate).strip() != "CHANGE_ME":
            return str(candidate).strip()

    raise GeeAuthError(
        "No Google Earth Engine project id was provided.\n"
        "Set one of the following:\n"
        "  * --ee-project on the command line, or\n"
        "  * EE_PROJECT in your .env file / environment, or\n"
        "  * scene_download.ee_project_default in config.yaml\n"
        "You can find your project id at https://code.earthengine.google.com/ "
        "(top-right project selector), or create one at "
        "https://console.cloud.google.com/earth-engine."
    )


def credentials_exist(credentials_file: Optional[str | Path] = None) -> bool:
    """Whether an Earth Engine credential file is present."""
    if credentials_file:
        return Path(credentials_file).is_file()
    default = Path.home() / ".config" / "earthengine" / "credentials"
    return default.is_file()


def initialize_ee(
    project: Optional[str] = None,
    config_default: Optional[str] = None,
    credentials_file: Optional[str | Path] = None,
    service_account_key: Optional[str | Path] = None,
    verify: bool = True,
):
    """Initialize Earth Engine, raising :class:`GeeAuthError` on any failure.

    Parameters
    ----------
    project
        Explicit project id (highest priority).
    config_default
        Fallback project id from config.yaml.
    credentials_file
        Optional non-default credential file to check for existence.
    service_account_key
        Path to a service-account JSON key, for unattended runs.
    verify
        Perform a round-trip call to prove the client really works.

    Returns
    -------
    module
        The initialized ``ee`` module.
    """
    try:
        import ee
    except ImportError as exc:
        raise GeeAuthError(
            "The earthengine-api package is not installed.\n"
            "Install it with:  pip install earthengine-api"
        ) from exc

    project_id = resolve_project(project, config_default)

    if service_account_key:
        key_path = Path(service_account_key)
        if not key_path.is_file():
            raise GeeAuthError(
                f"Service-account key file not found: {key_path}\n"
                f"Check scene_download credentials configuration, or omit it to "
                f"use the interactive credentials from `earthengine authenticate`."
            )
        try:
            import json

            account = json.loads(key_path.read_text(encoding="utf-8")).get("client_email")
            credentials = ee.ServiceAccountCredentials(account, str(key_path))
            ee.Initialize(credentials, project=project_id)
        except Exception as exc:
            raise GeeAuthError(
                f"Failed to initialize Earth Engine with the service account key "
                f"'{key_path}' for project '{project_id}': {exc}"
            ) from exc
    else:
        if not credentials_exist(credentials_file):
            location = credentials_file or (
                Path.home() / ".config" / "earthengine" / "credentials"
            )
            raise GeeAuthError(
                f"No Earth Engine credentials found at: {location}\n"
                f"Authenticate once with:\n"
                f"    earthengine authenticate\n"
                f"or, for an unattended run, provide a service-account key file."
            )
        try:
            ee.Initialize(project=project_id)
        except Exception as exc:
            raise GeeAuthError(
                f"Failed to initialize Earth Engine for project '{project_id}': {exc}\n"
                f"Common causes:\n"
                f"  * the project is not registered for Earth Engine "
                f"(register at https://code.earthengine.google.com/register)\n"
                f"  * your account lacks access to this project\n"
                f"  * stale credentials -- re-run `earthengine authenticate`"
            ) from exc

    if verify:
        _verify_connection(ee, project_id)

    return ee


def _verify_connection(ee, project_id: str, timeout_s: int = DEFAULT_AUTH_TIMEOUT_S) -> None:
    """Prove the client works with a tiny round-trip, bounded by a timeout.

    Runs the call on a daemon thread so a hung network call surfaces as a clear
    timeout error instead of blocking the pipeline indefinitely.
    """
    import threading

    result: dict = {}

    def _call() -> None:
        try:
            result["value"] = ee.Number(1).add(1).getInfo()
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=_call, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        raise GeeAuthError(
            f"Earth Engine did not respond within {timeout_s}s for project "
            f"'{project_id}'. Check your network connection and that the project "
            f"is registered for Earth Engine."
        )
    if "error" in result:
        raise GeeAuthError(
            f"Earth Engine connectivity check failed for project '{project_id}': "
            f"{result['error']}\n"
            f"Verify the project is registered for Earth Engine and that your "
            f"account has access to it."
        )
    if result.get("value") != 2:
        raise GeeAuthError(
            f"Earth Engine connectivity check returned an unexpected result "
            f"({result.get('value')!r}) for project '{project_id}'."
        )
