"""Spatial domain definitions for satrain_now.

Use get_domain(name) to retrieve a domain by name,
list_domains() to see all registered names, and
register_domain(name, area_def) to add custom domains.
"""

from pyresample import create_area_def
from pyresample.geometry import AreaDefinition

_domains: dict[str, AreaDefinition] = {}


def register_domain(name: str, area_def: AreaDefinition) -> AreaDefinition:
    _domains[name] = area_def
    return area_def


def get_domain(name: str) -> AreaDefinition:
    try:
        return _domains[name]
    except KeyError:
        raise KeyError(
            f"Unknown domain {name!r}. Available: {sorted(_domains)}"
        ) from None


def list_domains() -> list[str]:
    return sorted(_domains)


# ---------------------------------------------------------------------------
# Registered domains
# ---------------------------------------------------------------------------

conus = register_domain(
    "conus",
    create_area_def(
        area_id="conus",
        description="Regular lat/lon grid w 0.05 degree resolution covering conus.",
        projection={"proj": "longlat", "datum": "WGS84"},
        resolution=0.05,
        area_extent=(-125.0, 25.0, -65.0, 50.0),
    ),
)

conus_small = register_domain(
    "conus_small",
    create_area_def(
        area_id="conus_small",
        description="Regular lat/lon grid w 0.05 degree resolution covering a part of conus.",
        projection={"proj": "longlat", "datum": "WGS84"},
        resolution=0.05,
        area_extent=(-100.0, 30.0, -90.0, 40.0),
    ),
)
