from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from rasterio.io import MemoryFile
from shapely.geometry import LineString, Point

AHN_WCS_URL = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
DEFAULT_AHN_COVERAGE = "dtm_05m"


def _fetch_ahn_capabilities_xml() -> str:
    """Fetch WCS capabilities XML from PDOK."""
    params = {
        "SERVICE": "WCS",
        "VERSION": "1.0.0",
        "REQUEST": "GetCapabilities",
    }
    response = requests.get(AHN_WCS_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.text


def _parse_ahn_generation(xml: str) -> str | None:
    """Best-effort extraction of AHN generation label (e.g. AHN5)."""
    xml_l = xml.lower()
    patterns = [
        r"\bahn\s*([0-9]{1,2})\b",
        r"actueel[_\-]?ahn([0-9]{1,2})",
        r"\bahn([0-9]{1,2})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, xml_l, flags=re.IGNORECASE)
        if match:
            return f"AHN{int(match.group(1))}"
    return None


def get_ahn_service_info() -> tuple[str | None, str]:
    """Return AHN generation label (if found) and DTM coverage id."""
    xml = _fetch_ahn_capabilities_xml()

    coverage = DEFAULT_AHN_COVERAGE
    if re.search(r"<name>dtm_05m</name>", xml, flags=re.IGNORECASE):
        coverage = "dtm_05m"
    else:
        matches = re.findall(r"<name>(dtm_[0-9]+m)</name>", xml, flags=re.IGNORECASE)
        if matches:
            coverage = matches[0].lower()

    generation = _parse_ahn_generation(xml)
    return generation, coverage


def get_latest_ahn_dtm_coverage() -> str:
    """Read WCS capabilities and return the latest AHN DTM coverage id."""
    _, coverage = get_ahn_service_info()
    return coverage


def sample_distances(length_m: float, step_m: float) -> np.ndarray:
    """Generate chainage distances along a line, including the endpoint."""
    if length_m <= 0:
        return np.array([0.0])

    distances = np.arange(0.0, length_m, step_m)
    if len(distances) == 0 or distances[-1] < length_m:
        distances = np.append(distances, length_m)
    return distances


def points_from_line(line: LineString, distances_m: Iterable[float]) -> list[Point]:
    """Interpolate shapely points on a line at the given distances."""
    return [line.interpolate(float(d)) for d in distances_m]


def fetch_ahn_raster_for_line(
    line_rd: LineString,
    coverage_id: str,
    resolution_m: float = 0.5,
    padding_m: float = 5.0,
) -> bytes:
    """Request a small WCS GeoTIFF around the profile line in RD New."""
    minx, miny, maxx, maxy = line_rd.bounds
    minx -= padding_m
    miny -= padding_m
    maxx += padding_m
    maxy += padding_m

    width = max(2, int(np.ceil((maxx - minx) / resolution_m)))
    height = max(2, int(np.ceil((maxy - miny) / resolution_m)))

    params = {
        "SERVICE": "WCS",
        "VERSION": "1.0.0",
        "REQUEST": "GetCoverage",
        "COVERAGE": coverage_id,
        "CRS": "EPSG:28992",
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": "GEOTIFF",
    }
    response = requests.get(AHN_WCS_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.content


def sample_ahn_profile(
    line_rd: LineString,
    step_m: float = 0.5,
    max_length_m: float = 200.0,
    coverage_id: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Sample AHN heights along a line and return an XYZ profile table."""
    if line_rd.length > max_length_m:
        raise ValueError(f"Lijn is te lang ({line_rd.length:.2f} m). Maximum is {max_length_m:.0f} m.")

    coverage = coverage_id or get_latest_ahn_dtm_coverage()
    distances = sample_distances(line_rd.length, step_m)
    points = points_from_line(line_rd, distances)
    raster_bytes = fetch_ahn_raster_for_line(line_rd, coverage_id=coverage, resolution_m=step_m)

    xy_coords = [(pt.x, pt.y) for pt in points]
    with MemoryFile(raster_bytes) as memfile:
        with memfile.open() as dataset:
            sampled = list(dataset.sample(xy_coords))
            nodata = dataset.nodata

    z_values: list[float] = []
    for sample in sampled:
        value = float(sample[0])
        if nodata is not None and value == nodata:
            z_values.append(float("nan"))
        else:
            z_values.append(value)

    profile_df = pd.DataFrame(
        {
            "x": [pt.x for pt in points],
            "y": [pt.y for pt in points],
            "z": z_values,
            "distance_m": distances,
        }
    )
    return profile_df, coverage