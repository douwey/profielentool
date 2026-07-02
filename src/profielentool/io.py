from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString


REQUIRED_REGIONALE_LAYER_TYPES: dict[str, str] = {
    "As_waterkering_BWK": "line",
    "Kernzone_BWK": "polygon",
    "Beschermingszone_BWK": "polygon",
}

ARCGIS_WATERKERINGEN_FEATURESERVER = (
    "https://services.arcgis.com/OnnVX2wGkBfflKqu/arcgis/rest/services/"
    "Waterkeringen_legger/FeatureServer"
)

REMOTE_REGIONALE_LAYER_IDS: dict[str, int] = {
    "As_waterkering_BWK": 10,
    "Kernzone_BWK": 11,
    "Beschermingszone_BWK": 12,
}

DEFAULT_LOCAL_AXIS_PATH = Path(
    "data/input/Legger2022/Shape/Regionale waterkeringen/As_waterkering_BWK.shp"
)
DEFAULT_OVERIGE_SHAPE_DIR = Path("data/input/Legger2022/Shape/Overige waterkeringen")


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame."""
    return pd.read_csv(path)


def read_vector(path: str | Path) -> gpd.GeoDataFrame:
    """Read a vector GIS file (e.g. SHP or GPKG)."""
    return gpd.read_file(path)


def normalize_axis_code(value: Any) -> str:
    """Normalize axis code values for robust joins across sources."""
    if value is None:
        return ""
    raw = str(value).strip().upper()
    # Harmonize codes like 'PLSA-007', 'PLSA 007', and 'PLSA007' to the same key.
    return re.sub(r"[^A-Z0-9]", "", raw)


def validate_xyz_columns(df: pd.DataFrame, columns: tuple[str, str, str] = ("x", "y", "z")) -> None:
    """Validate that the required XYZ columns are available in the DataFrame."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"CSV mist verplichte kolommen: {', '.join(missing)}")


def csv_to_points_gdf(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    crs: str = "EPSG:28992",
) -> gpd.GeoDataFrame:
    """Convert CSV point table to a GeoDataFrame with preserved row order."""
    validate_xyz_columns(df, (x_col, y_col, z_col))
    points_gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs=crs,
    )
    points_gdf["point_order"] = range(1, len(points_gdf) + 1)
    return points_gdf


def points_to_profile_line(points_gdf: gpd.GeoDataFrame) -> LineString:
    """Create a profile line from ordered points."""
    if len(points_gdf) < 2:
        raise ValueError("Minimaal 2 punten nodig om een lijn te maken.")
    coords = [(geom.x, geom.y) for geom in points_gdf.geometry]
    return LineString(coords)


def find_intersecting_lines(profile_line: LineString, lines_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return lines that intersect the profile line."""
    mask = lines_gdf.geometry.intersects(profile_line)
    return lines_gdf.loc[mask].copy()


def first_intersection_distance(line: LineString, other_geometry: Any) -> float | None:
    """Return projected distance on line for the first intersection point."""
    intersection = line.intersection(other_geometry)
    if intersection.is_empty:
        return None

    geom_type = intersection.geom_type
    if geom_type == "Point":
        return float(line.project(intersection))
    if geom_type == "MultiPoint":
        return float(min(line.project(pt) for pt in intersection.geoms))
    if geom_type in {"LineString", "LinearRing"}:
        return float(line.project(intersection.interpolate(0.0)))
    if geom_type == "MultiLineString":
        return float(min(line.project(g.interpolate(0.0)) for g in intersection.geoms if not g.is_empty))

    if hasattr(intersection, "geoms"):
        projected = []
        for geom in intersection.geoms:
            if geom.is_empty:
                continue
            if geom.geom_type == "Point":
                projected.append(float(line.project(geom)))
            elif geom.geom_type in {"LineString", "LinearRing"}:
                projected.append(float(line.project(geom.interpolate(0.0))))
        if projected:
            return min(projected)
    return None


def point_distances_to_line(points_gdf: gpd.GeoDataFrame, line_geometry: Any) -> pd.Series:
    """Calculate distance from each point to a target line geometry."""
    return points_gdf.geometry.distance(line_geometry)


def classify_points_by_zone(
    points_gdf: gpd.GeoDataFrame,
    kernzone_gdf: gpd.GeoDataFrame,
    beschermingszone_gdf: gpd.GeoDataFrame,
) -> pd.Series:
    """Classify each point as Kernzone, Beschermingszone or Geen."""
    kern_union = kernzone_gdf.geometry.union_all()
    bescherm_union = beschermingszone_gdf.geometry.union_all()

    zones: list[str] = []
    for point in points_gdf.geometry:
        if not kern_union.is_empty and kern_union.covers(point):
            zones.append("Kernzone")
        elif not bescherm_union.is_empty and bescherm_union.covers(point):
            zones.append("Beschermingszone")
        else:
            zones.append("Geen")

    return pd.Series(zones, index=points_gdf.index, name="zone_type")


def read_arcgis_feature_layer(service_url: str, layer_id: int) -> gpd.GeoDataFrame:
    """Read an ArcGIS FeatureServer layer as GeoJSON."""
    params = urlencode(
        {
            "where": "1=1",
            "outFields": "*",
            "outSR": 28992,
            "f": "geojson",
        }
    )
    query_url = f"{service_url.rstrip('/')}/{layer_id}/query?{params}"
    return gpd.read_file(query_url)


def load_regionale_layers_from_remote(
    service_url: str = ARCGIS_WATERKERINGEN_FEATURESERVER,
) -> dict[str, gpd.GeoDataFrame]:
    """Load required regionale layers from ArcGIS FeatureServer."""
    return {
        layer_name: read_arcgis_feature_layer(service_url, layer_id)
        for layer_name, layer_id in REMOTE_REGIONALE_LAYER_IDS.items()
    }


def load_local_axis_attribute_lookup(
    local_axis_path: str | Path = DEFAULT_LOCAL_AXIS_PATH,
    overige_shape_dir: str | Path = DEFAULT_OVERIGE_SHAPE_DIR,
) -> pd.DataFrame:
    """Load local axis attributes that are missing in the online service."""
    local_paths: list[Path] = [Path(local_axis_path)]
    overige_dir = Path(overige_shape_dir)
    if overige_dir.exists():
        local_paths.extend(sorted(overige_dir.glob("As_waterkering_*.shp")))

    frames: list[gpd.GeoDataFrame] = []
    for path in local_paths:
        if path.exists():
            frames.append(read_vector(path))

    if not frames:
        raise ValueError("Geen lokale as-lagen gevonden voor attribuutverrijking.")

    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    required = ["Code", "Dijktafelh", "Profiel"]
    missing = [col for col in required if col not in gdf.columns]
    if missing:
        raise ValueError(f"Lokale as-laag mist velden: {', '.join(missing)}")

    lookup = gdf[required].copy()
    lookup["CODE_NORM"] = lookup["Code"].map(normalize_axis_code)
    lookup = lookup.dropna(subset=["CODE_NORM"])
    lookup = lookup.drop_duplicates(subset=["CODE_NORM"])
    return lookup[["CODE_NORM", "Dijktafelh", "Profiel"]]


def enrich_axis_with_local_attributes(
    axis_gdf: gpd.GeoDataFrame,
    local_lookup: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Add Dijktafelh/Profiel from local lookup to remote axis lines by CODE."""
    enriched = axis_gdf.copy()
    if "CODE" not in enriched.columns:
        return enriched

    enriched["CODE_NORM"] = enriched["CODE"].map(normalize_axis_code)
    enriched = enriched.merge(local_lookup, how="left", on="CODE_NORM")

    for field in ["Dijktafelh", "Profiel"]:
        candidates = [c for c in [field, f"{field}_x", f"{field}_y"] if c in enriched.columns]
        if candidates:
            enriched[field] = enriched[candidates].bfill(axis=1).iloc[:, 0]

    drop_cols = [c for c in ["Dijktafelh_x", "Dijktafelh_y", "Profiel_x", "Profiel_y"] if c in enriched.columns]
    if drop_cols:
        enriched = enriched.drop(columns=drop_cols)

    return enriched


def load_overige_layers_from_local(
    overige_shape_dir: str | Path = DEFAULT_OVERIGE_SHAPE_DIR,
) -> dict[str, gpd.GeoDataFrame]:
    """Load Overige waterkeringen layers and aggregate them by type."""
    base = Path(overige_shape_dir)
    if not base.exists():
        return {}

    patterns = {
        "As_waterkering_BWK": "As_waterkering_*.shp",
        "Kernzone_BWK": "Kernzone_*.shp",
        "Beschermingszone_BWK": "Beschermingszone_*.shp",
    }

    result: dict[str, gpd.GeoDataFrame] = {}
    for key, pattern in patterns.items():
        files = sorted(base.glob(pattern))
        if not files:
            continue

        frames: list[gpd.GeoDataFrame] = []
        for shp in files:
            gdf = read_vector(shp)
            suffix = shp.stem.split("_")[-1]
            gdf = gdf.copy()
            gdf["OVERIGE_DEEL"] = suffix
            frames.append(gdf)

        result[key] = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)

    return result


def inspect_regionale_waterkeringen(shape_dir: str | Path) -> dict[str, dict[str, object]]:
    """Inspect required Regionale waterkeringen layers and validate geometry types."""
    base = Path(shape_dir)
    summary: dict[str, dict[str, object]] = {}

    for layer_name, expected_geom in REQUIRED_REGIONALE_LAYER_TYPES.items():
        shp_path = base / f"{layer_name}.shp"
        layer_info: dict[str, object] = {
            "path": str(shp_path),
            "exists": shp_path.exists(),
            "expected_geometry": expected_geom,
            "feature_count": 0,
            "geometry_types": [],
            "geometry_ok": False,
        }

        if shp_path.exists():
            gdf = gpd.read_file(shp_path)
            geometry_types = sorted([str(g) for g in gdf.geometry.geom_type.dropna().unique()])
            layer_info["feature_count"] = int(len(gdf))
            layer_info["geometry_types"] = geometry_types

            if expected_geom == "line":
                allowed = {"LineString", "MultiLineString"}
            else:
                allowed = {"Polygon", "MultiPolygon"}

            layer_info["geometry_ok"] = set(geometry_types).issubset(allowed) and len(geometry_types) > 0

        summary[layer_name] = layer_info

    return summary
