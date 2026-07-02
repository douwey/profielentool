from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from profielentool.io import (
    classify_points_by_zone,
    csv_to_points_gdf,
    find_intersecting_lines,
    point_distances_to_line,
    points_to_profile_line,
)


def test_app_file_exists() -> None:
    assert Path("app.py").exists()


def test_csv_to_points_requires_xyz_columns() -> None:
    df = pd.DataFrame({"x": [0], "y": [0]})
    with pytest.raises(ValueError):
        csv_to_points_gdf(df)


def test_intersection_and_distance_workflow() -> None:
    df = pd.DataFrame(
        {
            "x": [0.0, 5.0, 10.0],
            "y": [0.0, 0.0, 0.0],
            "z": [1.0, 1.1, 1.2],
        }
    )
    points = csv_to_points_gdf(df)
    profile = points_to_profile_line(points)

    axis = gpd.GeoDataFrame(
        {"name": ["axis_a"]},
        geometry=[LineString([(5.0, -5.0), (5.0, 5.0)])],
        crs="EPSG:28992",
    )

    intersecting = find_intersecting_lines(profile, axis)
    assert len(intersecting) == 1

    distances = point_distances_to_line(points, intersecting.iloc[0].geometry)
    assert distances.tolist() == [5.0, 0.0, 5.0]


def test_classify_points_by_zone() -> None:
    points = gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=gpd.points_from_xy([0.5, 1.5, 5.0], [0.5, 1.5, 5.0]),
        crs="EPSG:28992",
    )
    kernzone = gpd.GeoDataFrame(
        {"name": ["kern"]},
        geometry=[Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
        crs="EPSG:28992",
    )
    beschermingszone = gpd.GeoDataFrame(
        {"name": ["besch"]},
        geometry=[Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])],
        crs="EPSG:28992",
    )

    zone_type = classify_points_by_zone(points, kernzone, beschermingszone)
    assert zone_type.tolist() == ["Kernzone", "Beschermingszone", "Geen"]
