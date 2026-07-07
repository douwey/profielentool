from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / "src"))

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
from folium.plugins import Draw
from pyproj import Transformer
from shapely.geometry import LineString
from streamlit_folium import st_folium

from profielentool.ahn import sample_ahn_profile
from profielentool.io import (
    classify_points_by_zone,
    enrich_axis_with_local_attributes,
    find_intersecting_lines,
    first_intersection_distance,
    load_local_axis_attribute_lookup,
    load_overige_layers_from_local,
    load_regionale_layers_from_remote,
    point_distances_to_line,
)


st.set_page_config(page_title="Dwarsprofielen Tool", page_icon="📈", layout="wide")

st.title("Dwarsprofielen uit kaart en AHN")
st.write(
    "Teken een lijn op de kaart waar je een dwarsprofiel wilt maken. "
    "Indien beschikbaar wordt automatisch gekoppeld met het bijbehorende "
    "leggerprofiel en profiel van vrije ruimte."
)

st.caption("Databronnen: Waterkeringen FeatureServer (online) en AHN via PDOK.")

try:
    regionale_layers = load_regionale_layers_from_remote()
except Exception as exc:
    st.error(f"Online bron niet beschikbaar: {exc}")
    st.stop()

overige_layers = load_overige_layers_from_local()
if overige_layers:
    for key, gdf_extra in overige_layers.items():
        if key in regionale_layers:
            base_gdf = regionale_layers[key]
            if base_gdf.crs is not None and gdf_extra.crs is not None and str(base_gdf.crs) != str(gdf_extra.crs):
                gdf_extra = gdf_extra.to_crs(base_gdf.crs)
            regionale_layers[key] = gpd.GeoDataFrame(
                pd.concat([base_gdf, gdf_extra], ignore_index=True),
                crs=base_gdf.crs,
            )
        else:
            regionale_layers[key] = gdf_extra

axis_gdf = regionale_layers["As_waterkering_BWK"]
kernzone_gdf = regionale_layers["Kernzone_BWK"]
beschermingszone_gdf = regionale_layers["Beschermingszone_BWK"]

try:
    local_axis_lookup = load_local_axis_attribute_lookup()
    axis_gdf = enrich_axis_with_local_attributes(axis_gdf, local_axis_lookup)
except Exception as exc:
    st.warning(f"Verrijking met lokale dijktafel/profiel niet beschikbaar: {exc}")

axis_4326 = axis_gdf.to_crs("EPSG:4326")
kernzone_4326 = kernzone_gdf.to_crs("EPSG:4326")
bescherm_4326 = beschermingszone_gdf.to_crs("EPSG:4326")
center = axis_4326.union_all().centroid

st.subheader("Kaart")
st.caption("Teken een lijn (maximaal 200 m) voor de profielopbouw.")
map_obj = folium.Map(location=[center.y, center.x], zoom_start=14, tiles="CartoDB positron")

folium.GeoJson(
    kernzone_4326,
    name="Kernzone",
    style_function=lambda _: {
        "color": "#c78f00",
        "weight": 1,
        "fillColor": "#ffd56a",
        "fillOpacity": 0.25,
    },
).add_to(map_obj)

folium.GeoJson(
    bescherm_4326,
    name="Beschermingszone",
    style_function=lambda _: {
        "color": "#007f5f",
        "weight": 1,
        "fillColor": "#80ed99",
        "fillOpacity": 0.2,
    },
).add_to(map_obj)

folium.GeoJson(
    axis_4326,
    name="As waterkering",
    style_function=lambda _: {"color": "#145da0", "weight": 3},
).add_to(map_obj)

Draw(
    draw_options={
        "polyline": True,
        "polygon": False,
        "rectangle": False,
        "circle": False,
        "marker": False,
        "circlemarker": False,
    },
    edit_options={"edit": True, "remove": True},
).add_to(map_obj)

folium.LayerControl(collapsed=False, position="bottomleft").add_to(map_obj)
map_state = st_folium(map_obj, height=550, use_container_width=True)

drawn_feature = map_state.get("last_active_drawing") if map_state else None
if not drawn_feature:
    st.info("Teken een lijn op de kaart om het AHN-profiel op te halen.")
    st.stop()

geometry = drawn_feature.get("geometry", {})
if geometry.get("type") != "LineString":
    st.warning("Teken een lijn (LineString).")
    st.stop()

transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
rd_coords = [transformer.transform(lon, lat) for lon, lat in geometry.get("coordinates", [])]
profile_line_rd = LineString(rd_coords)

st.write(f"Lengte getekende lijn: {profile_line_rd.length:.2f} m")
if profile_line_rd.length > 200:
    st.error("De lijn is langer dan 200 m. Teken een kortere lijn.")
    st.stop()

axis_for_rd = axis_gdf.to_crs("EPSG:28992")
kern_for_rd = kernzone_gdf.to_crs("EPSG:28992")
bescherm_for_rd = beschermingszone_gdf.to_crs("EPSG:28992")

try:
    with st.spinner("AHN-hoogtes ophalen..."):
        ahn_profile_df, ahn_coverage = sample_ahn_profile(
            profile_line_rd,
            step_m=0.5,
            max_length_m=max(250.0, profile_line_rd.length + 5.0),
        )
except Exception as exc:
    st.error(f"Fout bij AHN-profielopbouw: {exc}")
    st.stop()

points_ahn = gpd.GeoDataFrame(
    ahn_profile_df.copy(),
    geometry=gpd.points_from_xy(ahn_profile_df["x"], ahn_profile_df["y"]),
    crs="EPSG:28992",
)

zone_type = classify_points_by_zone(points_ahn, kern_for_rd, bescherm_for_rd)
ahn_profile_df["zone_type"] = zone_type.values

intersects = find_intersecting_lines(profile_line_rd, axis_for_rd)
if "CODE" in intersects.columns:
    intersects = intersects[intersects["CODE"].notna() & intersects["CODE"].astype(str).str.strip().ne("")].copy()
st.write(f"Aantal kruisende as-lijnen: {len(intersects)}")

selected_axis = None
crossing_distance = None

if len(intersects) > 0:
    crossing_rows: list[dict[str, object]] = []
    for row_idx, row in intersects.iterrows():
        row_dict = row.drop(labels=["geometry"]).to_dict()
        cross_d = first_intersection_distance(profile_line_rd, row.geometry)
        crossing_rows.append(
            {
                "axis_row_index": int(row_idx),
                "crossing_distance_m": cross_d,
                **row_dict,
            }
        )

    crossing_df = pd.DataFrame(crossing_rows)
    crossing_df = crossing_df.sort_values(by="crossing_distance_m", na_position="last").reset_index(drop=True)
    st.caption("Kruisende as-lijn(en)")
    st.dataframe(crossing_df, use_container_width=True)

    if len(crossing_df) > 1:
        crossing_df["CODE_display"] = crossing_df["CODE"].fillna("ONBEKEND").astype(str)
        code_options = crossing_df["CODE_display"].drop_duplicates().tolist()
        code_col, _ = st.columns([1, 4])
        with code_col:
            selected_code = st.selectbox("Kies CODE", options=code_options, index=0)
        selected_rows = crossing_df[crossing_df["CODE_display"] == selected_code].copy()
        selected_rows = selected_rows.sort_values(by="crossing_distance_m", na_position="last")
        selected_axis_row_index = int(selected_rows.iloc[0]["axis_row_index"])
    else:
        selected_axis_row_index = int(crossing_df.iloc[0]["axis_row_index"])

    selected_axis = intersects.loc[selected_axis_row_index]
    crossing_distance = first_intersection_distance(profile_line_rd, selected_axis.geometry)

else:
    st.warning("Geen kruising met as gevonden: grafiek-as start op het lijnbegin.")

if crossing_distance is None:
    ahn_profile_df["distance_from_crossing_m"] = ahn_profile_df["distance_m"]
else:
    ahn_profile_df["distance_from_crossing_m"] = ahn_profile_df["distance_m"] - crossing_distance

# Flip corrigeert de tekenrichting (binnen->buiten versus buiten->binnen)
# voor de volledige profielopbouw.
flip_profile_lr = bool(st.session_state.get("flip_profile_lr", False))
ahn_profile_df["distance_for_profile_m"] = ahn_profile_df["distance_from_crossing_m"]
if flip_profile_lr:
    ahn_profile_df["distance_for_profile_m"] = -ahn_profile_df["distance_for_profile_m"]

# Leggerprofiel (alleen binnen kernzone)
# - Alleen voor profieltype A/B/C
# - Kruin: x van -1 t/m 1 op dijktafelhoogte
# - Links van de kruin (x < -1): talud 1:2 omlaag
# - Rechts: van x=1 t/m x=4 talud 1:2, daarna profielafhankelijk:
#   A -> 1:4, B -> 1:6, C -> 1:8
ahn_profile_df["z_legger"] = np.nan
if selected_axis is not None and "Dijktafelh" in selected_axis.index:
    try:
        dijktafelhoogte = float(selected_axis["Dijktafelh"])
        profiel_raw = selected_axis.get("Profiel", None)
        profiel_code = ""
        if pd.notna(profiel_raw):
            profiel_code = str(profiel_raw).strip().upper()

        x_vals = ahn_profile_df["distance_for_profile_m"]
        mask_kernzone = ahn_profile_df["zone_type"].eq("Kernzone")

        # Basisprofiel zonder zoneclip
        z_legger_calc = np.full(len(ahn_profile_df), np.nan, dtype=float)
        mask_kruin = (x_vals >= -1.0) & (x_vals <= 1.0)
        z_legger_calc[mask_kruin.to_numpy()] = dijktafelhoogte

        if profiel_code in {"A", "B", "C"}:
            tail_slope_by_profile = {"A": 4.0, "B": 6.0, "C": 8.0}
            mask_links = x_vals < -1.0
            mask_rechts_12 = (x_vals > 1.0) & (x_vals <= 4.0)
            mask_rechts_tail = x_vals > 4.0

            z_legger_calc[mask_links.to_numpy()] = dijktafelhoogte - ((-1.0 - x_vals[mask_links]) / 2.0)
            z_legger_calc[mask_rechts_12.to_numpy()] = dijktafelhoogte - ((x_vals[mask_rechts_12] - 1.0) / 2.0)

            tail_slope = tail_slope_by_profile[profiel_code]
            z_at_x4 = dijktafelhoogte - ((4.0 - 1.0) / 2.0)
            z_legger_calc[mask_rechts_tail.to_numpy()] = z_at_x4 - ((x_vals[mask_rechts_tail] - 4.0) / tail_slope)

        elif profiel_code == "EDP_A":
            mask_links = x_vals < -1.0
            mask_rechts = x_vals > 1.0

            z_legger_calc[mask_links.to_numpy()] = dijktafelhoogte - ((-1.0 - x_vals[mask_links]) / 4.0)
            z_legger_calc[mask_rechts.to_numpy()] = dijktafelhoogte - ((x_vals[mask_rechts] - 1.0) / 4.0)

        elif profiel_code == "EDP_B":
            mask_links = x_vals < -1.0
            mask_rechts_14 = (x_vals > 1.0) & (x_vals <= 7.0)
            mask_rechts_16 = x_vals > 7.0

            z_legger_calc[mask_links.to_numpy()] = dijktafelhoogte - ((-1.0 - x_vals[mask_links]) / 4.0)
            z_legger_calc[mask_rechts_14.to_numpy()] = dijktafelhoogte - ((x_vals[mask_rechts_14] - 1.0) / 4.0)

            z_at_x7 = dijktafelhoogte - ((7.0 - 1.0) / 4.0)
            z_legger_calc[mask_rechts_16.to_numpy()] = z_at_x7 - ((x_vals[mask_rechts_16] - 7.0) / 6.0)

        # Alleen binnen kernzone tekenen
        ahn_profile_df.loc[mask_kernzone, "z_legger"] = z_legger_calc[mask_kernzone.to_numpy()]
    except (TypeError, ValueError):
        pass

# Profiel van vrije ruimte:
# - Start bij x=4 met hoogte op dijktafelhoogte
# - Loopt met talud 1:2 omlaag tot kruising met leggerlijn
# - Volgt daarna de leggerlijn
ahn_profile_df["z_vrije_ruimte"] = np.nan
vrije_x_cross: float | None = None
if selected_axis is not None and "Dijktafelh" in selected_axis.index:
    try:
        dijktafelhoogte = float(selected_axis["Dijktafelh"])
        x_vals = ahn_profile_df["distance_for_profile_m"]
        z_legger_vals = ahn_profile_df["z_legger"]

        mask_base = (x_vals >= 4.0) & z_legger_vals.notna()
        if mask_base.any():
            z_vr_line = dijktafelhoogte - ((x_vals - 4.0) / 2.0)
            diff = z_vr_line - z_legger_vals

            # Bepaal snijpunt nauwkeuriger via lineaire interpolatie tussen opeenvolgende punten.
            cross_df = pd.DataFrame(
                {
                    "x": x_vals[mask_base].astype(float).to_numpy(),
                    "d": diff[mask_base].astype(float).to_numpy(),
                }
            ).sort_values("x", kind="mergesort")

            x_cross = None
            if not cross_df.empty:
                exact_zero = cross_df[cross_df["d"] == 0.0]
                if not exact_zero.empty:
                    x_cross = float(exact_zero.iloc[0]["x"])
                else:
                    xs = cross_df["x"].to_numpy(dtype=float)
                    ds = cross_df["d"].to_numpy(dtype=float)
                    for i in range(len(xs) - 1):
                        d0 = ds[i]
                        d1 = ds[i + 1]
                        # Zoek eerste tekenwisseling (of raakpunt) en interpoleer x op d=0.
                        if (d0 > 0.0 and d1 < 0.0) or (d0 < 0.0 and d1 > 0.0) or d1 == 0.0:
                            x0 = xs[i]
                            x1 = xs[i + 1]
                            if d1 == d0:
                                x_cross = float(x1)
                            else:
                                x_cross = float(x0 + ((0.0 - d0) * (x1 - x0) / (d1 - d0)))
                            break

            if x_cross is not None:
                vrije_x_cross = float(x_cross)
                mask_before = mask_base & (x_vals <= x_cross)
                mask_after = mask_base & (x_vals > x_cross)
                ahn_profile_df.loc[mask_before, "z_vrije_ruimte"] = z_vr_line[mask_before]
                ahn_profile_df.loc[mask_after, "z_vrije_ruimte"] = z_legger_vals[mask_after]
            else:
                # Geen snijpunt binnen het profielbereik: toon 1:2-talud waar beschikbaar.
                ahn_profile_df.loc[mask_base, "z_vrije_ruimte"] = z_vr_line[mask_base]
    except (TypeError, ValueError):
        pass

if selected_axis is not None:
    distances = point_distances_to_line(points_ahn, selected_axis.geometry)
    ahn_profile_df["distance_to_axis_m"] = distances.values
    ahn_profile_df["matched_axis_row_index"] = int(selected_axis.name)

    for col, val in selected_axis.drop(labels=["geometry"]).to_dict().items():
        ahn_profile_df[f"axis_{col}"] = val

graph_code = "-"
graph_profiel = "-"
if selected_axis is not None:
    code_val = selected_axis.get("CODE", None)
    profiel_val = selected_axis.get("Profiel", None)
    if pd.notna(code_val) and str(code_val).strip() != "":
        graph_code = str(code_val)
    if pd.notna(profiel_val) and str(profiel_val).strip() != "":
        graph_profiel = str(profiel_val)

st.subheader(f"Profielgrafiek - CODE: {graph_code} | Leggerprofiel: {graph_profiel}")
st.caption("0 op de x-as is de kruising met de as-waterkering.")
profile_df = ahn_profile_df.sort_values("distance_for_profile_m").copy()
st.checkbox("Flip maaiveld links/rechts", key="flip_profile_lr")
profile_df["plot_distance_maaiveld"] = profile_df["distance_for_profile_m"]
profile_df = profile_df.sort_values("plot_distance_maaiveld")


def interpolate_at_x(df: pd.DataFrame, x_col: str, y_col: str, x_value: float) -> float | None:
    series_df = df[[x_col, y_col]].dropna().copy()
    if series_df.empty:
        return None

    series_df = series_df.sort_values(x_col)
    series_df = series_df.drop_duplicates(subset=[x_col], keep="first")

    x_vals = series_df[x_col].to_numpy(dtype=float)
    y_vals = series_df[y_col].to_numpy(dtype=float)
    if len(x_vals) < 2:
        return None
    if x_value < float(x_vals[0]) or x_value > float(x_vals[-1]):
        return None

    return float(np.interp(x_value, x_vals, y_vals))


x_min_available = float(profile_df["plot_distance_maaiveld"].min())
x_max_available = float(profile_df["plot_distance_maaiveld"].max())
x_default = 0.0 if (x_min_available <= 0.0 <= x_max_available) else x_min_available
x_query_state = st.session_state.get("x_query_interp", x_default)
x_query = float(min(max(float(x_query_state), x_min_available), x_max_available))

z_maaiveld_at_x = interpolate_at_x(profile_df, "plot_distance_maaiveld", "z", x_query)
z_legger_at_x = interpolate_at_x(profile_df, "plot_distance_maaiveld", "z_legger", x_query)
z_vrije_ruimte_at_x = interpolate_at_x(profile_df, "plot_distance_maaiveld", "z_vrije_ruimte", x_query)

marker_rows: list[dict[str, object]] = []
if z_maaiveld_at_x is not None:
    marker_rows.append({"lijn": "Maaiveld", "plot_distance_maaiveld": x_query, "z": z_maaiveld_at_x})
if z_legger_at_x is not None:
    marker_rows.append({"lijn": "Legger", "plot_distance_maaiveld": x_query, "z": z_legger_at_x})
if z_vrije_ruimte_at_x is not None:
    marker_rows.append({"lijn": "Profiel vrije ruimte", "plot_distance_maaiveld": x_query, "z": z_vrije_ruimte_at_x})

zone_df = profile_df[["plot_distance_maaiveld", "zone_type"]].copy()
zone_df = zone_df.rename(columns={"plot_distance_maaiveld": "start"})
zone_df["end"] = zone_df["start"].shift(-1)

if len(zone_df) > 1:
    median_step = float((zone_df["start"].diff().dropna().abs().median()))
    if median_step <= 0:
        median_step = 0.5
else:
    median_step = 0.5

zone_df["end"] = zone_df["end"].fillna(zone_df["start"] + median_step)
zone_df["zone_band"] = "zone"

scale_mode = st.radio(
    "Asschaal",
    options=["Standaard", "Gelijke schaal (x=y)"],
    index=0,
    horizontal=True,
)

line_height = 260
zone_height = 50
chart_width_px = 1000

if scale_mode == "Gelijke schaal (x=y)":
    x_min = float(profile_df["plot_distance_maaiveld"].min())
    x_max = float(profile_df["plot_distance_maaiveld"].max())
    y_values = [profile_df["z"].to_numpy(dtype=float)]
    if profile_df["z_vrije_ruimte"].notna().any():
        y_values.append(profile_df["z_vrije_ruimte"].to_numpy(dtype=float))
    if profile_df["z_legger"].notna().any():
        y_values.append(profile_df["z_legger"].to_numpy(dtype=float))
    y_concat = np.concatenate(y_values)
    y_min = float(np.nanmin(y_concat))
    y_max = float(np.nanmax(y_concat))

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    # Keep chart size fixed and adapt only the y-axis domain for visual x=y scaling.
    target_y_span = x_span * (line_height / chart_width_px)
    effective_y_span = max(y_span, target_y_span)
    y_center = (y_min + y_max) / 2.0
    y_domain = [y_center - (effective_y_span / 2.0), y_center + (effective_y_span / 2.0)]

    x_encoding_line = alt.X(
        "plot_distance_maaiveld:Q",
        title=None,
        axis=alt.Axis(labels=False, ticks=False),
        scale=alt.Scale(domain=[x_min, x_max], nice=False),
    )
    y_encoding_line = alt.Y(
        "z:Q",
        title="Hoogte (m NAP)",
        scale=alt.Scale(domain=y_domain, nice=False),
    )
    x_encoding_zone = alt.X(
        "start:Q",
        title="Afstand vanaf kruising (m)",
        scale=alt.Scale(domain=[x_min, x_max], nice=False),
    )
else:
    x_encoding_line = alt.X(
        "plot_distance_maaiveld:Q",
        title=None,
        axis=alt.Axis(labels=False, ticks=False),
    )
    y_encoding_line = alt.Y("z:Q", title="Hoogte (m NAP)")
    x_encoding_zone = alt.X("start:Q", title="Afstand vanaf kruising (m)")

measured_chart = (
    alt.Chart(profile_df)
    .mark_line(color="#2d6a4f", strokeWidth=2)
    .encode(
        x=x_encoding_line,
        y=y_encoding_line,
        tooltip=[
            alt.Tooltip("distance_from_crossing_m:Q", title="Afstand (m)", format=".2f"),
            alt.Tooltip("plot_distance_maaiveld:Q", title="Afstand weergegeven (m)", format=".2f"),
            alt.Tooltip("z:Q", title="Hoogte (m)", format=".2f"),
            alt.Tooltip("zone_type:N", title="Zone"),
        ],
    )
)
measured_chart = measured_chart.properties(width=chart_width_px, height=line_height)

line_chart = measured_chart
if profile_df["z_vrije_ruimte"].notna().any():
    vrije_df = profile_df.dropna(subset=["z_vrije_ruimte"]).copy()

    extra_rows: list[dict[str, float]] = []
    x_min_line = float(profile_df["distance_for_profile_m"].min())
    x_max_line = float(profile_df["distance_for_profile_m"].max())

    # Dwing een exact startknooppunt op x=4 af voor de weergave van vrije ruimte.
    if selected_axis is not None and "Dijktafelh" in selected_axis.index and x_min_line <= 4.0 <= x_max_line:
        try:
            dth = float(selected_axis["Dijktafelh"])
            extra_rows.append({"plot_distance_maaiveld": 4.0, "z_vrije_ruimte": dth})
        except (TypeError, ValueError):
            pass

    # Dwing ook een exact knooppunt op het berekende kruispunt vrije ruimte <-> legger af.
    if vrije_x_cross is not None and x_min_line <= vrije_x_cross <= x_max_line:
        z_legger_cross = interpolate_at_x(profile_df, "distance_for_profile_m", "z_legger", vrije_x_cross)
        if z_legger_cross is not None:
            extra_rows.append(
                {
                    "plot_distance_maaiveld": float(vrije_x_cross),
                    "z_vrije_ruimte": float(z_legger_cross),
                }
            )

    if extra_rows:
        vrije_df = pd.concat([vrije_df, pd.DataFrame(extra_rows)], ignore_index=True)
        vrije_df = vrije_df.sort_values("plot_distance_maaiveld", kind="mergesort")
        vrije_df = vrije_df.drop_duplicates(subset=["plot_distance_maaiveld"], keep="last")

    vrije_chart = (
        alt.Chart(vrije_df)
        .mark_line(color="#1d4ed8", strokeWidth=2)
        .encode(
            x=x_encoding_line,
            y=alt.Y("z_vrije_ruimte:Q", title="Hoogte (m NAP)"),
            tooltip=[
                alt.Tooltip("plot_distance_maaiveld:Q", title="Afstand weergegeven (m)", format=".2f"),
                alt.Tooltip("z_vrije_ruimte:Q", title="Vrije ruimte (m)", format=".2f"),
            ],
        )
        .properties(width=chart_width_px, height=line_height)
    )
    line_chart = alt.layer(measured_chart, vrije_chart)

if profile_df["z_legger"].notna().any():
    legger_df = profile_df.dropna(subset=["z_legger"]).copy()
    legger_chart = (
        alt.Chart(legger_df)
        .mark_line(color="#c1121f", strokeWidth=2, strokeDash=[6, 4])
        .encode(
            x=x_encoding_line,
            y=alt.Y("z_legger:Q", title="Hoogte (m NAP)"),
            tooltip=[
                alt.Tooltip("plot_distance_maaiveld:Q", title="Afstand weergegeven (m)", format=".2f"),
                alt.Tooltip("z_legger:Q", title="Leggerhoogte (m)", format=".2f"),
            ],
        )
        .properties(width=chart_width_px, height=line_height)
    )
    if profile_df["z_vrije_ruimte"].notna().any():
        line_chart = alt.layer(measured_chart, vrije_chart, legger_chart)
    else:
        line_chart = alt.layer(measured_chart, legger_chart)

if marker_rows:
    marker_df = pd.DataFrame(marker_rows)
    marker_layers = [line_chart]

    marker_style = {"filled": True, "size": 110, "stroke": "#ffffff", "strokeWidth": 1}
    marker_tooltip = [
        alt.Tooltip("lijn:N", title="Lijn"),
        alt.Tooltip("plot_distance_maaiveld:Q", title="X (m)", format=".2f"),
        alt.Tooltip("z:Q", title="Z (m)", format=".3f"),
    ]

    marker_colors = {
        "Maaiveld": "#2d6a4f",
        "Profiel vrije ruimte": "#1d4ed8",
        "Legger": "#c1121f",
    }

    for marker_name, marker_color in marker_colors.items():
        marker_slice = marker_df[marker_df["lijn"] == marker_name]
        if marker_slice.empty:
            continue

        marker_layers.append(
            alt.Chart(marker_slice)
            .mark_point(color=marker_color, **marker_style)
            .encode(
                x=x_encoding_line,
                y=alt.Y("z:Q", title="Hoogte (m NAP)"),
                tooltip=marker_tooltip,
            )
            .properties(width=chart_width_px, height=line_height)
        )

    line_chart = alt.layer(*marker_layers)

zone_chart = (
    alt.Chart(zone_df)
    .mark_bar()
    .encode(
        x=x_encoding_zone,
        x2="end:Q",
        y=alt.Y("zone_band:N", title=None, axis=None),
        color=alt.Color(
            "zone_type:N",
            title="Zone",
            scale=alt.Scale(
                domain=["Kernzone", "Beschermingszone", "Geen"],
                range=["#ffd56a", "#80ed99", "#d9d9d9"],
            ),
        ),
        tooltip=[
            alt.Tooltip("zone_type:N", title="Zone"),
            alt.Tooltip("start:Q", title="Van (m)", format=".2f"),
            alt.Tooltip("end:Q", title="Tot (m)", format=".2f"),
        ],
    )
)
zone_chart = zone_chart.properties(width=chart_width_px, height=zone_height)

combined_chart = alt.vconcat(line_chart, zone_chart).resolve_scale(x="shared", color="independent")
st.altair_chart(combined_chart, use_container_width=False)

export_df = profile_df[
    [
        "distance_from_crossing_m",
        "plot_distance_maaiveld",
        "z",
        "z_legger",
        "z_vrije_ruimte",
        "zone_type",
    ]
].copy()

export_col_csv, export_col_png = st.columns(2)
with export_col_csv:
    st.download_button(
        label="Exporteer grafiekdata (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="grafiek_profiel_data.csv",
        mime="text/csv",
    )

with export_col_png:
    png_bytes = None
    try:
        import vl_convert as vlc

        png_bytes = vlc.vegalite_to_png(combined_chart.to_dict(), scale=2)
    except Exception:
        png_bytes = None

    if png_bytes:
        st.download_button(
            label="Exporteer grafiek (PNG)",
            data=png_bytes,
            file_name="grafiek_profiel.png",
            mime="image/png",
        )
    else:
        st.caption("PNG-export niet beschikbaar (installeer vl-convert-python).")

x_query_input = st.number_input(
    "X-waarde voor interpolatie (grafiek-as)",
    min_value=x_min_available,
    max_value=x_max_available,
    value=x_query,
    step=0.5,
    format="%.2f",
    key="x_query_interp",
)

z_maaiveld_out = interpolate_at_x(profile_df, "plot_distance_maaiveld", "z", float(x_query_input))
z_legger_out = interpolate_at_x(profile_df, "plot_distance_maaiveld", "z_legger", float(x_query_input))
z_vrije_ruimte_out = interpolate_at_x(profile_df, "plot_distance_maaiveld", "z_vrije_ruimte", float(x_query_input))

interp_rows = [
    {"lijn": "Maaiveld", "z": z_maaiveld_out},
    {"lijn": "Legger", "z": z_legger_out},
    {"lijn": "Profiel vrije ruimte", "z": z_vrije_ruimte_out},
]
interp_df = pd.DataFrame(interp_rows)
interp_df["z"] = interp_df["z"].apply(lambda v: None if v is None else round(float(v), 3))
st.caption("Geinterpoleerde z-waarden op gekozen x")
st.dataframe(interp_df, use_container_width=True, hide_index=True)
