from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium


ROOT = Path(__file__).resolve().parent
SILVER_PATH = ROOT / "data" / "silver" / "silver_ev_charging_events.csv"
FEATURE_PATH = ROOT / "data" / "silver" / "ml_features.csv"
GOLD_DIR = ROOT / "data" / "gold"
METRICS_PATH = ROOT / "models" / "model_metrics.json"
SQLITE_PATH = ROOT / "data" / "ev_charging_platform.db"

CITY_COORDS = {
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Los Angeles": (34.0522, -118.2437),
    "New York": (40.7128, -74.0060),
    "San Francisco": (37.7749, -122.4194),
}


st.set_page_config(
    page_title="EV Charging Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


def stable_float(seed: str, low: float, high: float) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12 - 1)
    return low + (high - low) * value


def apply_theme(theme: str) -> None:
    dark = theme == "Dark"
    bg = "#071016" if dark else "#f5f7fb"
    panel = "#0e1b24" if dark else "#ffffff"
    panel_alt = "#132633" if dark else "#eef3f8"
    text = "#eef6fb" if dark else "#17212b"
    muted = "#9fb2c2" if dark else "#5c6975"
    border = "#243846" if dark else "#dbe4ec"
    accent = "#28c7a7"
    warning = "#f4b942"
    danger = "#f05f57"
    st.markdown(
        f"""
        <style>
        :root {{
            --ev-bg: {bg};
            --ev-panel: {panel};
            --ev-panel-alt: {panel_alt};
            --ev-text: {text};
            --ev-muted: {muted};
            --ev-border: {border};
            --ev-accent: {accent};
            --ev-warning: {warning};
            --ev-danger: {danger};
        }}
        .stApp {{ background: var(--ev-bg); color: var(--ev-text); }}
        [data-testid="stSidebar"] {{ background: var(--ev-panel); border-right: 1px solid var(--ev-border); }}
        h1, h2, h3, h4, h5, h6, p, label, span {{ color: var(--ev-text); }}
        div[data-testid="stMetric"] {{
            background: var(--ev-panel);
            border: 1px solid var(--ev-border);
            border-radius: 8px;
            padding: 16px;
            min-height: 116px;
        }}
        div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
            color: var(--ev-muted);
        }}
        .hero {{
            background: linear-gradient(135deg, rgba(40,199,167,.22), rgba(61,118,255,.12));
            border: 1px solid var(--ev-border);
            border-radius: 8px;
            padding: 22px 24px;
            margin-bottom: 18px;
        }}
        .hero h1 {{ margin: 0 0 8px 0; font-size: 2rem; }}
        .hero p {{ color: var(--ev-muted); margin: 0; max-width: 980px; }}
        .section-card {{
            background: var(--ev-panel);
            border: 1px solid var(--ev-border);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 14px;
        }}
        .station-row {{
            background: var(--ev-panel);
            border: 1px solid var(--ev-border);
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }}
        .status-open {{ color: var(--ev-accent); font-weight: 700; }}
        .status-busy {{ color: var(--ev-warning); font-weight: 700; }}
        .status-overloaded {{ color: var(--ev-danger); font-weight: 700; }}
        .small-muted {{ color: var(--ev-muted); font-size: .9rem; }}
        .stDataFrame, .stTable {{ background: var(--ev-panel); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict]:
    silver = pd.read_csv(SILVER_PATH, parse_dates=["Charging Start Time", "Charging End Time"])
    features = pd.read_csv(FEATURE_PATH)
    gold = {
        path.stem: pd.read_csv(path)
        for path in GOLD_DIR.glob("*.csv")
    }
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    return silver, features, gold, metrics


@st.cache_resource(show_spinner=False)
def initialize_sqlite(silver: pd.DataFrame, features: pd.DataFrame, gold: dict[str, pd.DataFrame]) -> str:
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(SQLITE_PATH) as conn:
        silver.to_sql("silver_charging_events", conn, if_exists="replace", index=False)
        features.to_sql("ml_features", conn, if_exists="replace", index=False)
        for name, frame in gold.items():
            frame.to_sql(name, conn, if_exists="replace", index=False)
    return str(SQLITE_PATH)


def with_station_coordinates(stations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in stations.iterrows():
        city = row["Charging Station Location"]
        base_lat, base_lon = CITY_COORDS.get(city, (39.5, -98.35))
        lat = row.get("latitude", np.nan)
        lon = row.get("longitude", np.nan)
        if pd.isna(lat) or pd.isna(lon) or float(lat) == 0 or float(lon) == 0:
            key = str(row["station_key"])
            lat = base_lat + stable_float(key + "lat", -0.11, 0.11)
            lon = base_lon + stable_float(key + "lon", -0.13, 0.13)
        row = row.copy()
        row["map_latitude"] = float(lat)
        row["map_longitude"] = float(lon)
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_station_state(silver: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    base = silver.merge(
        features[[
            "station_key",
            "station_occupancy_percent",
            "estimated_waiting_time_minutes",
            "overload_risk",
        ]],
        on="station_key",
        how="left",
    )
    grouped = (
        base.groupby(["station_key", "Charging Station ID", "Charging Station Location"], dropna=False)
        .agg(
            station_name=("name", lambda s: next((x for x in s.dropna().astype(str) if x.strip()), None)),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            total_sessions=("charging_session_count", "sum"),
            total_energy_kwh=("Energy Consumed (kWh)", "sum"),
            avg_duration_hours=("Charging Duration (hours)", "mean"),
            avg_rate_kw=("Charging Rate (kW)", "mean"),
            avg_traffic_score=("traffic_score", "mean"),
            avg_congestion=("road_congestion_index", "mean"),
            avg_temp_c=("temperature_c", "mean"),
            avg_waiting_minutes=("estimated_waiting_time_minutes", "mean"),
            predicted_occupancy=("station_occupancy_percent", "mean"),
            overload_risk=("overload_risk", "max"),
            chargers=("num_connectors", "max"),
            total_kw=("total_kw", "max"),
            latest_hour=("charging_hour", "max"),
        )
        .reset_index()
    )
    grouped["station_name"] = grouped["station_name"].fillna(grouped["Charging Station ID"])
    grouped["chargers"] = grouped["chargers"].fillna(0).replace(0, np.nan)
    grouped["chargers"] = grouped["chargers"].fillna(grouped["total_sessions"].rank(pct=True).mul(8).add(2)).round().clip(2, 16)
    grouped["occupied_chargers"] = np.ceil(grouped["chargers"] * grouped["predicted_occupancy"].fillna(45) / 100).clip(0, grouped["chargers"]).astype(int)
    grouped["available_chargers"] = (grouped["chargers"] - grouped["occupied_chargers"]).clip(lower=0).astype(int)
    grouped["current_vehicles_charging"] = grouped["occupied_chargers"] + np.ceil(grouped["avg_traffic_score"] / 35).astype(int)
    grouped["utilization_percent"] = (grouped["occupied_chargers"] / grouped["chargers"] * 100).round(1)
    grouped["status"] = np.select(
        [grouped["utilization_percent"] >= 90, grouped["utilization_percent"] >= 70, grouped["available_chargers"] >= 2],
        ["Overloaded", "Busy", "Available"],
        default="Limited",
    )
    grouped = with_station_coordinates(grouped)
    return grouped.sort_values(["Charging Station Location", "avg_waiting_minutes", "predicted_occupancy"])


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def city_from_query() -> str | None:
    params = st.query_params
    if "lat" not in params or "lon" not in params:
        return None
    try:
        lat = float(params["lat"])
        lon = float(params["lon"])
    except (TypeError, ValueError):
        return None
    return min(CITY_COORDS, key=lambda city: haversine_km(lat, lon, *CITY_COORDS[city]))


def selected_city_control(cities: list[str]) -> tuple[str, tuple[float, float]]:
    query_city = city_from_query()
    if query_city:
        st.session_state["selected_city"] = query_city
    st.sidebar.markdown("### City Selection")
    typed_city = st.sidebar.text_input("Enter city", value="")
    default_city = st.session_state.get("selected_city", "Los Angeles")
    if typed_city.strip() in cities:
        default_city = typed_city.strip()
    selected = st.sidebar.selectbox("Select city", cities, index=cities.index(default_city) if default_city in cities else 0)
    if st.sidebar.button("Use current location", use_container_width=True):
        with st.sidebar:
            components.html(
                """
                <script>
                navigator.geolocation.getCurrentPosition(
                  (pos) => {
                    const url = new URL(window.parent.location.href);
                    url.searchParams.set("lat", pos.coords.latitude);
                    url.searchParams.set("lon", pos.coords.longitude);
                    window.parent.location.href = url.toString();
                  },
                  () => alert("Location permission was not granted.")
                );
                </script>
                """,
                height=0,
            )
    st.session_state["selected_city"] = selected
    return selected, CITY_COORDS[selected]


def filter_city(df: pd.DataFrame, city: str) -> pd.DataFrame:
    return df[df["Charging Station Location"].eq(city)].copy()


def metric_row(summary: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Total Stations", f"{summary['stations']:,}")
    cols[1].metric("Total Active Sessions", f"{summary['sessions']:,}")
    cols[2].metric("Average Waiting Time", f"{summary['waiting']:.1f} min")
    cols[3].metric("Total Vehicles Charging", f"{summary['vehicles']:,}")
    cols[4].metric("Peak Demand Areas", f"{summary['peak_areas']:,}")


def city_summary(stations: pd.DataFrame, features: pd.DataFrame, city: str) -> dict:
    city_stations = filter_city(stations, city)
    city_features = filter_city(features, city)
    return {
        "stations": city_stations["station_key"].nunique(),
        "sessions": len(city_features),
        "waiting": city_stations["avg_waiting_minutes"].mean() if len(city_stations) else 0,
        "vehicles": int(city_stations["current_vehicles_charging"].sum()),
        "peak_areas": int((city_stations["utilization_percent"] >= 75).sum()),
        "chargers": int(city_stations["chargers"].sum()),
        "available": int(city_stations["available_chargers"].sum()),
        "occupied": int(city_stations["occupied_chargers"].sum()),
        "occupancy": city_stations["predicted_occupancy"].mean() if len(city_stations) else 0,
        "recommended": city_stations.iloc[0]["station_name"] if len(city_stations) else "No station found",
    }


def plot_hourly_occupancy(features: pd.DataFrame, city: str) -> go.Figure:
    hourly = filter_city(features, city).groupby("charging_hour", as_index=False).agg(
        predicted_occupancy=("station_occupancy_percent", "mean"),
        waiting_time=("estimated_waiting_time_minutes", "mean"),
        sessions=("station_key", "count"),
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly["charging_hour"], y=hourly["predicted_occupancy"], mode="lines+markers", name="Occupancy %"))
    fig.add_trace(go.Bar(x=hourly["charging_hour"], y=hourly["waiting_time"], name="Waiting min", opacity=.42, yaxis="y2"))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Hour",
        yaxis_title="Predicted occupancy %",
        yaxis2=dict(title="Waiting minutes", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    return fig


def station_cards(df: pd.DataFrame) -> None:
    for _, row in df.head(30).iterrows():
        status_class = {
            "Available": "status-open",
            "Busy": "status-busy",
            "Overloaded": "status-overloaded",
            "Limited": "status-busy",
        }.get(row["status"], "status-open")
        st.markdown(
            f"""
            <div class="station-row">
              <div style="display:flex; justify-content:space-between; gap:16px; align-items:flex-start;">
                <div>
                  <h4 style="margin:0 0 4px 0;">{row['station_name']}</h4>
                  <div class="small-muted">{row['Charging Station ID']} · {row['Charging Station Location']}</div>
                </div>
                <div class="{status_class}">{row['status']}</div>
              </div>
              <div style="display:grid; grid-template-columns: repeat(6, minmax(110px, 1fr)); gap:12px; margin-top:12px;">
                <div><b>{int(row['current_vehicles_charging'])}</b><div class="small-muted">Vehicles charging</div></div>
                <div><b>{int(row['available_chargers'])}</b><div class="small-muted">Available</div></div>
                <div><b>{int(row['occupied_chargers'])}</b><div class="small-muted">Occupied</div></div>
                <div><b>{row['avg_waiting_minutes']:.1f} min</b><div class="small-muted">Waiting time</div></div>
                <div><b>{row['predicted_occupancy']:.1f}%</b><div class="small-muted">Future occupancy</div></div>
                <div><b>{row['utilization_percent']:.1f}%</b><div class="small-muted">Utilization</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def station_map(stations: pd.DataFrame, city: str, user_coord: tuple[float, float]) -> None:
    city_stations = filter_city(stations, city)
    m = folium.Map(location=user_coord, zoom_start=11, tiles="CartoDB positron")
    cluster = MarkerCluster().add_to(m)
    for _, row in city_stations.iterrows():
        color = "green" if row["status"] == "Available" else "orange" if row["status"] in {"Busy", "Limited"} else "red"
        folium.CircleMarker(
            location=[row["map_latitude"], row["map_longitude"]],
            radius=7 + row["utilization_percent"] / 18,
            color=color,
            fill=True,
            fill_opacity=.72,
            popup=(
                f"<b>{row['station_name']}</b><br>"
                f"Status: {row['status']}<br>"
                f"Occupancy: {row['predicted_occupancy']:.1f}%<br>"
                f"Waiting: {row['avg_waiting_minutes']:.1f} min<br>"
                f"Available chargers: {int(row['available_chargers'])}"
            ),
        ).add_to(cluster)
    folium.Marker(user_coord, tooltip="Selected city center", icon=folium.Icon(color="blue", icon="flash")).add_to(m)
    st_folium(m, use_container_width=True, height=560)


def recommendations(stations: pd.DataFrame, city: str, user_coord: tuple[float, float]) -> pd.DataFrame:
    df = filter_city(stations, city)
    df["distance_km"] = df.apply(lambda r: haversine_km(user_coord[0], user_coord[1], r["map_latitude"], r["map_longitude"]), axis=1)
    df["recommendation_score"] = (
        df["available_chargers"] * 9
        + (100 - df["predicted_occupancy"]) * .55
        + (70 - df["avg_waiting_minutes"]).clip(lower=0) * .75
        + df["avg_rate_kw"].fillna(25) * .16
        - df["distance_km"] * 1.8
    )
    return df.sort_values("recommendation_score", ascending=False)


def forecast_dataframe(features: pd.DataFrame, city: str, horizon: int = 30) -> pd.DataFrame:
    city_df = filter_city(features, city)
    daily = city_df.groupby("charging_month", as_index=False).agg(
        sessions=("station_key", "count"),
        occupancy=("station_occupancy_percent", "mean"),
        waiting=("estimated_waiting_time_minutes", "mean"),
    )
    base_sessions = city_df.groupby("charging_hour").size().mean()
    today = datetime.today().date()
    rows = []
    for i in range(1, horizon + 1):
        date = today + timedelta(days=i)
        weekday_factor = .86 if date.weekday() >= 5 else 1.08
        seasonal = 1 + .12 * math.sin(i / horizon * 2 * math.pi)
        demand = max(1, base_sessions * 24 * weekday_factor * seasonal)
        rows.append({
            "date": date,
            "daily_demand": round(demand, 1),
            "predicted_occupancy": round(city_df["station_occupancy_percent"].mean() * weekday_factor * seasonal, 1),
            "predicted_waiting": round(city_df["estimated_waiting_time_minutes"].mean() * weekday_factor * seasonal, 1),
        })
    return pd.DataFrame(rows)


def ai_insights(stations: pd.DataFrame, features: pd.DataFrame, city: str) -> list[str]:
    city_stations = filter_city(stations, city)
    city_features = filter_city(features, city)
    busiest_hour = int(city_features.groupby("charging_hour").size().idxmax())
    high_risk = city_stations[city_stations["status"].isin(["Busy", "Overloaded"])]
    least_crowded = city_stations.sort_values(["predicted_occupancy", "avg_waiting_minutes"]).head(1)
    best = recommendations(stations, city, CITY_COORDS[city]).head(1)
    insights = [
        f"Peak demand in {city} concentrates around {busiest_hour}:00, so dynamic pricing or reservation controls should target that hour first.",
        f"{len(high_risk)} stations are currently in Busy or Overloaded state and should be prioritized for queue balancing.",
        f"{least_crowded.iloc[0]['station_name']} is the least crowded station based on predicted occupancy and waiting time.",
        f"{best.iloc[0]['station_name']} is the best overall recommendation after balancing availability, speed, queue length, and distance.",
        "Traffic score is a strong operational signal for waiting time, especially during morning and evening commute windows.",
    ]
    return insights


def render_home(stations, features, city, user_coord, metrics) -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>AI-Powered Smart EV Charging Network Optimization Platform</h1>
          <p>Enterprise command center for EV charging demand, live station state, queue intelligence, traffic and weather impact, and AI-assisted user recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    summary = city_summary(stations, features, city)
    metric_row(summary)
    st.markdown("### Selected City Intelligence")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Chargers", f"{summary['chargers']:,}")
    c2.metric("Available Chargers", f"{summary['available']:,}")
    c3.metric("Occupied Chargers", f"{summary['occupied']:,}")
    c4.metric("Predicted Occupancy", f"{summary['occupancy']:.1f}%")
    st.info(f"Recommended station for {city}: {summary['recommended']}")
    left, right = st.columns([1.25, 1])
    with left:
        st.plotly_chart(plot_hourly_occupancy(features, city), use_container_width=True)
    with right:
        if metrics:
            st.markdown("### Model Health")
            st.json(metrics, expanded=False)


def render_nearby(stations, city, user_coord) -> None:
    st.title("Nearby Charging Stations")
    df = recommendations(stations, city, user_coord)
    q = st.text_input("Search stations", "")
    if q:
        df = df[df["station_name"].str.contains(q, case=False, na=False) | df["Charging Station ID"].str.contains(q, case=False, na=False)]
    st.dataframe(
        df[[
            "station_name", "status", "available_chargers", "occupied_chargers", "current_vehicles_charging",
            "avg_waiting_minutes", "distance_km", "predicted_occupancy", "utilization_percent",
        ]].rename(columns={
            "station_name": "Station Name",
            "avg_waiting_minutes": "Estimated Waiting Time",
            "distance_km": "Distance from User",
            "predicted_occupancy": "Future Occupancy",
            "utilization_percent": "Utilization %",
        }).head(100),
        use_container_width=True,
        hide_index=True,
    )
    station_cards(df)


def render_recommendations(stations, city, user_coord) -> None:
    st.title("Smart Recommendation Engine")
    df = recommendations(stations, city, user_coord)
    choices = {
        "Best Charging Station": df.sort_values("recommendation_score", ascending=False).head(1),
        "Least Crowded Station": df.sort_values(["predicted_occupancy", "avg_waiting_minutes"]).head(1),
        "Fastest Charging Station": df.sort_values("avg_rate_kw", ascending=False).head(1),
        "Nearest Station": df.sort_values("distance_km").head(1),
    }
    cols = st.columns(4)
    for col, (label, rec) in zip(cols, choices.items()):
        row = rec.iloc[0]
        col.metric(label, row["station_name"], f"{row['avg_waiting_minutes']:.1f} min wait")
        col.caption(f"{row['available_chargers']} chargers available · {row['distance_km']:.1f} km")
    st.plotly_chart(px.bar(df.head(15), x="recommendation_score", y="station_name", orientation="h", color="status", title="Recommendation Ranking"), use_container_width=True)


def render_occupancy(features, city) -> None:
    st.title("Live Occupancy Prediction")
    st.plotly_chart(plot_hourly_occupancy(features, city), use_container_width=True)
    hourly = filter_city(features, city).groupby("charging_hour", as_index=False).agg(occupancy=("station_occupancy_percent", "mean"))
    busy = hourly[hourly["occupancy"] >= 70]["charging_hour"].tolist()
    free = hourly[hourly["occupancy"] < 45]["charging_hour"].tolist()
    c1, c2 = st.columns(2)
    c1.success("Free time windows: " + ", ".join(f"{h}:00" for h in free[:8]) if free else "No low-occupancy window detected")
    c2.warning("Busy time windows: " + ", ".join(f"{h}:00" for h in busy[:8]) if busy else "No high-occupancy window detected")
    st.plotly_chart(px.imshow(hourly.set_index("charging_hour")[["occupancy"]].T, aspect="auto", color_continuous_scale="RdYlGn_r", title="Hourly Occupancy Heatmap"), use_container_width=True)


def render_waiting(features, city) -> None:
    st.title("Waiting Time Prediction")
    city_df = filter_city(features, city)
    by_station = city_df.groupby("station_key", as_index=False).agg(waiting=("estimated_waiting_time_minutes", "mean"), occupancy=("station_occupancy_percent", "mean"))
    st.plotly_chart(px.scatter(by_station, x="occupancy", y="waiting", title="Queue Analysis: Occupancy vs Waiting Time"), use_container_width=True)
    wait = city_df["estimated_waiting_time_minutes"].mean()
    start = datetime.now() + timedelta(minutes=float(wait))
    st.metric("Expected Waiting Time", f"{wait:.1f} min")
    st.metric("Expected Charging Start Time", start.strftime("%I:%M %p"))


def render_traffic(gold, features, city) -> None:
    st.title("Traffic Analytics")
    traffic = gold["traffic_analytics"]
    traffic = traffic[traffic["Charging Station Location"].eq(city)]
    st.plotly_chart(px.line(traffic, x="charging_hour", y="avg_traffic_score", color="traffic_level", markers=True, title="Traffic Trends by Hour"), use_container_width=True)
    st.plotly_chart(px.bar(traffic, x="traffic_level", y="sessions", color="is_weekend", title="Traffic Impact on Charging Demand"), use_container_width=True)
    city_features = filter_city(features, city)
    st.plotly_chart(px.density_heatmap(city_features, x="traffic_score", y="station_occupancy_percent", title="Peak Traffic Zones and Occupancy"), use_container_width=True)


def render_weather(gold, features, city) -> None:
    st.title("Weather Analytics")
    weather = gold["weather_impact_analytics"]
    weather = weather[weather["Charging Station Location"].eq(city)]
    st.plotly_chart(px.bar(weather, x="weather_condition", y="sessions", color="charging_month", title="Weather Impact on Demand"), use_container_width=True)
    st.plotly_chart(px.scatter(filter_city(features, city), x="temperature_c", y="station_occupancy_percent", color="weather_condition", title="Temperature vs Occupancy"), use_container_width=True)
    st.plotly_chart(px.line(weather.groupby("charging_month", as_index=False).agg(sessions=("sessions", "sum")), x="charging_month", y="sessions", title="Seasonal Demand Pattern"), use_container_width=True)


def render_forecast(features, city) -> None:
    st.title("Demand Forecasting")
    forecast = forecast_dataframe(features, city, 30)
    st.plotly_chart(px.line(forecast, x="date", y=["daily_demand", "predicted_occupancy", "predicted_waiting"], markers=True, title="30-Day Demand Forecast"), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Daily Forecast", f"{forecast.head(1)['daily_demand'].iloc[0]:.0f} sessions")
    c2.metric("Weekly Forecast", f"{forecast.head(7)['daily_demand'].sum():.0f} sessions")
    c3.metric("Monthly Forecast", f"{forecast['daily_demand'].sum():.0f} sessions")


def render_station_performance(stations, city) -> None:
    st.title("Station Performance")
    city_stations = filter_city(stations, city)
    top = city_stations.sort_values(["total_sessions", "total_energy_kwh"], ascending=False).head(15)
    low = city_stations.sort_values(["total_sessions", "utilization_percent"]).head(15)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(top, x="total_sessions", y="station_name", orientation="h", title="Top Performing Stations"), use_container_width=True)
    c2.plotly_chart(px.bar(low, x="total_sessions", y="station_name", orientation="h", title="Low Performing Stations"), use_container_width=True)
    st.plotly_chart(px.scatter(city_stations, x="utilization_percent", y="total_energy_kwh", size="chargers", color="status", hover_name="station_name", title="Utilization Analysis"), use_container_width=True)


def render_ai_insights(stations, features, city) -> None:
    st.title("AI Insights Center")
    for insight in ai_insights(stations, features, city):
        st.markdown(f"<div class='section-card'>{insight}</div>", unsafe_allow_html=True)
    city_features = filter_city(features, city)
    st.plotly_chart(px.box(city_features, x="User Type", y="estimated_waiting_time_minutes", color="Charger Type", title="User Behavior Patterns"), use_container_width=True)


def render_maps(stations, city, user_coord) -> None:
    st.title("Maps")
    station_map(stations, city, user_coord)


def main() -> None:
    silver, features, gold, metrics = load_data()
    database_path = initialize_sqlite(silver, features, gold)
    stations = build_station_state(silver, features)
    cities = sorted(features["Charging Station Location"].dropna().unique().tolist())

    st.sidebar.title("EV Intelligence")
    theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True)
    apply_theme(theme)
    city, user_coord = selected_city_control(cities)
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Pages",
        [
            "Home",
            "Nearby Charging Stations",
            "Smart Recommendation Engine",
            "Live Occupancy Prediction",
            "Waiting Time Prediction",
            "Traffic Analytics",
            "Weather Analytics",
            "Demand Forecasting",
            "Station Performance",
            "AI Insights Center",
            "Maps Page",
        ],
    )
    summary = city_summary(stations, features, city)
    st.sidebar.markdown(
        f"""
        <div class="section-card">
          <b>{city}</b><br>
          <span class="small-muted">{summary['stations']} stations · {summary['available']} chargers available · {summary['waiting']:.1f} min avg wait</span><br>
          <span class="small-muted">SQLite: {Path(database_path).name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Home":
        render_home(stations, features, city, user_coord, metrics)
    elif page == "Nearby Charging Stations":
        render_nearby(stations, city, user_coord)
    elif page == "Smart Recommendation Engine":
        render_recommendations(stations, city, user_coord)
    elif page == "Live Occupancy Prediction":
        render_occupancy(features, city)
    elif page == "Waiting Time Prediction":
        render_waiting(features, city)
    elif page == "Traffic Analytics":
        render_traffic(gold, features, city)
    elif page == "Weather Analytics":
        render_weather(gold, features, city)
    elif page == "Demand Forecasting":
        render_forecast(features, city)
    elif page == "Station Performance":
        render_station_performance(stations, city)
    elif page == "AI Insights Center":
        render_ai_insights(stations, features, city)
    elif page == "Maps Page":
        render_maps(stations, city, user_coord)


if __name__ == "__main__":
    main()
