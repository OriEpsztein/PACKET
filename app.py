import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================================
# 1) Read CSV -> Wide dataframe (index=Timestamp, cols=sensors)
# =========================================================
def _detect_timestamp_column(df: pd.DataFrame) -> str:
    candidates = ["Timestamp", "TimeStamp", "time_stamp", "timestamp",
                  "Datetime", "DateTime", "date_time", "time", "date"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        if "time" in cl or "date" in cl:
            return c
    raise ValueError("Could not detect a timestamp column. Rename it to 'Timestamp'.")

def _to_wide_timeseries(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    sensor_col_candidates = ["Sensor Location", "Sensor", "sensor", "Name", "name", "ID", "id"]
    sensor_col = next((c for c in sensor_col_candidates if c in df.columns), None)

    # WIDE
    if sensor_col is None:
        wide = df.set_index(ts_col).sort_index()
        wide = wide.select_dtypes(include="number")
        if wide.empty:
            wide = df.set_index(ts_col).sort_index().apply(pd.to_numeric, errors="coerce")
        return wide

    # LONG -> pivot
    value_candidates = [c for c in df.columns if c not in [ts_col, sensor_col]]
    numeric_cols = [c for c in value_candidates if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        df["_value_"] = 1.0
        value_col = "_value_"
    else:
        value_col = numeric_cols[0]

    wide = (
        df.pivot_table(index=ts_col, columns=sensor_col, values=value_col, aggfunc="mean")
          .sort_index()
    )
    return wide


# =========================================================
# 2) Packet-loss computation
# =========================================================
def packet_loss_hourly_sensor_matrix(
    file_buffer,
    freq_minutes: int = 3,
    keep_full_hours_only: bool = True
) -> dict:
    if hasattr(file_buffer, 'seek'):
        file_buffer.seek(0)
    
    raw = pd.read_csv(file_buffer)
    ts_col = _detect_timestamp_column(raw)
    wide = _to_wide_timeseries(raw, ts_col)

    if wide.empty:
        raise ValueError("No usable numeric sensor columns found after parsing.")

    sensors = list(wide.columns)
    n_total = len(sensors)
    expected_per_hour = int(round(60 / freq_minutes))

    # Full timestamp grid so missing timestamps count as lost packets
    start = wide.index.min().floor(f"{freq_minutes}min")
    end = wide.index.max().ceil(f"{freq_minutes}min")
    full_ts = pd.date_range(start=start, end=end, freq=f"{freq_minutes}min")
    wide_full = wide.reindex(full_ts)

    hour_idx = wide_full.index.floor("h")
    ts_per_hour = pd.Series(hour_idx).value_counts().sort_index()

    if keep_full_hours_only:
        hours_used = ts_per_hour[ts_per_hour == expected_per_hour].index
    else:
        hours_used = ts_per_hour.index

    wide_use = wide_full.loc[hour_idx.isin(hours_used)]
    if wide_use.empty:
        raise ValueError("No hours left after filtering. Try unchecking 'Show full hours only'.")

    present = wide_use.notna()
    received_hour_sensor = present.groupby(wide_use.index.floor("h")).sum()

    if keep_full_hours_only:
        expected_hour_sensor = pd.DataFrame(
            expected_per_hour,
            index=received_hour_sensor.index,
            columns=received_hour_sensor.columns,
        )
    else:
        ts_per_hour_used = pd.Series(wide_use.index.floor("h")).value_counts().sort_index()
        expected_hour_sensor = pd.DataFrame(
            np.repeat(ts_per_hour_used.values[:, None], n_total, axis=1),
            index=ts_per_hour_used.index,
            columns=sensors,
        ).reindex(received_hour_sensor.index)

    lost_hour_sensor = (expected_hour_sensor - received_hour_sensor).clip(lower=0)
    hourly_sensor_loss = (lost_hour_sensor / expected_hour_sensor) * 100.0

    # Hourly totals (across sensors)
    rec_all = received_hour_sensor.sum(axis=1).astype(int)
    exp_all = expected_hour_sensor.sum(axis=1).astype(int)
    lost_all = (exp_all - rec_all).clip(lower=0).astype(int)
    overall_loss_pct = (lost_all / exp_all.replace(0, np.nan) * 100.0).astype(float)

    hourly_overall = pd.DataFrame({
        "hour": pd.to_datetime(rec_all.index),
        "packets_received": rec_all.values,
        "expected_packets": exp_all.values,
        "lost_packets": lost_all.values,
        "loss_pct": overall_loss_pct.values,
    }).sort_values("hour")

    hourly_sensor_loss = hourly_sensor_loss.reindex(pd.to_datetime(hourly_overall["hour"]))

    stats = pd.DataFrame({
        "mean": hourly_sensor_loss.mean(axis=1),
        "min":  hourly_sensor_loss.min(axis=1),
        "max":  hourly_sensor_loss.max(axis=1),
    }, index=pd.to_datetime(hourly_overall["hour"]))

    return {
        "hourly_overall": hourly_overall,
        "hourly_sensor_loss": hourly_sensor_loss,
        "stats": stats,
        "n_total": n_total,
    }


# =========================================================
# 3) Plot functions
# =========================================================
def plot_hourly_loss(
    rep: dict,
    view_mode: str,
    selected_sensors: list = None,
    show_raw_points: bool = True,
    show_overall_line: bool = False,
    tick_every_hours: int = 4,
    bin_size: float = 0.5
):
    hourly = rep["hourly_overall"]
    loss_mat = rep["hourly_sensor_loss"]
    stats = rep["stats"]
    n_total = rep["n_total"]

    hours_index = pd.to_datetime(hourly["hour"])
    hours = hours_index.to_numpy()

    fig = go.Figure()

    # Helper function to consistently draw the overall line
    def draw_overall_line(marker_size=6):
        y_overall = hourly["loss_pct"].to_numpy(dtype=float)
        custom_line = np.stack([
            hourly["packets_received"].to_numpy(),
            hourly["expected_packets"].to_numpy(),
            hourly["lost_packets"].to_numpy(),
            stats["mean"].to_numpy(dtype=float),
            stats["min"].to_numpy(dtype=float),
            stats["max"].to_numpy(dtype=float),
        ], axis=1)

        fig.add_trace(go.Scatter(
            x=hours, y=y_overall, mode="lines+markers", name="Overall Average",
            line=dict(color="royalblue", width=2), marker=dict(size=marker_size, color="royalblue"),
            customdata=custom_line,
            hovertemplate=(
                "Hour=%{x|%Y-%m-%d %H:%M}<br>Packet Loss (%)=%{y:.2f}<br>"
                "packets_received=%{customdata[0]}<br>expected_packets=%{customdata[1]}<br>"
                "lost_packets=%{customdata[2]}<br>MEAN=%{customdata[3]:.2f}<br>"
                "MIN=%{customdata[4]:.2f}<br>MAX=%{customdata[5]:.2f}<extra></extra>"
            )
        ))

    if view_mode == "Overall Average":
        draw_overall_line(marker_size=6)

    elif view_mode == "Raw Points":
        if show_raw_points:
            rows = []
            for h in hours_index:
                row = loss_mat.loc[h].dropna()
                for v in row.values:
                    v = float(np.clip(v, 0, 100))
                    loss_bin = float(np.round(v / bin_size) * bin_size)
                    rows.append((h, v, loss_bin))

            dfp = pd.DataFrame(rows, columns=["hour", "loss", "loss_bin"])
            dfp["bin_count"] = dfp.groupby(["hour", "loss_bin"])["loss_bin"].transform("size")

            xs = dfp["hour"].to_numpy()
            ys = dfp["loss"].to_numpy(dtype=float)

            cd = np.stack([
                dfp["bin_count"].to_numpy(),
                np.full(len(dfp), n_total),
                dfp["loss_bin"].to_numpy(dtype=float),
            ], axis=1)

            fig.add_trace(go.Scattergl(
                x=xs, y=ys, mode="markers", name="Sensors (raw points)",
                marker=dict(size=7, color="rgba(255,140,0,0.5)"), customdata=cd,
                hovertemplate="Hour=%{x|%Y-%m-%d %H:%M}<br>Packet Loss (%)=%{y:.2f}<br>Sensors=%{customdata[0]} / %{customdata[1]}<extra></extra>"
            ))
            
        if show_overall_line:
            # Draw slightly larger markers to stand out from the raw points
            draw_overall_line(marker_size=10)

    elif view_mode == "Specific Sensors" and selected_sensors:
        for sensor in selected_sensors:
            if sensor in loss_mat.columns:
                y_sensor = loss_mat[sensor].to_numpy(dtype=float)
                fig.add_trace(go.Scatter(
                    x=hours, y=y_sensor, mode="lines+markers", name=f"Sensor {sensor}",
                    hovertemplate="Hour=%{x|%Y-%m-%d %H:%M}<br>Sensor " + str(sensor) + "<br>Packet Loss (%)=%{y:.2f}<extra></extra>"
                ))

    # Formatting
    tick0 = pd.to_datetime(hours_index.min()).floor("d")
    tick_vals = pd.date_range(start=tick0, end=pd.to_datetime(hours_index.max()).ceil("h"), freq=f"{tick_every_hours}h")
    tick_text = [f"{t:%H:%M}<br>{t:%Y-%m-%d}" if t.hour == 0 else f"{t:%H:%M}<br>" for t in tick_vals]

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Hour", yaxis_title="Packet Loss (%)",
        margin=dict(t=30, b=90),
        yaxis=dict(range=[0, 100], autorange=True),
        hovermode="x unified" if view_mode != "Raw Points" else "closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(type="date", tickmode="array", tickvals=tick_vals, ticktext=tick_text, tickangle=0, automargin=True)

    return fig


def sensor_overall_packet_loss(file_buffer, freq_minutes: int = 3, keep_full_hours_only: bool = True) -> pd.DataFrame:
    if hasattr(file_buffer, 'seek'):
        file_buffer.seek(0)
    raw = pd.read_csv(file_buffer)
    ts_col = _detect_timestamp_column(raw)
    wide = _to_wide_timeseries(raw, ts_col)

    if wide.empty:
        raise ValueError("No usable numeric sensor columns found after parsing.")

    sensors = list(wide.columns)
    expected_per_hour = int(round(60 / freq_minutes))

    start = wide.index.min().floor(f"{freq_minutes}min")
    end = wide.index.max().ceil(f"{freq_minutes}min")
    full_ts = pd.date_range(start=start, end=end, freq=f"{freq_minutes}min")
    wide_full = wide.reindex(full_ts)

    hour_idx = wide_full.index.floor("h")
    ts_per_hour = pd.Series(hour_idx).value_counts().sort_index()

    if keep_full_hours_only:
        hours_used = ts_per_hour[ts_per_hour == expected_per_hour].index
    else:
        hours_used = ts_per_hour.index

    wide_use = wide_full.loc[hour_idx.isin(hours_used)]
    present = wide_use.notna()
    received_hour_sensor = present.groupby(wide_use.index.floor("h")).sum()

    if keep_full_hours_only:
        expected_hour_sensor = pd.DataFrame(expected_per_hour, index=received_hour_sensor.index, columns=received_hour_sensor.columns)
    else:
        ts_per_hour_used = pd.Series(wide_use.index.floor("h")).value_counts().sort_index()
        expected_hour_sensor = pd.DataFrame(
            np.repeat(ts_per_hour_used.values[:, None], len(sensors), axis=1),
            index=ts_per_hour_used.index, columns=sensors
        ).reindex(received_hour_sensor.index)

    rec_total = received_hour_sensor.sum(axis=0).astype(int)
    exp_total = expected_hour_sensor.sum(axis=0).astype(int)
    lost_total = (exp_total - rec_total).clip(lower=0).astype(int)
    loss_pct = (lost_total / exp_total.replace(0, np.nan) * 100.0)

    out = pd.DataFrame({
        "sensor": rec_total.index.astype(str),
        "packets_received": rec_total.values,
        "expected_packets": exp_total.values,
        "lost_packets": lost_total.values,
        "loss_pct": loss_pct.values,
    }).dropna(subset=["loss_pct"])

    return out.sort_values("loss_pct")


def plot_sensor_loss_distribution(file_buffer, freq_minutes: int = 3, keep_full_hours_only: bool = True, bin_size: float = 0.5) -> go.Figure:
    df_s = sensor_overall_packet_loss(file_buffer, freq_minutes, keep_full_hours_only)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_s["loss_pct"], xbins=dict(start=0, end=100, size=bin_size),
        hovertemplate="Packet Loss (%)=%{x:.2f}<br>count=%{y}<extra></extra>", name="Sensors"
    ))

    fig.update_layout(
        template="plotly_white", title="Distribution of Sensor Packet Loss (%)",
        xaxis_title="Packet Loss (%)", yaxis_title="Count", bargap=0.05,
    )
    fig.update_xaxes(range=[0, df_s["loss_pct"].max()+1])
    return fig


# =========================================================
# 4) Streamlit App UI
# =========================================================
st.set_page_config(page_title="Field 4D - Packet Loss Analyzer", layout="wide")
st.title("Field 4D: Packet Loss Analysis")

st.sidebar.header("Data Settings")

# Checkbox with help tooltip ('?' icon)
keep_full = st.sidebar.checkbox(
    "Show full hours only", 
    value=True, 
    help="Ignores partial hours at the start or end of your dataset to ensure percentage math is perfectly based on exactly 60 minutes of expected data."
)

FREQ_MINUTES = 3 

uploaded_file = st.file_uploader("Upload Sensor Data File (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Extract and display first/last timestamps
        df_raw = pd.read_csv(uploaded_file)
        ts_col = _detect_timestamp_column(df_raw)
        df_raw[ts_col] = pd.to_datetime(df_raw[ts_col])
        
        first_time = df_raw[ts_col].min()
        last_time = df_raw[ts_col].max()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Dataset Range:**")
        st.sidebar.text(f"Start: {first_time.strftime('%Y-%m-%d %H:%M')}")
        st.sidebar.text(f"End:   {last_time.strftime('%Y-%m-%d %H:%M')}")

        st.success("File loaded and analyzed successfully!")

        # 1. Main Data Computation
        rep = packet_loss_hourly_sensor_matrix(
            uploaded_file, 
            freq_minutes=FREQ_MINUTES, 
            keep_full_hours_only=keep_full
        )

        st.markdown("---")
        st.subheader("Hourly Packet Loss")

        # 2. View Mode Selection
        view_mode = st.radio(
            "Display Mode:", 
            ["Overall Average", "Raw Points", "Specific Sensors"], 
            horizontal=True
        )
        
        # 3. Dynamic Sub-settings based on chosen view
        selected_sensors = []
        show_raw = True
        show_overall = False
        
        if view_mode == "Specific Sensors":
            sensor_list = rep["hourly_sensor_loss"].columns.tolist()
            # Select the first two sensors by default so the chart isn't empty
            default_selection = sensor_list[:2] if len(sensor_list) >= 2 else sensor_list
            selected_sensors = st.multiselect("Select sensors to display:", sensor_list, default=default_selection)
            
        elif view_mode == "Raw Points":
            col1, col2 = st.columns(2)
            with col1:
                show_raw = st.checkbox("Show Raw Sensor Points", value=True)
            with col2:
                show_overall = st.checkbox("Overlay Overall Average Line", value=True)

        # 4. Render Main Plot
        fig_overall = plot_hourly_loss(
            rep, 
            view_mode, 
            selected_sensors=selected_sensors,
            show_raw_points=show_raw,
            show_overall_line=show_overall
        )
        st.plotly_chart(fig_overall, use_container_width=True)
        
        st.markdown("---")
        
        # 5. Render Distribution Plot
        fig_dist = plot_sensor_loss_distribution(
            uploaded_file, 
            freq_minutes=FREQ_MINUTES, 
            keep_full_hours_only=keep_full
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Waiting for a CSV file upload to begin data analysis.")
