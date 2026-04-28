import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# 0) Streamlit page setup
# =========================================================
st.set_page_config(
    page_title="Field 4D - Sensor Analyzer",
    layout="wide",
)

st.title("Field 4D: Sensor Data Analyzer")


# =========================================================
# 1) Fixed app settings
# =========================================================
# You asked to remove this from the UI because the data is always 3 minutes.
FREQ_MINUTES = 3

# Summary / packet-loss problem threshold.
PACKET_LOSS_ALERT_PCT = 5.0

# Default health-check rules.
DEFAULT_BATTERY_LOW_MV = 2700
DEFAULT_BATTERY_LAST_N = 20
DEFAULT_BATTERY_ALLOWED_LOW_COUNT = 3
DEFAULT_STUCK_RUN_THRESHOLD = 4
DEFAULT_STUCK_ROUND_DECIMALS = 2


# =========================================================
# 2) General CSV helpers
# =========================================================
@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    """Read CSV from uploaded bytes.

    Streamlit UploadedFile objects can be consumed after reading.
    Using bytes + cache makes the app safer when the same file is reused
    by several tabs/functions.
    """
    return pd.read_csv(io.BytesIO(file_bytes))


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Return a fresh dataframe from an uploaded Streamlit CSV file."""
    return read_csv_cached(uploaded_file.getvalue()).copy()


def _detect_timestamp_column(df: pd.DataFrame) -> str:
    """Find the timestamp/date column in a flexible way."""
    candidates = [
        "Timestamp", "TimeStamp", "time_stamp", "timestamp",
        "Datetime", "DateTime", "date_time", "Date", "date",
        "time", "Time",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    # Fallback: any column name that contains time/date.
    for c in df.columns:
        cl = str(c).lower()
        if "time" in cl or "date" in cl:
            return c

    raise ValueError("Could not detect a timestamp column. Rename it to 'Timestamp'.")


def _detect_sensor_column(df: pd.DataFrame):
    """Detect sensor/name/id column for long-format CSVs.

    If this returns None, the file is treated as wide format:
    Timestamp + one numeric column per sensor.
    """
    sensor_col_candidates = [
        "Sensor Location", "Sensor", "sensor",
        "Name", "name",
        "ID", "id",
        "sensor_id", "Sensor_ID",
    ]

    for c in sensor_col_candidates:
        if c in df.columns:
            return c

    return None


def detect_data_type(file_name: str, df: pd.DataFrame = None) -> str:
    """Auto-detect whether the uploaded CSV is Battery / Temperature / Light / Other."""
    name = str(file_name).lower()

    if any(x in name for x in ["battery", "batt", "bat", "mv"]):
        return "Battery"
    if any(x in name for x in ["temperature", "temp"]):
        return "Temperature"
    if any(x in name for x in ["light", "par", "lux", "radiation"]):
        return "Light"

    if df is not None:
        cols = " ".join([str(c).lower() for c in df.columns])
        if any(x in cols for x in ["battery", "batt", "battery_mv", "mv"]):
            return "Battery"
        if any(x in cols for x in ["temperature", "temp"]):
            return "Temperature"
        if any(x in cols for x in ["light", "par", "lux", "radiation"]):
            return "Light"

    return "Other"


def _value_candidates_for_type(data_type: str) -> list:
    """Preferred value columns for long-format files."""
    if data_type == "Battery":
        return [
            "battery", "Battery", "battery_mv", "Battery_mV", "batt",
            "BATT", "VBAT", "vbat", "mV", "mv",
        ]

    if data_type == "Temperature":
        return [
            "temperature", "Temperature", "temp", "Temp", "TEMP",
            "temperature_c", "Temperature (°C)", "Temperature_C",
        ]

    if data_type == "Light":
        return [
            "light", "Light", "LIGHT", "PARlight", "PAR", "par",
            "lux", "Lux", "radiation", "Radiation",
        ]

    return []


def _choose_value_column(df: pd.DataFrame, ignore_cols: list, data_type: str = None) -> str:
    """Choose the value column for long-format data."""
    if data_type is not None:
        for c in _value_candidates_for_type(data_type):
            if c in df.columns and c not in ignore_cols:
                return c

    possible_cols = [c for c in df.columns if c not in ignore_cols]

    # Prefer columns already read by pandas as numeric.
    numeric_cols = [c for c in possible_cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[0]

    # Fallback: try converting columns to numeric and keep the one with most valid numbers.
    best_col = None
    best_count = -1

    for c in possible_cols:
        converted = pd.to_numeric(df[c], errors="coerce")
        count = int(converted.notna().sum())

        if count > best_count:
            best_col = c
            best_count = count

    if best_col is None or best_count == 0:
        raise ValueError("Could not detect a numeric value column for analysis.")

    return best_col


def _to_wide_timeseries(df: pd.DataFrame, ts_col: str, data_type: str = None) -> pd.DataFrame:
    """Convert wide or long sensor data into wide time-series format.

    Output format:
        index   = Timestamp
        columns = sensors
        values  = numeric measurement
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    if df.empty:
        raise ValueError("No valid timestamps after parsing.")

    sensor_col = _detect_sensor_column(df)

    # -----------------------------------------------------
    # WIDE FORMAT:
    # Timestamp | sensor_1 | sensor_2 | sensor_3 | ...
    # -----------------------------------------------------
    if sensor_col is None:
        wide = df.set_index(ts_col).sort_index()
        wide = wide.apply(pd.to_numeric, errors="coerce")
        wide = wide.dropna(axis=1, how="all")

        # If the CSV has duplicate timestamps, average them before reindexing.
        wide = wide.groupby(wide.index).mean()

        if wide.empty:
            raise ValueError("No usable numeric sensor columns found after parsing.")

        return wide

    # -----------------------------------------------------
    # LONG FORMAT:
    # Timestamp | Sensor/Name | value
    # -----------------------------------------------------
    value_col = _choose_value_column(
        df,
        ignore_cols=[ts_col, sensor_col],
        data_type=data_type,
    )

    temp = df[[ts_col, sensor_col, value_col]].copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna(subset=[value_col])

    if temp.empty:
        raise ValueError("No numeric values after parsing the selected value column.")

    wide = (
        temp.pivot_table(
            index=ts_col,
            columns=sensor_col,
            values=value_col,
            aggfunc="mean",
        )
        .sort_index()
        .dropna(axis=1, how="all")
    )

    if wide.empty:
        raise ValueError("No usable sensor columns after pivoting long-format data.")

    return wide


def get_basic_file_info(uploaded_file) -> dict:
    """Return simple file metadata for display."""
    raw = read_uploaded_csv(uploaded_file)
    ts_col = _detect_timestamp_column(raw)
    wide = _to_wide_timeseries(raw, ts_col)

    return {
        "file": uploaded_file.name,
        "rows": len(raw),
        "sensors": len(wide.columns),
        "timestamp_col": ts_col,
        "start": pd.to_datetime(wide.index.min()),
        "end": pd.to_datetime(wide.index.max()),
        "auto_type": detect_data_type(uploaded_file.name, raw),
    }


# =========================================================
# 3) Packet-loss computation
# =========================================================
def _build_full_timestamp_grid(wide: pd.DataFrame, freq_minutes: int = FREQ_MINUTES) -> pd.DatetimeIndex:
    """Create the expected timestamp grid from the first timestamp to the last timestamp.

    Important change:
    - We do NOT remove partial hours.
    - Partial hours are calculated according to how many 3-minute timestamps should exist
      inside that partial time window.

    Example:
    If the file starts at 10:15 and ends at 12:42, the 10:00 hour is expected to have
    only the timestamps from 10:15, 10:18, ... 10:57. It is not forced to 20.
    """
    start = pd.to_datetime(wide.index.min())
    end = pd.to_datetime(wide.index.max())

    if pd.isna(start) or pd.isna(end):
        raise ValueError("Could not build timestamp grid because start/end time is missing.")

    return pd.date_range(start=start, end=end, freq=f"{freq_minutes}min")


def packet_loss_hourly_sensor_matrix(df: pd.DataFrame, freq_minutes: int = FREQ_MINUTES) -> dict:
    """Compute hourly packet loss per sensor and overall.

    This version always uses 3-minute sampling and always keeps partial hours.
    For partial hours, expected packets are based on the real number of expected
    3-minute timestamps in that partial hour.
    """
    ts_col = _detect_timestamp_column(df)
    wide = _to_wide_timeseries(df, ts_col)

    sensors = list(wide.columns)
    n_total = len(sensors)

    full_ts = _build_full_timestamp_grid(wide, freq_minutes=freq_minutes)
    wide_full = wide.reindex(full_ts)

    # Count how many expected timestamps exist in each hour.
    hour_idx = wide_full.index.floor("h")
    expected_timestamps_per_hour = pd.Series(hour_idx).value_counts().sort_index()

    present = wide_full.notna()
    received_hour_sensor = present.groupby(wide_full.index.floor("h")).sum()

    expected_hour_sensor = pd.DataFrame(
        np.repeat(expected_timestamps_per_hour.values[:, None], n_total, axis=1),
        index=expected_timestamps_per_hour.index,
        columns=sensors,
    ).reindex(received_hour_sensor.index)

    lost_hour_sensor = (expected_hour_sensor - received_hour_sensor).clip(lower=0)
    hourly_sensor_loss = (lost_hour_sensor / expected_hour_sensor.replace(0, np.nan)) * 100.0

    # Hourly totals across all sensors.
    rec_all = received_hour_sensor.sum(axis=1).astype(int)
    exp_all = expected_hour_sensor.sum(axis=1).astype(int)
    lost_all = (exp_all - rec_all).clip(lower=0).astype(int)
    overall_loss_pct = lost_all / exp_all.replace(0, np.nan) * 100.0

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
        "min": hourly_sensor_loss.min(axis=1),
        "max": hourly_sensor_loss.max(axis=1),
    }, index=pd.to_datetime(hourly_overall["hour"]))

    return {
        "hourly_overall": hourly_overall,
        "hourly_sensor_loss": hourly_sensor_loss,
        "stats": stats,
        "n_total": n_total,
        "freq_minutes": freq_minutes,
    }


def sensor_overall_packet_loss(df: pd.DataFrame, freq_minutes: int = FREQ_MINUTES) -> pd.DataFrame:
    """Overall packet loss per sensor across the file.

    Partial first/last hours are included using the expected number of timestamps
    between the first timestamp and last timestamp.
    """
    ts_col = _detect_timestamp_column(df)
    wide = _to_wide_timeseries(df, ts_col)

    sensors = list(wide.columns)
    n_total = len(sensors)

    full_ts = _build_full_timestamp_grid(wide, freq_minutes=freq_minutes)
    wide_full = wide.reindex(full_ts)

    hour_idx = wide_full.index.floor("h")
    expected_timestamps_per_hour = pd.Series(hour_idx).value_counts().sort_index()

    present = wide_full.notna()
    received_hour_sensor = present.groupby(wide_full.index.floor("h")).sum()

    expected_hour_sensor = pd.DataFrame(
        np.repeat(expected_timestamps_per_hour.values[:, None], n_total, axis=1),
        index=expected_timestamps_per_hour.index,
        columns=sensors,
    ).reindex(received_hour_sensor.index)

    rec_total = received_hour_sensor.sum(axis=0).astype(int)
    exp_total = expected_hour_sensor.sum(axis=0).astype(int)
    lost_total = (exp_total - rec_total).clip(lower=0).astype(int)
    loss_pct = lost_total / exp_total.replace(0, np.nan) * 100.0

    out = pd.DataFrame({
        "sensor": rec_total.index.astype(str),
        "packets_received": rec_total.values,
        "expected_packets": exp_total.values,
        "lost_packets": lost_total.values,
        "loss_pct": loss_pct.values,
    }).dropna(subset=["loss_pct"])

    return out.sort_values("loss_pct", ascending=False)


# =========================================================
# 4) Packet-loss plot functions
# =========================================================
def plot_hourly_loss_combined(
    rep: dict,
    show_raw_points: bool = False,
    show_specific_sensors: bool = False,
    selected_sensors: list = None,
    bin_size: float = 0.5,
) -> go.Figure:
    """Hourly packet-loss plot.

    Always shows the overall average.
    Optional overlays:
    - raw sensor points
    - selected specific sensor lines
    """
    hourly = rep["hourly_overall"]
    loss_mat = rep["hourly_sensor_loss"]
    stats = rep["stats"]
    n_total = rep["n_total"]

    hours_index = pd.to_datetime(hourly["hour"])
    hours = hours_index.to_numpy()

    fig = go.Figure()

    # -------------------------
    # Overall average line
    # -------------------------
    custom_line = np.stack([
        hourly["packets_received"].to_numpy(),
        hourly["expected_packets"].to_numpy(),
        hourly["lost_packets"].to_numpy(),
        stats["mean"].to_numpy(dtype=float),
        stats["min"].to_numpy(dtype=float),
        stats["max"].to_numpy(dtype=float),
    ], axis=1)

    fig.add_trace(go.Scatter(
        x=hours,
        y=hourly["loss_pct"].to_numpy(dtype=float),
        mode="lines+markers",
        name="Overall Average",
        line=dict(color="royalblue", width=2),
        marker=dict(size=6, color="royalblue"),
        customdata=custom_line,
        hovertemplate=(
            "Hour: %{x|%Y-%m-%d %H:00}<br>"
            "Packet Loss (%)=%{y:.2f}<br>"
            "packets_received=%{customdata[0]}<br>"
            "expected_packets=%{customdata[1]}<br>"
            "lost_packets=%{customdata[2]}<br>"
            "MEAN sensor loss=%{customdata[3]:.2f}<br>"
            "MIN sensor loss=%{customdata[4]:.2f}<br>"
            "MAX sensor loss=%{customdata[5]:.2f}<extra></extra>"
        ),
    ))

    # -------------------------
    # Optional raw points
    # -------------------------
    if show_raw_points:
        rows = []

        for h in hours_index:
            row = loss_mat.loc[h].dropna()

            for sensor_name, value in row.items():
                value = float(np.clip(value, 0, 100))
                loss_bin = float(np.round(value / bin_size) * bin_size)
                rows.append((h, str(sensor_name), value, loss_bin))

        raw_points_df = pd.DataFrame(rows, columns=["hour", "sensor", "loss", "loss_bin"])

        if not raw_points_df.empty:
            raw_points_df["bin_count"] = raw_points_df.groupby(["hour", "loss_bin"])["loss_bin"].transform("size")

            custom_points = np.stack([
                raw_points_df["sensor"].astype(str).to_numpy(),
                raw_points_df["bin_count"].to_numpy(),
                np.full(len(raw_points_df), n_total),
                raw_points_df["loss_bin"].to_numpy(dtype=float),
            ], axis=1)

            # Split raw points into two traces so hover text can be cleaner:
            # - if only one sensor is represented by that hour/loss-bin, show the sensor name
            # - if several sensors overlap in the same hour/loss-bin, do NOT show one random sensor name
            single_sensor_points = raw_points_df[raw_points_df["bin_count"] == 1].copy()
            multi_sensor_points = raw_points_df[raw_points_df["bin_count"] > 1].copy()

            if not single_sensor_points.empty:
                custom_single = np.stack([
                    single_sensor_points["sensor"].astype(str).to_numpy(),
                    single_sensor_points["bin_count"].to_numpy(),
                    np.full(len(single_sensor_points), n_total),
                    single_sensor_points["loss_bin"].to_numpy(dtype=float),
                ], axis=1)

                fig.add_trace(go.Scattergl(
                    x=single_sensor_points["hour"].to_numpy(),
                    y=single_sensor_points["loss"].to_numpy(dtype=float),
                    mode="markers",
                    name="Raw Sensor Point - single sensor",
                    marker=dict(size=7, color="rgba(255,140,0,0.45)"),
                    customdata=custom_single,
                    hovertemplate=(
                        "Hour: %{x|%Y-%m-%d %H:00}<br>"
                        "Sensor=%{customdata[0]}<br>"
                        "Packet Loss (%)=%{y:.2f}<br>"
                        "Sensors in same bin=%{customdata[1]} / %{customdata[2]}<extra></extra>"
                    ),
                ))

            if not multi_sensor_points.empty:
                custom_multi = np.stack([
                    multi_sensor_points["bin_count"].to_numpy(),
                    np.full(len(multi_sensor_points), n_total),
                    multi_sensor_points["loss_bin"].to_numpy(dtype=float),
                ], axis=1)

                fig.add_trace(go.Scattergl(
                    x=multi_sensor_points["hour"].to_numpy(),
                    y=multi_sensor_points["loss"].to_numpy(dtype=float),
                    mode="markers",
                    name="Raw Sensor Points - grouped",
                    marker=dict(size=7, color="rgba(255,140,0,0.45)"),
                    customdata=custom_multi,
                    hovertemplate=(
                        "Hour: %{x|%Y-%m-%d %H:00}<br>"
                        "Packet Loss (%)=%{y:.2f}<br>"
                        "Sensors in same bin=%{customdata[0]} / %{customdata[1]}<extra></extra>"
                    ),
                ))

    # -------------------------
    # Optional specific sensors
    # -------------------------
    if show_specific_sensors and selected_sensors:
        for sensor in selected_sensors:
            if sensor in loss_mat.columns:
                fig.add_trace(go.Scatter(
                    x=hours,
                    y=loss_mat[sensor].to_numpy(dtype=float),
                    mode="lines+markers",
                    name=f"Sensor {sensor}",
                    hovertemplate=(
                        "Hour: %{x|%Y-%m-%d %H:00}<br>"
                        "Sensor: " + str(sensor) + "<br>"
                        "Specific sensor packet loss (%)=%{y:.2f}<extra></extra>"
                    ),
                ))

    fig.update_layout(
        template="plotly_white",
        title="Hourly Packet Loss",
        xaxis_title="Hour Start Time",
        yaxis_title="Packet Loss (%)",
        margin=dict(t=55, b=90),
        yaxis=dict(rangemode="tozero"),
        hovermode="closest" if (show_raw_points or show_specific_sensors) else "x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )

    fig.update_xaxes(type="date", automargin=True)

    return fig


def plot_hourly_specific_sensors(
    rep: dict,
    selected_sensors: list,
) -> go.Figure:
    """Hourly packet-loss plot for selected sensors only."""
    loss_mat = rep["hourly_sensor_loss"]
    hours_index = pd.to_datetime(loss_mat.index)

    fig = go.Figure()

    for sensor in selected_sensors:
        if sensor not in loss_mat.columns:
            continue

        fig.add_trace(go.Scatter(
            x=hours_index,
            y=loss_mat[sensor].to_numpy(dtype=float),
            mode="lines+markers",
            name=f"Sensor {sensor}",
            hovertemplate=(
                "Hour: %{x|%Y-%m-%d %H:00}<br>"
                "Sensor " + str(sensor) + "<br>"
                "Packet Loss (%)=%{y:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_white",
        title="Hourly Packet Loss - Specific Sensors",
        xaxis_title="Hour Start Time",
        yaxis_title="Packet Loss (%)",
        margin=dict(t=55, b=90),
        yaxis=dict(rangemode="tozero"),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )

    fig.update_xaxes(type="date", automargin=True)

    return fig


def plot_sensor_loss_distribution(df_s: pd.DataFrame, bin_size: float = 0.5) -> go.Figure:
    """Histogram of sensor-level packet loss."""
    fig = go.Figure()

    if df_s.empty:
        fig.update_layout(
            template="plotly_white",
            title="Distribution of Sensor Packet Loss (%)",
            xaxis_title="Packet Loss (%)",
            yaxis_title="Count",
        )
        return fig

    max_x = max(1.0, float(df_s["loss_pct"].max()) + 1.0)

    fig.add_trace(go.Histogram(
        x=df_s["loss_pct"],
        xbins=dict(start=0, end=100, size=bin_size),
        hovertemplate="Packet Loss (%)=%{x:.2f}<br>count=%{y}<extra></extra>",
        name="Sensors",
    ))

    fig.update_layout(
        template="plotly_white",
        title="Distribution of Sensor Packet Loss (%)",
        xaxis_title="Packet Loss (%)",
        yaxis_title="Count",
        bargap=0.05,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )
    fig.update_xaxes(range=[0, max_x])

    return fig


# =========================================================
# 5) Data-analysis health checks
# =========================================================
def analyze_battery(
    wide: pd.DataFrame,
    low_mv_threshold: float = DEFAULT_BATTERY_LOW_MV,
    last_n: int = DEFAULT_BATTERY_LAST_N,
    low_count_limit: int = DEFAULT_BATTERY_ALLOWED_LOW_COUNT,
) -> pd.DataFrame:
    """Battery health rule.

    Requested rule:
    - Take last 20 values.
    - If more than 3 values are under 2700 mV -> LOW_BATTERY.
    - If there are 3 or fewer timestamps to check -> flag if at least 1 value is under 2700 mV.
    """
    rows = []

    for sensor in wide.columns:
        s = pd.to_numeric(wide[sensor], errors="coerce").dropna().sort_index()
        last_values = s.tail(last_n)

        n_checked = int(len(last_values))
        under_count = int((last_values < low_mv_threshold).sum())

        if n_checked == 0:
            status = "NO_DATA"
            issue = True
            rule_used = "No values available"
        elif n_checked <= low_count_limit:
            issue = under_count >= 1
            status = "LOW_BATTERY" if issue else "OK"
            rule_used = f"Only {n_checked} timestamp(s): flag if at least 1 value < {low_mv_threshold:.0f} mV"
        else:
            issue = under_count > low_count_limit
            status = "LOW_BATTERY" if issue else "OK"
            rule_used = f"Last {last_n}: flag if more than {low_count_limit} values < {low_mv_threshold:.0f} mV"

        low_times = last_values[last_values < low_mv_threshold]

        rows.append({
            "sensor": str(sensor),
            "status": status,
            "has_issue": bool(issue),
            "values_checked": n_checked,
            "under_threshold_count": under_count,
            "threshold_mV": low_mv_threshold,
            "last_value_mV": float(last_values.iloc[-1]) if n_checked else np.nan,
            "min_last_values_mV": float(last_values.min()) if n_checked else np.nan,
            "first_low_time_in_last_values": low_times.index.min() if not low_times.empty else pd.NaT,
            "last_low_time_in_last_values": low_times.index.max() if not low_times.empty else pd.NaT,
            "rule_used": rule_used,
        })

    return pd.DataFrame(rows).sort_values(
        by=["has_issue", "under_threshold_count", "min_last_values_mV"],
        ascending=[False, False, True],
    )


def longest_equal_run(series: pd.Series, decimals: int = DEFAULT_STUCK_ROUND_DECIMALS, ignore_values: set = None) -> dict:
    """Find the longest consecutive run of equal values.

    Example: 23, 23, 23, 23 -> run length 4.
    Rounding helps avoid tiny floating-point differences.
    """
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()

    best_len = 0
    best_value = np.nan
    best_start = pd.NaT
    best_end = pd.NaT

    current_len = 0
    current_value = None
    current_start = pd.NaT
    current_end = pd.NaT

    for timestamp, value in s.items():
        rounded_value = round(float(value), decimals)

        # Values that should not count as stuck, for example light=0 at night.
        if ignore_values is not None and rounded_value in ignore_values:
            current_len = 0
            current_value = None
            current_start = pd.NaT
            current_end = pd.NaT
            continue

        if current_len == 0 or rounded_value != current_value:
            current_len = 1
            current_value = rounded_value
            current_start = timestamp
            current_end = timestamp
        else:
            current_len += 1
            current_end = timestamp

        if current_len > best_len:
            best_len = current_len
            best_value = current_value
            best_start = current_start
            best_end = current_end

    return {
        "max_stuck_run": int(best_len),
        "stuck_value": best_value,
        "stuck_start_time": best_start,
        "stuck_end_time": best_end,
    }


def analyze_temperature(
    wide: pd.DataFrame,
    stuck_run_threshold: int = DEFAULT_STUCK_RUN_THRESHOLD,
    stuck_round_decimals: int = DEFAULT_STUCK_ROUND_DECIMALS,
    bad_temp_value: float = -40,
) -> pd.DataFrame:
    """Temperature checks: -40 values and stuck repeated values."""
    rows = []

    for sensor in wide.columns:
        s = pd.to_numeric(wide[sensor], errors="coerce").dropna().sort_index()

        bad_temp_mask = s <= (bad_temp_value + 0.1)
        bad_temp_count = int(bad_temp_mask.sum())

        run_info = longest_equal_run(s, decimals=stuck_round_decimals)
        is_stuck = run_info["max_stuck_run"] >= stuck_run_threshold

        issues = []
        if bad_temp_count > 0:
            issues.append("TEMP_-40")
        if is_stuck:
            issues.append("STUCK_VALUE")

        has_issue = len(issues) > 0

        rows.append({
            "sensor": str(sensor),
            "status": "OK" if not has_issue else " / ".join(issues),
            "has_issue": has_issue,
            "values_count": int(len(s)),
            "minus_40_count": bad_temp_count,
            "first_minus_40_time": s[bad_temp_mask].index.min() if bad_temp_count else pd.NaT,
            "last_minus_40_time": s[bad_temp_mask].index.max() if bad_temp_count else pd.NaT,
            "max_stuck_run": run_info["max_stuck_run"],
            "stuck_value": run_info["stuck_value"],
            "stuck_start_time": run_info["stuck_start_time"],
            "stuck_end_time": run_info["stuck_end_time"],
            "last_value": float(s.iloc[-1]) if len(s) else np.nan,
        })

    return pd.DataFrame(rows).sort_values(
        by=["has_issue", "minus_40_count", "max_stuck_run"],
        ascending=[False, False, False],
    )


def analyze_light(wide: pd.DataFrame) -> pd.DataFrame:
    """Light check.

    Current requested rule:
    - Do NOT run stuck-value detection for light.
    - Stuck-value detection is only for temperature.

    This function still returns a clean per-sensor table so the Light CSV can be
    previewed in Data Analysis, but it does not mark light sensors as issues.
    """
    rows = []

    for sensor in wide.columns:
        s = pd.to_numeric(wide[sensor], errors="coerce").dropna().sort_index()

        rows.append({
            "sensor": str(sensor),
            "status": "OK",
            "has_issue": False,
            "values_count": int(len(s)),
            "last_value": float(s.iloc[-1]) if len(s) else np.nan,
            "note": "No stuck-value check for Light. Stuck-value check is only for Temperature.",
        })

    return pd.DataFrame(rows).sort_values(by=["sensor"])


def run_data_health_check(
    df: pd.DataFrame,
    data_type: str,
    battery_threshold_mv: float = DEFAULT_BATTERY_LOW_MV,
    battery_last_n: int = DEFAULT_BATTERY_LAST_N,
    battery_low_count_limit: int = DEFAULT_BATTERY_ALLOWED_LOW_COUNT,
    stuck_run_threshold: int = DEFAULT_STUCK_RUN_THRESHOLD,
    stuck_round_decimals: int = DEFAULT_STUCK_ROUND_DECIMALS,
) -> pd.DataFrame:
    """Run the correct health check according to selected data type."""
    ts_col = _detect_timestamp_column(df)
    wide = _to_wide_timeseries(df, ts_col, data_type=data_type)

    if data_type == "Battery":
        return analyze_battery(
            wide,
            low_mv_threshold=battery_threshold_mv,
            last_n=battery_last_n,
            low_count_limit=battery_low_count_limit,
        )

    if data_type == "Temperature":
        return analyze_temperature(
            wide,
            stuck_run_threshold=stuck_run_threshold,
            stuck_round_decimals=stuck_round_decimals,
        )

    if data_type == "Light":
        return analyze_light(wide)

    return pd.DataFrame()


def count_data_issues(result_df: pd.DataFrame) -> int:
    """Count rows that have an issue."""
    if result_df.empty:
        return 0

    if "has_issue" in result_df.columns:
        return int(result_df["has_issue"].sum())

    if "status" in result_df.columns:
        return int((result_df["status"] != "OK").sum())

    return 0


def result_issues_only(result_df: pd.DataFrame) -> pd.DataFrame:
    """Return only problematic sensors from a health-check dataframe."""
    if result_df.empty:
        return result_df

    if "has_issue" in result_df.columns:
        return result_df[result_df["has_issue"]].copy()

    if "status" in result_df.columns:
        return result_df[result_df["status"] != "OK"].copy()

    return result_df.iloc[0:0].copy()


# =========================================================
# 6) Extra plotting for data analysis
# =========================================================
def plot_last_values_for_sensors(
    wide: pd.DataFrame,
    sensors: list,
    title: str,
    last_n: int = 20,
) -> go.Figure:
    """Plot last N values for selected sensors."""
    fig = go.Figure()

    for sensor in sensors:
        if sensor not in wide.columns:
            continue

        s = pd.to_numeric(wide[sensor], errors="coerce").dropna().sort_index().tail(last_n)

        if s.empty:
            continue

        fig.add_trace(go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines+markers",
            name=str(sensor),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>value=%{y}<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Timestamp",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )

    return fig


# =========================================================
# 7) Sidebar - Data Settings
# =========================================================
st.sidebar.header("Data Settings")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV / CSVs",
    type=["csv"],
    accept_multiple_files=True,
    help="Upload one or several CSV files. Packet Loss uses the first CSV automatically.",
)

if not uploaded_files:
    st.info("Upload one or more CSV files from the left sidebar to begin analysis.")
    st.stop()

# No sidebar file selector: packet loss always uses the first CSV.
packet_loss_file = uploaded_files[0]
packet_loss_df = read_uploaded_csv(packet_loss_file)

# =========================================================
# 8) Pre-compute packet loss for the first CSV
# =========================================================
packet_rep = None
packet_sensor_loss_df = pd.DataFrame()
packet_problem_df = pd.DataFrame()
packet_error = None

try:
    packet_rep = packet_loss_hourly_sensor_matrix(packet_loss_df, freq_minutes=FREQ_MINUTES)
    packet_sensor_loss_df = sensor_overall_packet_loss(packet_loss_df, freq_minutes=FREQ_MINUTES)
    packet_problem_df = packet_sensor_loss_df[
        packet_sensor_loss_df["loss_pct"] > PACKET_LOSS_ALERT_PCT
    ].copy()
except Exception as e:
    packet_error = e


# =========================================================
# 9) Pre-compute value issues for summary, using default rules
# =========================================================
summary_value_issue_rows = []
summary_file_rows = []

for i, file in enumerate(uploaded_files, start=1):
    row = {
        "file_number": i,
        "file": file.name,
        "rows": np.nan,
        "sensors": np.nan,
        "start": pd.NaT,
        "end": pd.NaT,
        "auto_type": "Unknown",
        "value_issues_count": np.nan,
        "status": "OK",
    }

    try:
        raw = read_uploaded_csv(file)
        ts_col = _detect_timestamp_column(raw)
        wide = _to_wide_timeseries(raw, ts_col)
        auto_type = detect_data_type(file.name, raw)

        row["rows"] = len(raw)
        row["sensors"] = len(wide.columns)
        row["start"] = pd.to_datetime(wide.index.min())
        row["end"] = pd.to_datetime(wide.index.max())
        row["auto_type"] = auto_type

        if auto_type in ["Battery", "Temperature", "Light"]:
            health_df = run_data_health_check(
                raw,
                data_type=auto_type,
                battery_threshold_mv=DEFAULT_BATTERY_LOW_MV,
                battery_last_n=DEFAULT_BATTERY_LAST_N,
                battery_low_count_limit=DEFAULT_BATTERY_ALLOWED_LOW_COUNT,
                stuck_run_threshold=DEFAULT_STUCK_RUN_THRESHOLD,
                stuck_round_decimals=DEFAULT_STUCK_ROUND_DECIMALS,
            )
            issue_df = result_issues_only(health_df)
            row["value_issues_count"] = len(issue_df)

            for _, issue_row in issue_df.iterrows():
                summary_value_issue_rows.append({
                    "problem_type": "VALUE_ISSUE",
                    "file": file.name,
                    "data_type": auto_type,
                    "sensor": issue_row.get("sensor", ""),
                    "issue": issue_row.get("status", ""),
                    "loss_pct": np.nan,
                    "details": " | ".join([
                        f"values_checked={issue_row.get('values_checked', issue_row.get('values_count', ''))}",
                        f"under_threshold_count={issue_row.get('under_threshold_count', '')}",
                        f"minus_40_count={issue_row.get('minus_40_count', '')}",
                        f"max_stuck_run={issue_row.get('max_stuck_run', '')}",
                        f"stuck_value={issue_row.get('stuck_value', '')}",
                    ]),
                })
        else:
            row["value_issues_count"] = np.nan

    except Exception as e:
        row["status"] = f"ERROR: {e}"

    summary_file_rows.append(row)

summary_files_df = pd.DataFrame(summary_file_rows)
summary_value_issues_df = pd.DataFrame(summary_value_issue_rows)


# =========================================================
# 10) Tabs
# =========================================================
tab_summary, tab_packet_loss, tab_data_analysis = st.tabs([
    "Summary",
    "Packet Loss Analysis",
    "Data Analysis",
])


# =========================================================
# 10A) Summary tab
# =========================================================
with tab_summary:
    st.subheader("Summary")

    # -----------------------------------------------------
    # Top-level summary cards
    # -----------------------------------------------------
    sensors_above_5_count = 0 if packet_problem_df.empty else len(packet_problem_df)
    total_value_issues = int(
        pd.to_numeric(summary_files_df["value_issues_count"], errors="coerce")
        .fillna(0)
        .sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Uploaded files", len(uploaded_files))

    with c2:
        st.metric("Packet-loss CSV", packet_loss_file.name)

    with c3:
        if packet_error is not None:
            st.metric("Sensors > 5% loss", "-")
        else:
            st.metric("Sensors > 5% loss", sensors_above_5_count)

    with c4:
        st.metric("Value issue sensors", total_value_issues)

    # -----------------------------------------------------
    # Distribution chart
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### Distribution of Sensor Packet Loss (%)")

    if packet_error is not None:
        st.error(f"Could not calculate packet loss from the first CSV: {packet_error}")
    else:
        st.caption(f"Packet loss is calculated from the first uploaded CSV: **{packet_loss_file.name}**")
        st.plotly_chart(
            plot_sensor_loss_distribution(packet_sensor_loss_df),
            use_container_width=True,
            key="summary_packet_loss_distribution",
        )

    # -----------------------------------------------------
    # Sensors above 5% packet loss: number + names
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### Sensors > 5% Packet Loss")

    if packet_error is not None:
        st.warning("Packet-loss sensors could not be listed because packet-loss calculation failed.")
    elif packet_problem_df.empty:
        st.success("✅ 0 sensors above 5% packet loss.")
    else:
        problem_names = packet_problem_df["sensor"].astype(str).tolist()

        st.error(f"🚨 {len(problem_names)} sensor(s) above 5% packet loss.")
        st.markdown("**Sensor names:**")
        st.write(", ".join(problem_names))

        packet_summary_display = (
            packet_problem_df[[
                "sensor", "loss_pct", "lost_packets", "expected_packets", "packets_received"
            ]]
            .sort_values("loss_pct", ascending=False)
            .rename(columns={
                "sensor": "Sensor",
                "loss_pct": "Loss (%)",
                "lost_packets": "Lost Packets",
                "expected_packets": "Expected Packets",
                "packets_received": "Received Packets",
            })
        )

        st.dataframe(
            packet_summary_display.style.format({"Loss (%)": "{:.2f}%"}),
            hide_index=True,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # Value issue sensors: sensor + issue
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### Value Issue Sensors")

    if summary_value_issues_df.empty:
        st.success("✅ No battery / temperature / light value issues detected.")
    else:
        value_issue_display = (
            summary_value_issues_df[["file", "data_type", "sensor", "issue", "details"]]
            .rename(columns={
                "file": "File",
                "data_type": "Data Type",
                "sensor": "Sensor",
                "issue": "Issue",
                "details": "Details",
            })
            .sort_values(["Data Type", "Sensor"])
        )

        st.error(f"🚨 {len(value_issue_display)} value issue sensor row(s) found.")
        st.dataframe(value_issue_display, hide_index=True, use_container_width=True)

    # -----------------------------------------------------
    # Optional combined problem table for quick export / debugging
    # -----------------------------------------------------
    with st.expander("Show combined problem table"):
        problem_rows = []

        if not packet_problem_df.empty:
            for _, r in packet_problem_df.iterrows():
                problem_rows.append({
                    "problem_type": "PACKET_LOSS_>5%",
                    "file": packet_loss_file.name,
                    "data_type": "Packet Loss",
                    "sensor": r["sensor"],
                    "issue": f"Packet loss above {PACKET_LOSS_ALERT_PCT:.0f}%",
                    "loss_pct": float(r["loss_pct"]),
                    "details": f"lost_packets={int(r['lost_packets'])} | expected_packets={int(r['expected_packets'])} | received_packets={int(r['packets_received'])}",
                })

        if not summary_value_issues_df.empty:
            problem_rows.extend(summary_value_issues_df.to_dict("records"))

        all_problem_df = pd.DataFrame(problem_rows)

        if all_problem_df.empty:
            st.success("No problematic sensors found.")
        else:
            st.dataframe(
                all_problem_df.style.format({"loss_pct": "{:.2f}"}),
                hide_index=True,
                use_container_width=True,
            )


# =========================================================
# 10B) Packet Loss Analysis tab - first CSV only
# =========================================================
with tab_packet_loss:
    st.subheader(f"Packet Loss Analysis - first CSV: {packet_loss_file.name}")

    if packet_error is not None:
        st.error(f"Error processing packet-loss analysis: {packet_error}")
    else:
        st.caption("Sampling interval is fixed at 3 minutes. Partial hours are included and calculated by their actual expected timestamp count.")

        st.markdown("### High Packet Loss Alert")

        high_loss_df = packet_sensor_loss_df[
            packet_sensor_loss_df["loss_pct"] > PACKET_LOSS_ALERT_PCT
        ].sort_values(by="loss_pct", ascending=False)

        if high_loss_df.empty:
            st.success("✅ All sensors are operating at 5% or less overall packet loss.")
        else:
            st.error(f"🚨 Attention: {len(high_loss_df)} sensor(s) have overall packet loss above 5%.")

            display_df = high_loss_df[[
                "sensor", "loss_pct", "lost_packets", "expected_packets", "packets_received",
            ]].rename(columns={
                "sensor": "Sensor",
                "loss_pct": "Loss (%)",
                "lost_packets": "Lost Packets",
                "expected_packets": "Expected Packets",
                "packets_received": "Received Packets",
            })

            st.dataframe(
                display_df.style.format({"Loss (%)": "{:.2f}%"}),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### Hourly Packet Loss")
        st.info("🕒 Data is grouped by hour. If the first/last hour is partial, expected packets are calculated only for the timestamps that should exist in that partial hour.")

        plot_type = st.radio(
            "Hourly plot type",
            ["Overall + Raw Sensor Points", "Specific Sensors"],
            horizontal=True,
            index=0,
            key="packet_hourly_plot_type",
        )

        # IMPORTANT:
        # The sensor multiselect is intentionally created ONLY inside this
        # Specific Sensors branch. It will not appear for the Overall + Raw plot.
        if plot_type == "Specific Sensors":
            sensor_list = packet_rep["hourly_sensor_loss"].columns.tolist()

            selected_sensors = st.multiselect(
                "Select sensors to display:",
                options=sensor_list,
                default=sensor_list,
                key="packet_specific_sensor_selector",
            )

            if selected_sensors:
                fig_hourly = plot_hourly_specific_sensors(
                    packet_rep,
                    selected_sensors=selected_sensors,
                )
                st.plotly_chart(
                    fig_hourly,
                    use_container_width=True,
                    key="packet_hourly_loss_specific_sensors",
                )
            else:
                st.info("Select at least one sensor to display the specific-sensors plot.")

        else:
            show_raw_points = st.checkbox(
                "Display raw sensor points",
                value=True,
                key="packet_show_raw_points",
            )

            fig_hourly = plot_hourly_loss_combined(
                packet_rep,
                show_raw_points=show_raw_points,
                show_specific_sensors=False,
                selected_sensors=None,
            )
            st.plotly_chart(
                fig_hourly,
                use_container_width=True,
                key="packet_hourly_loss_overall_raw",
            )

        st.markdown("---")
        st.markdown("### Distribution of Sensor Packet Loss (%)")
        st.plotly_chart(
            plot_sensor_loss_distribution(packet_sensor_loss_df),
            use_container_width=True,
            key="packet_sensor_loss_distribution",
        )

        with st.expander("Show full sensor packet-loss table"):
            st.dataframe(
                packet_sensor_loss_df.style.format({"loss_pct": "{:.2f}%"}),
                hide_index=True,
                use_container_width=True,
            )


# =========================================================
# 10C) Data Analysis tab - all uploaded CSVs
# =========================================================
with tab_data_analysis:
    st.subheader("Data Analysis - All Uploaded CSVs")

    st.caption(
        "Each CSV is analyzed separately. Auto-detection uses the file name and column names. "
        "You can override the detected data type inside each file section."
    )

    st.info(
        "**How stuck values are checked:** this check is now used only for **Temperature** CSVs. "
        "For each temperature sensor, the app sorts values by timestamp, rounds each value to "
        "2 digits after the decimal point by default, and looks for the longest consecutive run "
        "of the same rounded value. Example: `23.001, 23.004, 23.002, 23.003` becomes "
        "`23.00, 23.00, 23.00, 23.00`, so it is a stuck run of 4. "
        "If the run length is equal to or above the threshold, the sensor is flagged. "
        "Battery does not use stuck-value checks. Light also does not use stuck-value checks."
    )

    with st.expander("Data Analysis Settings", expanded=True):
        col_b1, col_b2, col_b3 = st.columns(3)

        with col_b1:
            battery_threshold_mv = st.number_input(
                "Battery low threshold (mV)",
                min_value=0,
                max_value=5000,
                value=DEFAULT_BATTERY_LOW_MV,
                step=50,
            )

        with col_b2:
            battery_last_n = st.number_input(
                "Battery: check last N values",
                min_value=1,
                max_value=200,
                value=DEFAULT_BATTERY_LAST_N,
                step=1,
            )

        with col_b3:
            battery_low_count_limit = st.number_input(
                "Battery: allowed low values",
                min_value=0,
                max_value=20,
                value=DEFAULT_BATTERY_ALLOWED_LOW_COUNT,
                step=1,
            )

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            stuck_run_threshold = st.number_input(
                "TEMP stuck-value run threshold",
                min_value=2,
                max_value=100,
                value=DEFAULT_STUCK_RUN_THRESHOLD,
                step=1,
                help="Used only for Temperature. Example: threshold 4 flags values like 23.00, 23.00, 23.00, 23.00.",
            )

        with col_s2:
            stuck_round_decimals = st.number_input(
                "TEMP stuck check decimal places",
                min_value=0,
                max_value=6,
                value=DEFAULT_STUCK_ROUND_DECIMALS,
                step=1,
                help="Used only for Temperature. Default 2 means 23.001 and 23.004 are both treated as 23.00.",
            )

    # Analyze each uploaded file.
    for file_index, file in enumerate(uploaded_files, start=1):
        try:
            raw = read_uploaded_csv(file)
            auto_type = detect_data_type(file.name, raw)
            data_type_options = ["Battery", "Temperature", "Light", "Other"]
            default_type = auto_type if auto_type in data_type_options else "Other"

            with st.expander(f"{file_index}. {file.name}  |  Auto type: {auto_type}", expanded=(file_index == 1)):
                data_type = st.selectbox(
                    "Data type for this CSV",
                    data_type_options,
                    index=data_type_options.index(default_type),
                    key=f"data_type_{file_index}_{file.name}",
                )

                ts_col = _detect_timestamp_column(raw)
                wide_selected = _to_wide_timeseries(raw, ts_col, data_type=data_type)

                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    st.metric("Rows", len(raw))

                with m2:
                    st.metric("Sensors", len(wide_selected.columns))

                with m3:
                    st.metric("Start", f"{wide_selected.index.min():%Y-%m-%d %H:%M}")

                with m4:
                    st.metric("End", f"{wide_selected.index.max():%Y-%m-%d %H:%M}")

                if data_type == "Other":
                    st.warning("Choose Battery, Temperature, or Light to run automatic checks for this CSV.")
                    with st.expander("Preview wide data"):
                        st.dataframe(wide_selected.head(100), use_container_width=True)
                    continue

                result_df = run_data_health_check(
                    raw,
                    data_type=data_type,
                    battery_threshold_mv=float(battery_threshold_mv),
                    battery_last_n=int(battery_last_n),
                    battery_low_count_limit=int(battery_low_count_limit),
                    stuck_run_threshold=int(stuck_run_threshold),
                    stuck_round_decimals=int(stuck_round_decimals),
                )

                issue_df = result_issues_only(result_df)

                st.markdown("### Sensor Health Results")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric("Sensors checked", len(result_df))

                with c2:
                    st.metric("Sensors with issues", len(issue_df))

                with c3:
                    st.metric("OK sensors", int(len(result_df) - len(issue_df)))

                if issue_df.empty:
                    st.success("✅ No data issues detected for this CSV.")
                else:
                    st.error(f"🚨 {len(issue_df)} sensor(s) have data issues.")
                    st.dataframe(issue_df, hide_index=True, use_container_width=True)

                with st.expander("Show full health-check table", expanded=issue_df.empty):
                    st.dataframe(result_df, hide_index=True, use_container_width=True)

                st.markdown("### Plot Last Values")

                if not issue_df.empty:
                    default_plot_sensors = issue_df["sensor"].astype(str).head(10).tolist()
                else:
                    default_plot_sensors = [str(c) for c in wide_selected.columns[:10]]

                wide_for_plot = wide_selected.copy()
                wide_for_plot.columns = wide_for_plot.columns.astype(str)

                plot_sensors = st.multiselect(
                    "Choose sensors to plot",
                    [str(c) for c in wide_for_plot.columns],
                    default=default_plot_sensors,
                    key=f"plot_sensors_{file_index}_{file.name}",
                )

                if plot_sensors:
                    fig_last = plot_last_values_for_sensors(
                        wide_for_plot,
                        sensors=plot_sensors,
                        title=f"Last values - {data_type} - {file.name}",
                        last_n=int(battery_last_n) if data_type == "Battery" else 100,
                    )

                    if data_type == "Battery":
                        fig_last.add_hline(
                            y=float(battery_threshold_mv),
                            line_dash="dash",
                            annotation_text=f"Low threshold: {battery_threshold_mv:.0f} mV",
                        )

                    st.plotly_chart(fig_last, use_container_width=True, key=f"last_values_plot_{file_index}")

                with st.expander("Preview wide data"):
                    st.dataframe(wide_selected.head(100), use_container_width=True)

        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
