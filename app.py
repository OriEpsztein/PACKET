import io
import json
import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


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
DEFAULT_BATTERY_LAST_N = 20  # Fixed: always analyze the last 20 available battery values.
BATTERY_MIN_OK_VALUES = 20
BATTERY_REPLACE_CONSECUTIVE_LOW = 6
BATTERY_POSSIBLE_LOW_COUNT = 3
BATTERY_REPLACE_LOW_PCT = 20.0

# Severity colors. The severity boundaries themselves are calculated
# adaptively from the uploaded experiment data; they are not fixed percentages.
SEVERITY_COLORS = {
    "LOW": "#F2C94C",
    "MEDIUM": "#F2994A",
    "HIGH": "#EB5757",
}

BATTERY_STATUS_COLORS = {
    "LIMITED BATTERY DATA": "#F2C94C",
    "INSUFFICIENT BATTERY DATA": "#D4A72C",
    "POSSIBLE BATTERY ISSUE": "#F2994A",
    "REPLACE BATTERY": "#EB5757",
}


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


def infer_experiment_name(files) -> str:
    """Infer the experiment name from filenames.

    Examples:
        Morris_Grape_April26_battery_2026-09-03.csv
        Morris_Grape_April26_hdc_temp_2026-09-03.csv

    Both become:
        Morris_Grape_April26

    The experiment name is everything before the parameter/type token.
    """
    if not files:
        return "Unknown"

    # Multi-token parameter names must be checked before generic "temp".
    parameter_patterns = [
        r"[_\- ]hdc[_\- ]temp(?=[_\- ]|$)",
        r"[_\- ]temperature(?=[_\- ]|$)",
        r"[_\- ]battery(?=[_\- ]|$)",
        r"[_\- ]batt(?=[_\- ]|$)",
        r"[_\- ]light(?=[_\- ]|$)",
        r"[_\- ]radiation(?=[_\- ]|$)",
        r"[_\- ]lux(?=[_\- ]|$)",
        r"[_\- ]par(?=[_\- ]|$)",
        r"[_\- ]temp(?=[_\- ]|$)",
    ]

    candidates = []

    for file in files:
        stem = os.path.splitext(str(file.name))[0]
        lower = stem.lower()

        matches = []
        for pattern in parameter_patterns:
            match = re.search(pattern, lower)
            if match:
                matches.append(match.start())

        if matches:
            stem = stem[:min(matches)]

        candidates.append(stem.rstrip("_- "))

    if len(set(candidates)) == 1:
        return candidates[0] or "Unknown"

    common = os.path.commonprefix(candidates).rstrip("_- ")
    return common if len(common) >= 3 else candidates[0]


def _adaptive_cluster_severity(values: pd.Series) -> pd.Series:
    """Split problem values into Low / Medium / High using the data itself.

    This is a deterministic 1-D clustering approach:
    - no fixed percentage boundaries are used;
    - one unique problem level -> Medium;
    - two unique levels -> Low / High;
    - three or more levels -> up to three natural clusters, ordered by centroid.

    The output index matches the input index.
    """
    s = pd.to_numeric(values, errors="coerce")
    valid = s.dropna()

    result = pd.Series(index=s.index, dtype="object")
    if valid.empty:
        return result

    unique_values = np.sort(valid.unique())

    if len(unique_values) == 1:
        result.loc[valid.index] = "MEDIUM"
        return result

    if len(unique_values) == 2:
        low_value, high_value = unique_values[0], unique_values[-1]
        result.loc[valid.index] = np.where(valid <= low_value, "LOW", "HIGH")
        return result

    k = 3

    # Quantile-based deterministic starting centers.
    centers = np.quantile(valid.to_numpy(dtype=float), [0.15, 0.50, 0.85]).astype(float)

    for _ in range(100):
        vals = valid.to_numpy(dtype=float)
        distances = np.abs(vals[:, None] - centers[None, :])
        labels = distances.argmin(axis=1)

        new_centers = centers.copy()
        for cluster_i in range(k):
            cluster_vals = vals[labels == cluster_i]
            if len(cluster_vals):
                new_centers[cluster_i] = cluster_vals.mean()

        if np.allclose(new_centers, centers, rtol=0, atol=1e-10):
            centers = new_centers
            break
        centers = new_centers

    # Recalculate final labels and order clusters from lowest to highest centroid.
    vals = valid.to_numpy(dtype=float)
    labels = np.abs(vals[:, None] - centers[None, :]).argmin(axis=1)

    used_clusters = sorted(set(labels.tolist()), key=lambda i: centers[i])

    if len(used_clusters) == 1:
        cluster_to_severity = {used_clusters[0]: "MEDIUM"}
    elif len(used_clusters) == 2:
        cluster_to_severity = {
            used_clusters[0]: "LOW",
            used_clusters[1]: "HIGH",
        }
    else:
        cluster_to_severity = {
            used_clusters[0]: "LOW",
            used_clusters[1]: "MEDIUM",
            used_clusters[-1]: "HIGH",
        }

    result.loc[valid.index] = [
        cluster_to_severity[int(label)] for label in labels
    ]
    return result


def add_adaptive_severity(
    df: pd.DataFrame,
    value_col: str,
    severity_col: str = "severity",
) -> pd.DataFrame:
    """Return a copy with adaptive Low / Medium / High severity."""
    out = df.copy()
    if out.empty:
        out[severity_col] = pd.Series(dtype="object")
        return out

    out[severity_col] = _adaptive_cluster_severity(out[value_col])
    return out


def normalized_severity(priority: str) -> str:
    """Map health-check priority labels to LOW / MEDIUM / HIGH."""
    value = str(priority).upper()
    if value == "HIGH":
        return "HIGH"
    if value == "MEDIUM":
        return "MEDIUM"
    return "LOW"


def plot_severity_donut(
    rows_df: pd.DataFrame,
    title: str,
    range_col: str = None,
    range_label: str = "Problem range",
    range_decimals: int = 1,
    range_suffix: str = "",
) -> go.Figure:
    """Compact donut chart.

    The donut itself shows category + number of sensors.
    Hover intentionally does NOT show the slice percentage. Instead it shows:
    - number of sensors
    - the actual problem-value range represented by that category
    """
    order = ["LOW", "MEDIUM", "HIGH"]

    labels = []
    values = []
    colors = []
    range_texts = []

    for level in order:
        part = rows_df[rows_df["severity"] == level].copy()
        if part.empty:
            continue

        labels.append(level.title())
        values.append(int(len(part)))
        colors.append(SEVERITY_COLORS[level])

        if range_col and range_col in part.columns:
            numeric = pd.to_numeric(part[range_col], errors="coerce").dropna()
            if numeric.empty:
                range_text = "-"
            else:
                lo = float(numeric.min())
                hi = float(numeric.max())

                if range_decimals == 0:
                    lo_text = f"{lo:.0f}"
                    hi_text = f"{hi:.0f}"
                else:
                    lo_text = f"{lo:.{range_decimals}f}"
                    hi_text = f"{hi:.{range_decimals}f}"

                if np.isclose(lo, hi):
                    range_text = f"{lo_text}{range_suffix}"
                else:
                    range_text = f"{lo_text}{range_suffix} – {hi_text}{range_suffix}"
        else:
            range_text = "-"

        range_texts.append(range_text)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.58,
        marker=dict(colors=colors),
        textinfo="label+value",
        customdata=np.array(range_texts, dtype=object),
        hovertemplate=(
            "%{label}<br>"
            "Sensors=%{value}<br>"
            + range_label + "=%{customdata}<extra></extra>"
        ),
        sort=False,
    )])

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, xanchor="center"),
        margin=dict(t=55, b=10, l=10, r=10),
        showlegend=False,
        height=300,
    )
    return fig


def plot_battery_status_donut(rows_df: pd.DataFrame, title: str = "Battery") -> go.Figure:
    """Battery donut using the real battery status names.

    Hover shows sensor count + range of low readings in the analyzed last values.
    It intentionally does not show the slice percentage.
    """
    order = [
        "LIMITED BATTERY DATA",
        "INSUFFICIENT BATTERY DATA",
        "POSSIBLE BATTERY ISSUE",
        "REPLACE BATTERY",
    ]

    labels = []
    values = []
    colors = []
    ranges = []

    for status in order:
        part = rows_df[rows_df["issue"] == status].copy()
        if part.empty:
            continue

        labels.append(status)
        values.append(int(len(part)))
        colors.append(BATTERY_STATUS_COLORS[status])

        if "under_threshold_count" in part.columns:
            lows = pd.to_numeric(part["under_threshold_count"], errors="coerce").dropna()
        else:
            lows = pd.Series(dtype=float)

        if lows.empty:
            range_text = "-"
        else:
            lo = int(lows.min())
            hi = int(lows.max())
            range_text = str(lo) if lo == hi else f"{lo} – {hi}"

        ranges.append(range_text)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.58,
        marker=dict(colors=colors),
        textinfo="value",
        customdata=np.array(ranges, dtype=object),
        hovertemplate=(
            "%{label}<br>"
            "Sensors=%{value}<br>"
            "Low readings range=%{customdata}<extra></extra>"
        ),
        sort=False,
    )])

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, xanchor="center"),
        margin=dict(t=55, b=65, l=10, r=10),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        height=340,
    )
    return fig


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
        # Keep every sensor column that exists in the CSV, even if that sensor
        # has no numeric measurements at all. A completely empty sensor is still
        # part of the experiment and must be counted.
        sensor_columns = [c for c in df.columns if c != ts_col]

        wide = df.set_index(ts_col)[sensor_columns].sort_index()
        wide = wide.apply(pd.to_numeric, errors="coerce")

        # If the CSV has duplicate timestamps, average them while preserving
        # all original sensor columns, including all-NaN columns.
        wide = wide.groupby(wide.index).mean()
        wide = wide.reindex(columns=sensor_columns)

        if len(wide.columns) == 0:
            raise ValueError("No sensor columns found after parsing.")

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

    # Preserve the complete sensor list before dropping invalid measurements.
    sensor_order = df[sensor_col].dropna().astype(str).drop_duplicates().tolist()

    temp = df[[ts_col, sensor_col, value_col]].copy()
    temp[sensor_col] = temp[sensor_col].astype(str)
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp_valid = temp.dropna(subset=[value_col])

    if temp_valid.empty and not sensor_order:
        raise ValueError("No sensor data found after parsing the selected value column.")

    if temp_valid.empty:
        valid_times = pd.DatetimeIndex(sorted(temp[ts_col].dropna().unique()))
        wide = pd.DataFrame(index=valid_times, columns=sensor_order, dtype=float)
    else:
        wide = (
            temp_valid.pivot_table(
                index=ts_col,
                columns=sensor_col,
                values=value_col,
                aggfunc="mean",
            )
            .sort_index()
            .reindex(columns=sensor_order)
        )

    if len(wide.columns) == 0:
        raise ValueError("No sensor columns found after pivoting long-format data.")

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




def select_packet_loss_source(uploaded_files):
    """Choose the best uploaded CSV to estimate packet loss.

    Preference:
    1. file containing the largest sensor set;
    2. among those, the highest numeric-data coverage;
    3. then the largest number of timestamp rows.

    This avoids depending on upload order and reduces false packet loss from a
    parameter file that has extra value-specific gaps.
    """
    candidates = []

    for file in uploaded_files:
        try:
            raw = read_uploaded_csv(file)
            ts_col = _detect_timestamp_column(raw)
            wide = _to_wide_timeseries(raw, ts_col)

            n_sensors = int(len(wide.columns))
            n_rows = int(len(wide))

            total_cells = int(wide.shape[0] * wide.shape[1])
            valid_cells = int(wide.notna().sum().sum())
            coverage = (valid_cells / total_cells) if total_cells else 0.0

            candidates.append({
                "file": file,
                "raw": raw,
                "n_sensors": n_sensors,
                "coverage": float(coverage),
                "n_rows": n_rows,
            })
        except Exception:
            continue

    if not candidates:
        raise ValueError("Could not find a usable CSV for packet-loss analysis.")

    best = max(
        candidates,
        key=lambda x: (x["n_sensors"], x["coverage"], x["n_rows"]),
    )

    return best["file"], best["raw"], best

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
def _longest_true_run(mask: pd.Series) -> dict:
    """Return the longest consecutive True run in a boolean Series."""
    best_len = 0
    best_start = pd.NaT
    best_end = pd.NaT

    current_len = 0
    current_start = pd.NaT
    current_end = pd.NaT

    for timestamp, is_true in mask.items():
        if bool(is_true):
            if current_len == 0:
                current_start = timestamp
            current_len += 1
            current_end = timestamp

            if current_len > best_len:
                best_len = current_len
                best_start = current_start
                best_end = current_end
        else:
            current_len = 0
            current_start = pd.NaT
            current_end = pd.NaT

    return {
        "max_run": int(best_len),
        "run_start_time": best_start,
        "run_end_time": best_end,
    }


def analyze_battery(
    wide: pd.DataFrame,
    low_mv_threshold: float = DEFAULT_BATTERY_LOW_MV,
    last_n: int = DEFAULT_BATTERY_LAST_N,
) -> pd.DataFrame:
    """Battery health decision tree.

    Important:
    - Battery status is based ONLY on the last 20 available values per sensor.
    - Current time is never used, so historical CSV files are analyzed correctly.
    - Timestamp is used only to show the analyzed/issue range.

    Decision tree:
    0-2 values:
        INSUFFICIENT BATTERY DATA

    3-5 values:
        all values < threshold -> REPLACE BATTERY
        otherwise -> LIMITED BATTERY DATA

    6-19 values:
        >= 6 consecutive low values -> REPLACE BATTERY
        otherwise, >= 3 low values -> POSSIBLE BATTERY ISSUE
        otherwise -> LIMITED BATTERY DATA

    20 values:
        >= 6 consecutive low values -> REPLACE BATTERY
        otherwise, > 20% low values -> REPLACE BATTERY
        otherwise, >= 3 low values -> POSSIBLE BATTERY ISSUE
        otherwise -> OK
    """
    rows = []

    # User requirement: fixed analysis window of 20 available measurements.
    last_n = DEFAULT_BATTERY_LAST_N

    for sensor in wide.columns:
        s = pd.to_numeric(wide[sensor], errors="coerce").dropna().sort_index()
        last_values = s.tail(last_n)

        n_checked = int(len(last_values))
        low_mask = last_values < low_mv_threshold
        low_count = int(low_mask.sum())
        low_pct = (low_count / n_checked * 100.0) if n_checked else np.nan

        run_info = _longest_true_run(low_mask)
        max_low_run = int(run_info["max_run"])

        analysis_start = last_values.index.min() if n_checked else pd.NaT
        analysis_end = last_values.index.max() if n_checked else pd.NaT
        low_values = last_values[low_mask]

        if n_checked <= 2:
            status = "INSUFFICIENT BATTERY DATA"
            issue = True
            priority = "CHECK"
            recommended_action = "Check Data"
            why = f"Only {n_checked} battery value(s) available in the last 20-value window."
            rule_used = "0-2 values -> INSUFFICIENT BATTERY DATA"

        elif n_checked <= 5:
            if low_count == n_checked:
                status = "REPLACE BATTERY"
                issue = True
                priority = "HIGH"
                recommended_action = "Replace Battery"
                why = f"All {n_checked}/{n_checked} available values are below {low_mv_threshold:.0f} mV."
                rule_used = "3-5 values and all are low -> REPLACE BATTERY"
            else:
                status = "LIMITED BATTERY DATA"
                issue = True
                priority = "CHECK"
                recommended_action = "Check Data"
                why = f"Only {n_checked} values are available; not enough evidence for a reliable battery status."
                rule_used = "3-5 values and not all are low -> LIMITED BATTERY DATA"

        elif n_checked < BATTERY_MIN_OK_VALUES:
            if max_low_run >= BATTERY_REPLACE_CONSECUTIVE_LOW:
                status = "REPLACE BATTERY"
                issue = True
                priority = "HIGH"
                recommended_action = "Replace Battery"
                why = (
                    f"{max_low_run} consecutive values are below "
                    f"{low_mv_threshold:.0f} mV."
                )
                rule_used = f"6-19 values and >= {BATTERY_REPLACE_CONSECUTIVE_LOW} consecutive low -> REPLACE BATTERY"
            elif low_count >= BATTERY_POSSIBLE_LOW_COUNT:
                status = "POSSIBLE BATTERY ISSUE"
                issue = True
                priority = "MEDIUM"
                recommended_action = "Check Battery"
                why = (
                    f"{low_count}/{n_checked} values are below "
                    f"{low_mv_threshold:.0f} mV, but there are fewer than 20 values."
                )
                rule_used = f"6-19 values and >= {BATTERY_POSSIBLE_LOW_COUNT} low -> POSSIBLE BATTERY ISSUE"
            else:
                status = "LIMITED BATTERY DATA"
                issue = True
                priority = "CHECK"
                recommended_action = "Check Data"
                why = (
                    f"Only {n_checked} values are available and there is no strong low-battery pattern."
                )
                rule_used = "6-19 values without a strong low-battery pattern -> LIMITED BATTERY DATA"

        else:
            if max_low_run >= BATTERY_REPLACE_CONSECUTIVE_LOW:
                status = "REPLACE BATTERY"
                issue = True
                priority = "HIGH"
                recommended_action = "Replace Battery"
                why = (
                    f"{max_low_run} consecutive values are below "
                    f"{low_mv_threshold:.0f} mV."
                )
                rule_used = f"20 values and >= {BATTERY_REPLACE_CONSECUTIVE_LOW} consecutive low -> REPLACE BATTERY"
            elif low_pct > BATTERY_REPLACE_LOW_PCT:
                status = "REPLACE BATTERY"
                issue = True
                priority = "HIGH"
                recommended_action = "Replace Battery"
                why = (
                    f"{low_count}/20 values ({low_pct:.1f}%) are below "
                    f"{low_mv_threshold:.0f} mV (> {BATTERY_REPLACE_LOW_PCT:.0f}%)."
                )
                rule_used = f"20 values and > {BATTERY_REPLACE_LOW_PCT:.0f}% low -> REPLACE BATTERY"
            elif low_count >= BATTERY_POSSIBLE_LOW_COUNT:
                status = "POSSIBLE BATTERY ISSUE"
                issue = True
                priority = "MEDIUM"
                recommended_action = "Check Battery"
                why = (
                    f"{low_count}/20 values are below "
                    f"{low_mv_threshold:.0f} mV."
                )
                rule_used = f"20 values and >= {BATTERY_POSSIBLE_LOW_COUNT} low -> POSSIBLE BATTERY ISSUE"
            else:
                status = "OK"
                issue = False
                priority = "OK"
                recommended_action = "No Action"
                why = (
                    f"{low_count}/20 values are below {low_mv_threshold:.0f} mV "
                    "and no long consecutive low run was detected."
                )
                rule_used = "20 values with no battery warning rule -> OK"

        rows.append({
            "sensor": str(sensor),
            "status": status,
            "has_issue": bool(issue),
            "priority": priority,
            "severity": normalized_severity(priority),
            "recommended_action": recommended_action,
            "why": why,
            "values_checked": n_checked,
            "under_threshold_count": low_count,
            "low_percentage": low_pct,
            "threshold_mV": low_mv_threshold,
            "max_consecutive_low": max_low_run,
            "last_value_mV": float(last_values.iloc[-1]) if n_checked else np.nan,
            "min_last_values_mV": float(last_values.min()) if n_checked else np.nan,
            "analysis_start_time": analysis_start,
            "analysis_end_time": analysis_end,
            "first_low_time_in_last_values": low_values.index.min() if not low_values.empty else pd.NaT,
            "last_low_time_in_last_values": low_values.index.max() if not low_values.empty else pd.NaT,
            "consecutive_low_start_time": run_info["run_start_time"],
            "consecutive_low_end_time": run_info["run_end_time"],
            "rule_used": rule_used,
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    priority_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "CHECK": 2,
        "OK": 3,
    }
    out["_priority_order"] = out["priority"].map(priority_order).fillna(9)

    out = out.sort_values(
        by=["_priority_order", "under_threshold_count", "min_last_values_mV"],
        ascending=[True, False, True],
    ).drop(columns=["_priority_order"])

    return out

def analyze_temperature(
    wide: pd.DataFrame,
    bad_temp_value: float = -40,
) -> pd.DataFrame:
    """Temperature health check.

    Rules:
    - One -40°C reading is enough to mark a sensor as problematic.
    - A sensor column with zero usable measurements is NO TEMPERATURE DATA.
    - Severity for -40 problems is adaptive relative to the other problematic
      temperature sensors in the same uploaded experiment; no fixed percentage
      boundaries are used.
    """
    rows = []

    for sensor in wide.columns:
        s = pd.to_numeric(wide[sensor], errors="coerce").dropna().sort_index()

        values_count = int(len(s))

        if values_count == 0:
            rows.append({
                "sensor": str(sensor),
                "status": "NO TEMPERATURE DATA",
                "has_issue": True,
                "severity": "HIGH",
                "priority": "HIGH",
                "recommended_action": "Reset Sensor",
                "next_action": "Replace Sensor if persists",
                "why": "The sensor exists in the CSV but has no usable temperature measurements.",
                "values_count": 0,
                "minus_40_count": 0,
                "minus_40_pct": np.nan,
                "first_minus_40_time": pd.NaT,
                "last_minus_40_time": pd.NaT,
                "last_value": np.nan,
            })
            continue

        bad_temp_mask = s <= (bad_temp_value + 0.1)
        bad_temp_count = int(bad_temp_mask.sum())
        bad_temp_pct = bad_temp_count / values_count * 100.0
        has_issue = bad_temp_count > 0

        if has_issue:
            status = "TEMP_-40"
            priority = "MEDIUM"  # replaced by adaptive severity below
            recommended_action = "Reset Sensor"
            next_action = "Replace Sensor if persists"
            why = (
                f"{bad_temp_count}/{values_count} values ({bad_temp_pct:.2f}%) "
                f"are at or below {bad_temp_value:.0f}°C."
            )
            severity = None
        else:
            status = "OK"
            priority = "OK"
            recommended_action = "No Action"
            next_action = ""
            why = "No -40°C values detected."
            severity = "OK"

        rows.append({
            "sensor": str(sensor),
            "status": status,
            "has_issue": has_issue,
            "severity": severity,
            "priority": priority,
            "recommended_action": recommended_action,
            "next_action": next_action,
            "why": why,
            "values_count": values_count,
            "minus_40_count": bad_temp_count,
            "minus_40_pct": bad_temp_pct,
            "first_minus_40_time": s[bad_temp_mask].index.min() if bad_temp_count else pd.NaT,
            "last_minus_40_time": s[bad_temp_mask].index.max() if bad_temp_count else pd.NaT,
            "last_value": float(s.iloc[-1]),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Adaptively split only the actual -40 problems.
    temp_problem_mask = out["status"] == "TEMP_-40"
    if temp_problem_mask.any():
        adaptive = _adaptive_cluster_severity(
            out.loc[temp_problem_mask, "minus_40_pct"]
        )
        out.loc[temp_problem_mask, "severity"] = adaptive
        out.loc[temp_problem_mask, "priority"] = adaptive

    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "OK": 3}
    out["_severity_order"] = out["severity"].map(severity_order).fillna(9)

    return out.sort_values(
        by=["_severity_order", "minus_40_pct", "minus_40_count"],
        ascending=[True, False, False],
        na_position="first",
    ).drop(columns=["_severity_order"])


def analyze_light(wide: pd.DataFrame) -> pd.DataFrame:
    """Light check.

    No automatic Light value-health rule is currently enabled.
    The function returns a clean per-sensor table for preview.
    """
    rows = []

    for sensor in wide.columns:
        s = pd.to_numeric(wide[sensor], errors="coerce").dropna().sort_index()

        rows.append({
            "sensor": str(sensor),
            "status": "OK",
            "has_issue": False,
            "priority": "OK",
            "recommended_action": "No Action",
            "next_action": "",
            "why": "No automatic Light value-health rule is currently enabled.",
            "values_count": int(len(s)),
            "last_value": float(s.iloc[-1]) if len(s) else np.nan,
            "note": "No automatic Light value-health rule is currently enabled.",
        })

    return pd.DataFrame(rows).sort_values(by=["sensor"])


def run_data_health_check(
    df: pd.DataFrame,
    data_type: str,
    battery_threshold_mv: float = DEFAULT_BATTERY_LOW_MV,
    battery_last_n: int = DEFAULT_BATTERY_LAST_N,
) -> pd.DataFrame:
    """Run the correct health check according to selected data type."""
    ts_col = _detect_timestamp_column(df)
    wide = _to_wide_timeseries(df, ts_col, data_type=data_type)

    if data_type == "Battery":
        return analyze_battery(
            wide,
            low_mv_threshold=battery_threshold_mv,
            last_n=DEFAULT_BATTERY_LAST_N,
        )

    if data_type == "Temperature":
        return analyze_temperature(wide)

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


def _fmt_timestamp(ts) -> str:
    """Format a timestamp for compact dashboard display."""
    if pd.isna(ts):
        return "-"
    return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M")


def build_issue_time_range(issue_row: pd.Series, data_type: str) -> str:
    """Describe when the detected issue occurred.

    Time never affects battery status. For Battery it only shows the range of
    the last 20 values that were analyzed.
    """
    if data_type == "Battery":
        start = issue_row.get("analysis_start_time", pd.NaT)
        end = issue_row.get("analysis_end_time", pd.NaT)
        if pd.isna(start) or pd.isna(end):
            return "-"
        return f"Analyzed values: {_fmt_timestamp(start)} → {_fmt_timestamp(end)}"

    if data_type == "Temperature":
        ranges = []
        status = str(issue_row.get("status", ""))

        if "TEMP_-40" in status:
            start = issue_row.get("first_minus_40_time", pd.NaT)
            end = issue_row.get("last_minus_40_time", pd.NaT)
            if not pd.isna(start) and not pd.isna(end):
                ranges.append(f"-40: {_fmt_timestamp(start)} → {_fmt_timestamp(end)}")

        return " | ".join(ranges) if ranges else "-"

    return "-"


def detect_system_packet_loss(rep: dict, threshold_pct: float = PACKET_LOSS_ALERT_PCT) -> dict:
    """Detect a likely Pi/system-level packet-loss pattern adaptively.

    The 5% threshold still defines whether an individual sensor has a packet-loss
    problem. Pi-level detection is different: it compares the number of sensors
    affected together in each hour with the experiment's own normal pattern.

    A Pi-level recommendation is made when either:
    - many sensors are persistently affected together relative to experiment size; or
    - an hour is a statistical high outlier versus the experiment's usual number
      of simultaneously affected sensors.

    No fixed "50% of sensors" rule is used.
    """
    empty = {
        "detected": False,
        "hour": pd.NaT,
        "affected": 0,
        "total": 0,
        "affected_pct": 0.0,
        "typical_affected": 0.0,
        "dynamic_cutoff": np.nan,
        "mode": "",
    }

    if not rep or rep.get("hourly_sensor_loss") is None:
        return empty

    loss_mat = rep["hourly_sensor_loss"]
    if loss_mat.empty:
        return empty

    total = int(loss_mat.shape[1])
    if total < 3:
        return {**empty, "total": total}

    affected_counts = (loss_mat > threshold_pct).sum(axis=1).astype(float)

    max_hour = affected_counts.idxmax()
    max_affected = int(affected_counts.loc[max_hour])
    affected_pct = max_affected / total * 100.0 if total else 0.0

    typical = float(affected_counts.median())
    q1 = float(affected_counts.quantile(0.25))
    q3 = float(affected_counts.quantile(0.75))
    iqr = q3 - q1
    dynamic_cutoff = q3 + 1.5 * iqr

    # Experiment-size-aware minimum simultaneous group.
    # For 64 sensors this is 8; for 25 sensors it is 5.
    meaningful_group = max(3, int(np.ceil(np.sqrt(total))))

    persistent_system_pattern = typical >= meaningful_group

    if iqr > 0:
        spike_pattern = (
            max_affected >= meaningful_group
            and max_affected > dynamic_cutoff
        )
    else:
        spike_pattern = (
            max_affected >= meaningful_group
            and max_affected > typical
        )

    detected = bool(persistent_system_pattern or spike_pattern)

    if persistent_system_pattern:
        mode = "persistent common pattern"
    elif spike_pattern:
        mode = "simultaneous spike"
    else:
        mode = ""

    return {
        "detected": detected,
        "hour": pd.to_datetime(max_hour),
        "affected": max_affected,
        "total": total,
        "affected_pct": affected_pct,
        "typical_affected": typical,
        "dynamic_cutoff": dynamic_cutoff,
        "mode": mode,
    }


def build_packet_problem_actions(
    packet_problem_df: pd.DataFrame,
    packet_rep: dict,
) -> tuple[pd.DataFrame, dict]:
    """Attach repair recommendations to packet-loss problems."""
    system_info = detect_system_packet_loss(packet_rep)

    if packet_problem_df.empty:
        return pd.DataFrame(), system_info

    rows = []
    loss_mat = packet_rep.get("hourly_sensor_loss", pd.DataFrame())

    for _, r in packet_problem_df.iterrows():
        loss_pct = float(r["loss_pct"])
        sensor = str(r["sensor"])
        severity = str(r.get("severity", "MEDIUM"))

        sensor_bad_hours = pd.DatetimeIndex([])
        sensor_col = None

        # Match the original column while tolerating non-string column types.
        for col in loss_mat.columns:
            if str(col) == sensor:
                sensor_col = col
                break

        if sensor_col is not None:
            bad_mask = loss_mat[sensor_col] > PACKET_LOSS_ALERT_PCT
            sensor_bad_hours = pd.DatetimeIndex(loss_mat.index[bad_mask])

        if system_info["detected"] and len(sensor_bad_hours) > 0:
            action = "Reset Pi"
            priority = "HIGH"
            why = (
                f"Packet loss is {loss_pct:.2f}% and the experiment shows a "
                f"{system_info['mode']} of sensors failing together "
                f"(peak {system_info['affected']}/{system_info['total']} sensors; "
                f"typical {system_info['typical_affected']:.1f})."
            )
        else:
            action = "Reset Sensor"
            priority = severity
            why = (
                f"Packet loss is {loss_pct:.2f}%. Severity is {severity.title()} "
                "relative to the other sensors above 5% in this experiment."
            )

        if len(sensor_bad_hours):
            issue_range = (
                f"{_fmt_timestamp(sensor_bad_hours.min())} → "
                f"{_fmt_timestamp(sensor_bad_hours.max())}"
            )
        else:
            issue_range = "-"

        rows.append({
            "problem_type": "PACKET_LOSS",
            "file": packet_loss_file.name,
            "data_type": "Packet Loss",
            "sensor": sensor,
            "issue": f"Packet Loss {loss_pct:.2f}%",
            "priority": priority,
            "severity": severity,
            "recommended_action": action,
            "next_action": (
                "Retest → Reset Sensor → Replace Sensor if persists"
                if action == "Reset Pi"
                else "Retest → Replace Sensor if persists"
            ),
            "issue_time_range": issue_range,
            "loss_pct": loss_pct,
            "why": why,
            "details": (
                f"lost_packets={int(r['lost_packets'])} | "
                f"expected_packets={int(r['expected_packets'])} | "
                f"received_packets={int(r['packets_received'])}"
            ),
        })

    return pd.DataFrame(rows), system_info


def build_unified_problem_table(
    packet_action_df: pd.DataFrame,
    value_issue_df: pd.DataFrame,
    packet_system_info: dict,
) -> pd.DataFrame:
    """Create one action row per sensor while preserving every detected problem.

    Battery problems override the primary recommended action, as requested.
    Other problems remain visible in the same row.
    """
    parts = []
    if not packet_action_df.empty:
        parts.append(packet_action_df.copy())
    if not value_issue_df.empty:
        parts.append(value_issue_df.copy())

    if not parts:
        return pd.DataFrame()

    all_problems = pd.concat(parts, ignore_index=True, sort=False)

    severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    rows = []

    for sensor, group in all_problems.groupby("sensor", sort=False):
        group = group.copy()

        battery_rows = group[group["data_type"] == "Battery"]
        packet_rows = group[group["data_type"] == "Packet Loss"]
        temp_rows = group[group["data_type"] == "Temperature"]

        problem_labels = []
        for _, r in group.iterrows():
            if r["data_type"] == "Battery":
                problem_labels.append(f"Battery: {r['issue']}")
            elif r["data_type"] == "Packet Loss":
                problem_labels.append(str(r["issue"]))
            elif r["data_type"] == "Temperature":
                problem_labels.append(f"Temperature: {r['issue']}")
            else:
                problem_labels.append(f"{r['data_type']}: {r['issue']}")

        # Battery overrides every other primary action.
        if not battery_rows.empty:
            battery = battery_rows.iloc[0]
            battery_status = str(battery["issue"])

            if battery_status == "REPLACE BATTERY":
                recommended_action = (
                    "Replace Battery → Retest → if other issues persist, "
                    "Reset Sensor → Replace Sensor"
                )
            elif battery_status == "POSSIBLE BATTERY ISSUE":
                recommended_action = (
                    "Check Battery → Retest → if other issues persist, "
                    "Reset Sensor → Replace Sensor"
                )
            else:
                recommended_action = (
                    "Check Battery/Data → Retest → if other issues persist, "
                    "Reset Sensor → Replace Sensor"
                )

            primary_reason = str(battery.get("why", ""))

        elif not packet_rows.empty and packet_system_info.get("detected", False):
            recommended_action = (
                "Reset Pi → Retest → if this sensor still fails, "
                "Reset Sensor → Replace Sensor"
            )
            primary_reason = str(packet_rows.iloc[0].get("why", ""))

        else:
            recommended_action = "Reset Sensor → Retest → Replace Sensor if persists"

            reasons = []
            if not packet_rows.empty:
                reasons.append(str(packet_rows.iloc[0].get("why", "")))
            if not temp_rows.empty:
                reasons.append(str(temp_rows.iloc[0].get("why", "")))
            primary_reason = " | ".join([r for r in reasons if r])

        group_severities = [
            str(v) for v in group["severity"].dropna().tolist()
            if str(v) in severity_rank
        ]
        highest_severity = (
            max(group_severities, key=lambda v: severity_rank[v])
            if group_severities else "MEDIUM"
        )

        ranges = [
            str(v) for v in group["issue_time_range"].dropna().tolist()
            if str(v).strip() and str(v) != "-"
        ]
        unique_ranges = list(dict.fromkeys(ranges))

        rows.append({
            "Sensor": str(sensor),
            "Severity": highest_severity,
            "Detected Problems": " | ".join(dict.fromkeys(problem_labels)),
            "Recommended Action": recommended_action,
            "Why": primary_reason,
            "Time / Data Range": " | ".join(unique_ranges) if unique_ranges else "-",
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["_severity_rank"] = out["Severity"].map(severity_rank).fillna(0)
    out = out.sort_values(
        ["_severity_rank", "Sensor"],
        ascending=[False, True],
    ).drop(columns=["_severity_rank"])

    return out


def plot_packet_loss_heatmap(rep: dict) -> go.Figure:
    """Sensor x hour heatmap of packet-loss percentage."""
    loss_mat = rep["hourly_sensor_loss"].copy()

    # Sensors as rows, hours as columns.
    z = loss_mat.T.to_numpy(dtype=float)
    x = [pd.to_datetime(x) for x in loss_mat.index]
    y = [str(y) for y in loss_mat.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            zmin=0,
            zmax=100,
            colorbar=dict(title="Loss %"),
            hovertemplate=(
                "Hour: %{x|%Y-%m-%d %H:00}<br>"
                "Sensor: %{y}<br>"
                "Packet Loss: %{z:.2f}%<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        title="Packet Loss Heatmap - Sensor × Hour",
        xaxis_title="Hour",
        yaxis_title="Sensor",
        margin=dict(t=60, b=80),
        height=max(450, min(1000, 180 + 18 * len(y))),
    )
    fig.update_xaxes(type="date", automargin=True)

    return fig



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
    help="Upload one or several CSV files. Packet Loss source is selected automatically from the best-covered sensor file.",
)

if not uploaded_files:
    st.info("Upload one or more CSV files from the left sidebar to begin analysis.")
    st.stop()

# Compact uploaded-file cards.
st.sidebar.markdown("---")
st.sidebar.subheader("Uploaded Data")

sidebar_icons = {
    "Battery": "🔋",
    "Temperature": "🌡️",
    "Light": "☀️",
    "Other": "📄",
}

for sidebar_i, sidebar_file in enumerate(uploaded_files, start=1):
    try:
        sidebar_info = get_basic_file_info(sidebar_file)
        sidebar_start = sidebar_info["start"]
        sidebar_end = sidebar_info["end"]
        sidebar_type = sidebar_info["auto_type"]
        sidebar_icon = sidebar_icons.get(sidebar_type, "📄")

        st.sidebar.markdown(f"**{sidebar_icon} {sidebar_i}. {sidebar_file.name}**")
        st.sidebar.caption(
            f"{sidebar_type} • {sidebar_info['sensors']} sensors  \n"
            f"{sidebar_start:%Y-%m-%d %H:%M} → {sidebar_end:%Y-%m-%d %H:%M}"
        )

    except Exception as sidebar_e:
        st.sidebar.markdown(f"**📄 {sidebar_i}. {sidebar_file.name}**")
        st.sidebar.caption(f"Could not read file information: {sidebar_e}")

# Packet Loss source is selected automatically; upload order does not matter.
packet_loss_file, packet_loss_df, packet_loss_source_info = select_packet_loss_source(uploaded_files)

# =========================================================
# 8) Pre-compute packet loss
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

    # >5% remains the fixed definition of a packet-loss problem.
    # Low/Medium/High is adaptive relative to the problem sensors in this experiment.
    packet_problem_df = add_adaptive_severity(
        packet_problem_df,
        value_col="loss_pct",
        severity_col="severity",
    )
except Exception as e:
    packet_error = e


# =========================================================
# 9) Pre-compute health results for Summary
# =========================================================
summary_value_issue_rows = []
summary_health_rows = []
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
            )

            issue_df = result_issues_only(health_df)
            row["value_issues_count"] = len(issue_df)

            for _, health_row in health_df.iterrows():
                summary_health_rows.append({
                    "file": file.name,
                    "data_type": auto_type,
                    "sensor": str(health_row.get("sensor", "")),
                    "status": health_row.get("status", ""),
                    "has_issue": bool(health_row.get("has_issue", False)),
                    "priority": health_row.get("priority", ""),
                    "recommended_action": health_row.get("recommended_action", ""),
                })

            for _, issue_row in issue_df.iterrows():
                if auto_type == "Battery":
                    details = " | ".join([
                        f"values_checked={issue_row.get('values_checked', '')}",
                        f"low={issue_row.get('under_threshold_count', '')}",
                        f"low_pct={issue_row.get('low_percentage', np.nan):.1f}%"
                        if pd.notna(issue_row.get("low_percentage", np.nan)) else "low_pct=-",
                        f"max_consecutive_low={issue_row.get('max_consecutive_low', '')}",
                        f"last_mV={issue_row.get('last_value_mV', np.nan):.0f}"
                        if pd.notna(issue_row.get("last_value_mV", np.nan)) else "last_mV=-",
                    ])
                elif auto_type == "Temperature":
                    details = " | ".join([
                        f"values={issue_row.get('values_count', '')}",
                        f"minus_40_count={issue_row.get('minus_40_count', '')}",
                        f"minus_40_pct={issue_row.get('minus_40_pct', np.nan):.2f}%"
                        if pd.notna(issue_row.get("minus_40_pct", np.nan)) else "minus_40_pct=-",
                    ])
                else:
                    details = str(issue_row.get("note", ""))

                summary_value_issue_rows.append({
                    "problem_type": "VALUE_ISSUE",
                    "file": file.name,
                    "data_type": auto_type,
                    "sensor": str(issue_row.get("sensor", "")),
                    "issue": issue_row.get("status", ""),
                    "priority": issue_row.get("priority", ""),
                    "severity": issue_row.get("severity", normalized_severity(issue_row.get("priority", ""))),
                    "recommended_action": issue_row.get("recommended_action", ""),
                    "next_action": issue_row.get("next_action", ""),
                    "issue_time_range": build_issue_time_range(issue_row, auto_type),
                    "loss_pct": np.nan,
                    "under_threshold_count": issue_row.get("under_threshold_count", np.nan),
                    "values_checked": issue_row.get("values_checked", np.nan),
                    "minus_40_count": issue_row.get("minus_40_count", np.nan),
                    "minus_40_pct": issue_row.get("minus_40_pct", np.nan),
                    "why": issue_row.get("why", ""),
                    "details": details,
                })
        else:
            row["value_issues_count"] = np.nan

    except Exception as e:
        row["status"] = f"ERROR: {e}"

    summary_file_rows.append(row)

summary_files_df = pd.DataFrame(summary_file_rows)
summary_health_df = pd.DataFrame(summary_health_rows)
summary_value_issues_df = pd.DataFrame(summary_value_issue_rows)

# Packet-loss repair recommendations.
packet_action_df, packet_system_info = build_packet_problem_actions(
    packet_problem_df,
    packet_rep,
)


# =========================================================
# 9B) Browser-local Maintenance Log
# =========================================================
def _primary_action_from_recommendation(recommendation: str) -> str:
    """Choose the first concrete maintenance action from a recommendation sequence."""
    value = str(recommendation)

    if value.startswith("Replace Battery") or "Replace Battery →" in value:
        return "Replace Battery"
    if value.startswith("Reset Pi") or "Reset Pi →" in value:
        return "Reset Pi"
    if value.startswith("Replace Sensor") or "Replace Sensor →" in value:
        return "Replace Sensor"
    if value.startswith("Reset Sensor") or "Reset Sensor →" in value:
        return "Reset Sensor"
    if value.startswith("Check Battery"):
        return "Check Battery"

    return "Reset Sensor"


def build_maintenance_log_html(
    experiment_name: str,
    sensors: list,
    problem_df: pd.DataFrame,
) -> str:
    """Build a self-contained browser-local maintenance log.

    Data is stored in this browser's localStorage under a key specific to the
    experiment. It is append-only for normal actions; rows can be deleted if
    entered by mistake.
    """
    sensor_list = sorted({str(s) for s in sensors if str(s).strip()})

    recommendations = []
    if problem_df is not None and not problem_df.empty:
        for _, row in problem_df.iterrows():
            recommendation = str(row.get("Recommended Action", ""))
            recommendations.append({
                "sensor": str(row.get("Sensor", "")),
                "problems": str(row.get("Detected Problems", "")),
                "recommendation": recommendation,
                "primary_action": _primary_action_from_recommendation(recommendation),
            })

    payload = {
        "experiment": str(experiment_name),
        "sensors": sensor_list,
        "recommendations": recommendations,
    }

    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    storage_key_json = json.dumps(
        f"field4d_maintenance::{experiment_name}",
        ensure_ascii=False,
    )

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
    * {{ box-sizing: border-box; }}
    body {{
        margin: 0;
        background: #0e1117;
        color: #fafafa;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 14px;
    }}
    .wrap {{ padding: 4px 4px 20px 4px; }}
    .card {{
        border: 1px solid #30343b;
        background: #151922;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }}
    h3 {{ margin: 0 0 12px 0; font-size: 18px; }}
    .muted {{ color: #aab2bf; font-size: 12px; }}
    textarea, select, input {{
        width: 100%;
        background: #0e1117;
        color: #fafafa;
        border: 1px solid #3a404a;
        border-radius: 7px;
        padding: 9px;
        min-height: 38px;
    }}
    textarea {{ min-height: 90px; resize: vertical; }}
    button {{
        background: #262c36;
        color: #fafafa;
        border: 1px solid #414956;
        border-radius: 7px;
        padding: 8px 12px;
        cursor: pointer;
    }}
    button:hover {{ background: #313846; }}
    .primary {{
        background: #1f6f43;
        border-color: #2b8c56;
    }}
    .danger {{
        background: #5c2525;
        border-color: #8b3434;
        padding: 5px 8px;
    }}
    .grid {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 1fr;
        gap: 10px;
    }}
    .grid2 {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }}
    .actions {{
        display: flex;
        gap: 8px;
        margin-top: 10px;
        flex-wrap: wrap;
    }}
    .rec {{
        border-top: 1px solid #2d323b;
        padding: 10px 0;
    }}
    .rec:first-of-type {{ border-top: none; }}
    .rec-title {{ font-weight: 700; }}
    .problem {{ color: #ffbf69; margin: 4px 0; }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }}
    th, td {{
        border-bottom: 1px solid #30343b;
        padding: 8px 6px;
        text-align: left;
        vertical-align: top;
    }}
    th {{
        color: #cbd3df;
        font-size: 12px;
        position: sticky;
        top: 0;
        background: #151922;
    }}
    .okmsg {{ color: #73d699; }}
    .status {{ min-height: 18px; margin-top: 8px; }}
    @media (max-width: 900px) {{
        .grid, .grid2 {{ grid-template-columns: 1fr; }}
    }}
</style>
</head>
<body>
<div class="wrap">

    <div class="card">
        <h3>Experiment Notes</h3>
        <div class="muted">Saved only in this browser for: <b id="expName"></b></div>
        <textarea id="experimentNotes" placeholder="General notes about the experiment..."></textarea>
        <div class="actions">
            <button class="primary" onclick="saveExperimentNotes()">Save Notes</button>
        </div>
        <div id="notesStatus" class="status okmsg"></div>
    </div>

    <div class="card">
        <h3>Current Recommended Actions</h3>
        <div class="muted">Press ✓ only after you actually performed the action. The local date/time is recorded automatically and can be edited.</div>
        <div id="recommendations"></div>
    </div>

    <div class="card">
        <h3>Log Maintenance Action</h3>
        <div class="grid">
            <div>
                <div class="muted">Sensor</div>
                <select id="sensorSelect"></select>
            </div>
            <div>
                <div class="muted">Action</div>
                <select id="actionSelect">
                    <option>Reset Pi</option>
                    <option>Reset Sensor</option>
                    <option>Replace Battery</option>
                    <option>Replace Sensor</option>
                    <option>Check Battery</option>
                </select>
            </div>
            <div>
                <div class="muted">Date</div>
                <input type="date" id="actionDate">
            </div>
            <div>
                <div class="muted">Time</div>
                <input type="time" id="actionTime" step="60">
            </div>
        </div>

        <div style="margin-top:10px">
            <div class="muted">Optional note</div>
            <textarea id="actionNote" placeholder="What did you do / what did you observe?"></textarea>
        </div>

        <div class="actions">
            <button class="primary" onclick="saveManualAction()">✓ Save Action</button>
        </div>
        <div id="actionStatus" class="status okmsg"></div>
    </div>

    <div class="card">
        <h3>Maintenance History</h3>

        <div class="grid">
            <div>
                <div class="muted">Filter Sensor</div>
                <select id="filterSensor" onchange="renderHistory()"></select>
            </div>
            <div>
                <div class="muted">Filter Action</div>
                <select id="filterAction" onchange="renderHistory()">
                    <option value="">All actions</option>
                    <option>Reset Pi</option>
                    <option>Reset Sensor</option>
                    <option>Replace Battery</option>
                    <option>Replace Sensor</option>
                    <option>Check Battery</option>
                </select>
            </div>
            <div>
                <div class="muted">Filter Date</div>
                <input type="date" id="filterDate" onchange="renderHistory()">
            </div>
            <div style="display:flex;align-items:end">
                <button onclick="exportCSV()">Export CSV</button>
            </div>
        </div>

        <div style="overflow:auto; max-height:420px;">
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Sensor</th>
                        <th>Action</th>
                        <th>Problems at action time</th>
                        <th>Note</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody id="historyBody"></tbody>
            </table>
        </div>
    </div>

</div>

<script>
const APP = {payload_json};
const STORAGE_KEY = {storage_key_json};

function emptyStore() {{
    return {{
        experiment: APP.experiment,
        notes: "",
        logs: []
    }};
}}

function loadStore() {{
    try {{
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return emptyStore();
        const parsed = JSON.parse(raw);
        if (!parsed.logs) parsed.logs = [];
        if (parsed.notes === undefined) parsed.notes = "";
        return parsed;
    }} catch (e) {{
        return emptyStore();
    }}
}}

function saveStore(store) {{
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
}}

function localNowParts() {{
    const d = new Date();
    const pad = n => String(n).padStart(2, "0");
    return {{
        date: `${{d.getFullYear()}}-${{pad(d.getMonth()+1)}}-${{pad(d.getDate())}}`,
        time: `${{pad(d.getHours())}}:${{pad(d.getMinutes())}}`
    }};
}}

function setNow() {{
    const p = localNowParts();
    document.getElementById("actionDate").value = p.date;
    document.getElementById("actionTime").value = p.time;
}}

function escapeHtml(value) {{
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}}

function sensorOptions(includeAll=false) {{
    let html = "";
    if (includeAll) html += '<option value="">All sensors</option>';
    html += '<option value="Pi/System">Pi/System</option>';
    for (const s of APP.sensors) {{
        html += `<option value="${{escapeHtml(s)}}">${{escapeHtml(s)}}</option>`;
    }}
    return html;
}}

function saveExperimentNotes() {{
    const store = loadStore();
    store.notes = document.getElementById("experimentNotes").value;
    saveStore(store);
    document.getElementById("notesStatus").textContent = "Saved locally.";
    setTimeout(() => document.getElementById("notesStatus").textContent = "", 1800);
}}

function appendLog(sensor, action, problems, note, dateValue, timeValue) {{
    const store = loadStore();

    const date = dateValue || localNowParts().date;
    const time = timeValue || localNowParts().time;

    store.logs.push({{
        id: `${{Date.now()}}-${{Math.random().toString(16).slice(2)}}`,
        experiment: APP.experiment,
        sensor: sensor,
        action: action,
        problems: problems || "",
        note: note || "",
        local_datetime: `${{date}} ${{time}}`,
        date: date,
        time: time
    }});

    saveStore(store);
    renderHistory();
}}

function saveManualAction() {{
    const sensor = document.getElementById("sensorSelect").value;
    const action = document.getElementById("actionSelect").value;
    const date = document.getElementById("actionDate").value;
    const time = document.getElementById("actionTime").value;
    const note = document.getElementById("actionNote").value;

    let problems = "";
    const rec = APP.recommendations.find(r => r.sensor === sensor);
    if (rec) problems = rec.problems;

    appendLog(sensor, action, problems, note, date, time);

    document.getElementById("actionNote").value = "";
    document.getElementById("actionStatus").textContent = "Action saved locally.";
    setTimeout(() => document.getElementById("actionStatus").textContent = "", 1800);
    setNow();
}}

function logRecommended(index) {{
    const rec = APP.recommendations[index];
    if (!rec) return;

    const p = localNowParts();
    const note = prompt("Optional note:", "") ?? "";

    appendLog(
        rec.sensor,
        rec.primary_action,
        rec.problems,
        note,
        p.date,
        p.time
    );
}}

function renderRecommendations() {{
    const root = document.getElementById("recommendations");

    if (!APP.recommendations.length) {{
        root.innerHTML = '<div class="okmsg">No current recommended actions.</div>';
        return;
    }}

    root.innerHTML = APP.recommendations.map((r, i) => `
        <div class="rec">
            <div class="rec-title">Sensor ${{escapeHtml(r.sensor)}}</div>
            <div class="problem">${{escapeHtml(r.problems)}}</div>
            <div>${{escapeHtml(r.recommendation)}}</div>
            <div class="actions">
                <button class="primary" onclick="logRecommended(${{i}})">✓ ${{escapeHtml(r.primary_action)}}</button>
            </div>
        </div>
    `).join("");
}}

function filteredLogs() {{
    const store = loadStore();
    const sensor = document.getElementById("filterSensor").value;
    const action = document.getElementById("filterAction").value;
    const date = document.getElementById("filterDate").value;

    return store.logs
        .filter(r => !sensor || r.sensor === sensor)
        .filter(r => !action || r.action === action)
        .filter(r => !date || r.date === date)
        .sort((a,b) => String(b.local_datetime).localeCompare(String(a.local_datetime)));
}}

function renderHistory() {{
    const rows = filteredLogs();
    const body = document.getElementById("historyBody");

    if (!rows.length) {{
        body.innerHTML = '<tr><td colspan="6" class="muted">No saved maintenance actions.</td></tr>';
        return;
    }}

    body.innerHTML = rows.map(r => `
        <tr>
            <td>${{escapeHtml(r.local_datetime)}}</td>
            <td>${{escapeHtml(r.sensor)}}</td>
            <td>${{escapeHtml(r.action)}}</td>
            <td>${{escapeHtml(r.problems)}}</td>
            <td>${{escapeHtml(r.note)}}</td>
            <td><button class="danger" onclick="deleteLog('${{r.id}}')">Delete</button></td>
        </tr>
    `).join("");
}}

function deleteLog(id) {{
    if (!confirm("Delete this entry?")) return;
    const store = loadStore();
    store.logs = store.logs.filter(r => r.id !== id);
    saveStore(store);
    renderHistory();
}}

function csvEscape(value) {{
    const s = String(value ?? "");
    return '"' + s.replaceAll('"', '""') + '"';
}}

function exportCSV() {{
    const store = loadStore();
    const rows = store.logs
        .slice()
        .sort((a,b) => String(a.local_datetime).localeCompare(String(b.local_datetime)));

    const header = [
        "Experiment", "Time", "Sensor", "Action",
        "Problems at action time", "Note"
    ];

    const lines = [header.map(csvEscape).join(",")];

    for (const r of rows) {{
        lines.push([
            r.experiment,
            r.local_datetime,
            r.sensor,
            r.action,
            r.problems,
            r.note
        ].map(csvEscape).join(","));
    }}

    const blob = new Blob(["\\uFEFF" + lines.join("\\n")], {{
        type: "text/csv;charset=utf-8;"
    }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${{APP.experiment}}_maintenance_log.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}}

function init() {{
    document.getElementById("expName").textContent = APP.experiment;
    document.getElementById("sensorSelect").innerHTML = sensorOptions(false);
    document.getElementById("filterSensor").innerHTML = sensorOptions(true);

    const store = loadStore();
    document.getElementById("experimentNotes").value = store.notes || "";

    setNow();
    renderRecommendations();
    renderHistory();
}}

init();
</script>
</body>
</html>
"""


# =========================================================
# 10) Tabs
# =========================================================
experiment_name = infer_experiment_name(uploaded_files)
unified_problem_df = pd.DataFrame()

tab_summary, tab_packet_loss, tab_data_analysis, tab_maintenance = st.tabs([
    "Summary",
    "Packet Loss Analysis",
    "Data Analysis",
    "Maintenance Log",
])


# =========================================================
# 10A) Summary tab
# =========================================================
with tab_summary:
    st.subheader("Summary")

    all_sensor_names = set()
    if not packet_sensor_loss_df.empty:
        all_sensor_names.update(packet_sensor_loss_df["sensor"].astype(str).tolist())
    if not summary_health_df.empty:
        all_sensor_names.update(summary_health_df["sensor"].astype(str).tolist())

    attention_sensor_names = set()
    if not packet_action_df.empty:
        attention_sensor_names.update(packet_action_df["sensor"].astype(str).tolist())
    if not summary_value_issues_df.empty:
        attention_sensor_names.update(summary_value_issues_df["sensor"].astype(str).tolist())

    total_sensors = len(all_sensor_names)
    sensors_need_attention = len(attention_sensor_names)

    top1, top2, top3 = st.columns([2.2, 1, 1])
    with top1:
        st.metric("Experiment", experiment_name)
    with top2:
        st.metric("Total Sensors", total_sensors)
    with top3:
        st.metric("Need Attention", sensors_need_attention)

    st.caption(
        f"{sensors_need_attention} / {total_sensors} unique sensors require attention. "
        "A sensor is counted once even if it has more than one problem."
    )

    # -----------------------------------------------------
    # One donut per problem type, split by severity
    # -----------------------------------------------------
    st.markdown("### Problem Severity")

    packet_severity_df = pd.DataFrame(columns=["sensor", "severity", "loss_pct"])
    if not packet_action_df.empty:
        packet_severity_df = packet_action_df[["sensor", "severity", "loss_pct"]].copy()

    battery_status_df = pd.DataFrame(
        columns=["sensor", "issue", "severity", "under_threshold_count", "values_checked"]
    )
    temperature_severity_df = pd.DataFrame(
        columns=["sensor", "severity", "minus_40_count", "minus_40_pct"]
    )

    if not summary_value_issues_df.empty:
        battery_status_df = summary_value_issues_df[
            summary_value_issues_df["data_type"] == "Battery"
        ][
            ["sensor", "issue", "severity", "under_threshold_count", "values_checked"]
        ].copy()

        temperature_severity_df = summary_value_issues_df[
            summary_value_issues_df["data_type"] == "Temperature"
        ][
            ["sensor", "severity", "minus_40_count", "minus_40_pct"]
        ].copy()

    severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

    def _dedupe_highest_severity(df):
        if df.empty:
            return df
        temp = df.copy()
        temp["_rank"] = temp["severity"].map(severity_rank).fillna(1)
        temp = temp.sort_values("_rank", ascending=False).drop_duplicates("sensor")
        return temp.drop(columns=["_rank"])

    packet_severity_df = _dedupe_highest_severity(packet_severity_df)
    temperature_severity_df = _dedupe_highest_severity(temperature_severity_df)

    pie1, pie2, pie3 = st.columns(3)

    with pie1:
        if packet_severity_df.empty:
            st.markdown("#### Packet Loss")
            st.success("No sensors above 5%")
        else:
            st.plotly_chart(
                plot_severity_donut(
                    packet_severity_df,
                    "Packet Loss",
                    range_col="loss_pct",
                    range_label="Packet Loss range",
                    range_decimals=1,
                    range_suffix="%",
                ),
                use_container_width=True,
                key="summary_packet_loss_severity",
            )

    with pie2:
        if battery_status_df.empty:
            st.markdown("#### Battery")
            st.success("No battery problems")
        else:
            st.plotly_chart(
                plot_battery_status_donut(battery_status_df, "Battery"),
                use_container_width=True,
                key="summary_battery_status",
            )

    with pie3:
        if temperature_severity_df.empty:
            st.markdown("#### Temperature -40")
            st.success("No -40 temperature problems")
        else:
            st.plotly_chart(
                plot_severity_donut(
                    temperature_severity_df,
                    "Temperature -40",
                    range_col="minus_40_count",
                    range_label="-40 readings range",
                    range_decimals=0,
                    range_suffix="",
                ),
                use_container_width=True,
                key="summary_temperature_severity",
            )

    st.caption(
        "Packet Loss and Temperature severity are calculated adaptively from this experiment's own "
        "problem distribution. The only fixed Packet Loss rule is that a sensor must be above 5% "
        "to be considered a problem. Hover shows sensor count and the actual problem range, not pie percentages. "
        "Battery keeps its real decision-tree status names."
    )

    # -----------------------------------------------------
    # Central action table - one row per sensor
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### Sensors Needing Attention")

    if packet_system_info.get("detected", False):
        st.warning(
            "🔄 **Pi-level packet-loss pattern detected:** "
            f"peak {packet_system_info['affected']} / {packet_system_info['total']} sensors affected together "
            f"({_fmt_timestamp(packet_system_info['hour'])}); "
            f"typical simultaneous count is {packet_system_info['typical_affected']:.1f}. "
            "Reset Pi is recommended first for affected sensors unless a battery problem is present."
        )

    unified_problem_df = build_unified_problem_table(
        packet_action_df,
        summary_value_issues_df,
        packet_system_info,
    )

    if unified_problem_df.empty:
        st.success("✅ No action is currently required.")
    else:
        st.dataframe(
            unified_problem_df,
            hide_index=True,
            use_container_width=True,
        )


# =========================================================
# 10B) Packet Loss Analysis tab
# =========================================================
with tab_packet_loss:
    st.subheader(f"Packet Loss Analysis - source: {packet_loss_file.name}")

    if packet_error is not None:
        st.error(f"Error processing packet-loss analysis: {packet_error}")
    else:
        # High packet-loss table / message. No separate title here, to keep the tab cleaner.

        high_loss_df = packet_sensor_loss_df[
            packet_sensor_loss_df["loss_pct"] > PACKET_LOSS_ALERT_PCT
        ].sort_values(by="loss_pct", ascending=False)

        if high_loss_df.empty:
            st.success("✅ All sensors are operating at 5% or less overall packet loss.")
        else:
            st.error(f"🚨 Attention: {len(high_loss_df)} sensor(s) have overall packet loss above 5%.")

            if packet_system_info.get("detected", False):
                st.warning(
                    "🔄 Recommended first action: **Reset Pi**. "
                    f"Peak simultaneous impact: {packet_system_info['affected']} / {packet_system_info['total']} sensors "
                    f"({_fmt_timestamp(packet_system_info['hour'])}); "
                    f"typical simultaneous impact: {packet_system_info['typical_affected']:.1f} sensors."
                )

            display_df = packet_action_df[[
                "sensor",
                "loss_pct",
                "severity",
                "recommended_action",
                "next_action",
                "why",
            ]].rename(columns={
                "sensor": "Sensor",
                "loss_pct": "Loss (%)",
                "severity": "Relative Severity",
                "recommended_action": "Recommended Action",
                "next_action": "Next Step",
                "why": "Why",
            })

            st.dataframe(
                display_df.style.format({"Loss (%)": "{:.2f}%"}),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader(
            "Hourly Packet Loss",
            help=(
                "Data is grouped by hour. If the first/last hour is partial, "
                "expected packets are calculated only for the timestamps that should exist "
                "in that partial hour."
            ),
        )

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
        st.markdown("### Packet Loss Heatmap")
        st.caption(
            "Rows are sensors, columns are hours, and each cell shows packet loss (%). "
            "This helps distinguish a single-sensor problem from a Pi-wide event."
        )
        st.plotly_chart(
            plot_packet_loss_heatmap(packet_rep),
            use_container_width=True,
            key="packet_loss_heatmap",
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

    st.caption(
        "Battery: last 20 values only (<2700 mV = low) • "
        "Temperature: -40°C error values only • Light: no automatic value rule yet"
    )

    st.caption(
        "Battery rules are fixed: the last 20 available values are analyzed and current time is not used."
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
                    battery_threshold_mv=DEFAULT_BATTERY_LOW_MV,
                    battery_last_n=DEFAULT_BATTERY_LAST_N,
                )

                issue_df = result_issues_only(result_df)

                if not issue_df.empty:
                    issue_df = issue_df.copy()
                    issue_df["issue_time_range"] = issue_df.apply(
                        lambda r: build_issue_time_range(r, data_type),
                        axis=1,
                    )

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

                    preferred_cols = [
                        "sensor",
                        "status",
                        "recommended_action",
                        "why",
                        "issue_time_range",
                    ]

                    issue_display_cols = [
                        c for c in preferred_cols if c in issue_df.columns
                    ]

                    st.dataframe(
                        issue_df[issue_display_cols],
                        hide_index=True,
                        use_container_width=True,
                    )

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
                        last_n=DEFAULT_BATTERY_LAST_N if data_type == "Battery" else 100,
                    )

                    if data_type == "Battery":
                        fig_last.add_hline(
                            y=float(DEFAULT_BATTERY_LOW_MV),
                            line_dash="dash",
                            annotation_text=f"Low threshold: {DEFAULT_BATTERY_LOW_MV:.0f} mV",
                        )

                    st.plotly_chart(fig_last, use_container_width=True, key=f"last_values_plot_{file_index}")

                with st.expander("Preview wide data"):
                    st.dataframe(wide_selected.head(100), use_container_width=True)

        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

# =========================================================
# 10D) Maintenance Log - browser local storage
# =========================================================
with tab_maintenance:
    st.subheader("Maintenance Log")
    st.caption(
        "Notes and maintenance actions are stored locally in this browser for this experiment. "
        "Press ✓ only after performing an action. You can edit the action date/time before saving."
    )

    maintenance_sensors = set()

    if not packet_sensor_loss_df.empty:
        maintenance_sensors.update(packet_sensor_loss_df["sensor"].astype(str).tolist())

    if not summary_health_df.empty:
        maintenance_sensors.update(summary_health_df["sensor"].astype(str).tolist())

    maintenance_html = build_maintenance_log_html(
        experiment_name=experiment_name,
        sensors=sorted(maintenance_sensors),
        problem_df=unified_problem_df,
    )

    components.html(
        maintenance_html,
        height=1050,
        scrolling=True,
    )

