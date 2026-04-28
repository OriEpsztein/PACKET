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


# =========================================================
# 1) General helpers
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

    # fallback: any column name that contains time/date
    for c in df.columns:
        cl = str(c).lower()
        if "time" in cl or "date" in cl:
            return c

    raise ValueError("Could not detect a timestamp column. Rename it to 'Timestamp'.")


def _detect_sensor_column(df: pd.DataFrame) -> str | None:
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


def detect_data_type(file_name: str, df: pd.DataFrame | None = None) -> str:
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


def _value_candidates_for_type(data_type: str) -> list[str]:
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


def _choose_value_column(
    df: pd.DataFrame,
    ignore_cols: list[str],
    data_type: str | None = None,
) -> str:
    """Choose the value column for long-format data."""
    if data_type is not None:
        for c in _value_candidates_for_type(data_type):
            if c in df.columns and c not in ignore_cols:
                return c

    possible_cols = [c for c in df.columns if c not in ignore_cols]

    # Prefer numeric columns.
    numeric_cols = [c for c in possible_cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[0]

    # Fallback: try converting columns to numeric and keep the one with most values.
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


def _to_wide_timeseries(
    df: pd.DataFrame,
    ts_col: str,
    data_type: str | None = None,
) -> pd.DataFrame:
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
    value_col = _choose_value_column(df, ignore_cols=[ts_col, sensor_col], data_type=data_type)

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
# 2) Packet-loss computation
# =========================================================
def packet_loss_hourly_sensor_matrix(
    df: pd.DataFrame,
    freq_minutes: int = 3,
    keep_full_hours_only: bool = True,
) -> dict:
    """Compute hourly packet loss per sensor and overall."""
    ts_col = _detect_timestamp_column(df)
    wide = _to_wide_timeseries(df, ts_col)

    sensors = list(wide.columns)
    n_total = len(sensors)
    expected_per_hour = int(round(60 / freq_minutes))

    # Full timestamp grid so missing timestamps count as lost packets.
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

    # Hourly totals across sensors.
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
        "min": hourly_sensor_loss.min(axis=1),
        "max": hourly_sensor_loss.max(axis=1),
    }, index=pd.to_datetime(hourly_overall["hour"]))

    return {
        "hourly_overall": hourly_overall,
        "hourly_sensor_loss": hourly_sensor_loss,
        "stats": stats,
        "n_total": n_total,
    }


def sensor_overall_packet_loss(
    df: pd.DataFrame,
    freq_minutes: int = 3,
    keep_full_hours_only: bool = True,
) -> pd.DataFrame:
    """Overall packet loss per sensor across the selected file."""
    ts_col = _detect_timestamp_column(df)
    wide = _to_wide_timeseries(df, ts_col)

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
            np.repeat(ts_per_hour_used.values[:, None], len(sensors), axis=1),
            index=ts_per_hour_used.index,
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
# 3) Packet-loss plot functions
# =========================================================
def plot_hourly_loss(
    rep: dict,
    view_mode: str,
    selected_sensors: list | None = None,
    show_raw_points: bool = True,
    show_overall_line: bool = False,
    tick_every_hours: int = 4,
    bin_size: float = 0.5,
) -> go.Figure:
    """Main packet-loss plot."""
    hourly = rep["hourly_overall"]
    loss_mat = rep["hourly_sensor_loss"]
    stats = rep["stats"]
    n_total = rep["n_total"]

    hours_index = pd.to_datetime(hourly["hour"])
    hours = hours_index.to_numpy()

    fig = go.Figure()

    def draw_overall_line(marker_size: int = 6):
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
            x=hours,
            y=y_overall,
            mode="lines+markers",
            name="Overall Average",
            line=dict(color="royalblue", width=2),
            marker=dict(size=marker_size, color="royalblue"),
            customdata=custom_line,
            hovertemplate=(
                "Time Window: %{x|%Y-%m-%d %H:%M} to %{x|%H}:59<br>"
                "Packet Loss (%)=%{y:.2f}<br>"
                "packets_received=%{customdata[0]}<br>"
                "expected_packets=%{customdata[1]}<br>"
                "lost_packets=%{customdata[2]}<br>"
                "MEAN=%{customdata[3]:.2f}<br>"
                "MIN=%{customdata[4]:.2f}<br>"
                "MAX=%{customdata[5]:.2f}<extra></extra>"
            ),
        ))

    if view_mode == "Overall Average":
        draw_overall_line(marker_size=6)

    elif view_mode == "Raw Points":
        if show_raw_points:
            rows = []

            for h in hours_index:
                row = loss_mat.loc[h].dropna()
                for sensor_name, value in row.items():
                    value = float(np.clip(value, 0, 100))
                    loss_bin = float(np.round(value / bin_size) * bin_size)
                    rows.append((h, str(sensor_name), value, loss_bin))

            dfp = pd.DataFrame(rows, columns=["hour", "sensor", "loss", "loss_bin"])

            if not dfp.empty:
                dfp["bin_count"] = dfp.groupby(["hour", "loss_bin"])["loss_bin"].transform("size")

                custom_points = np.stack([
                    dfp["sensor"].astype(str).to_numpy(),
                    dfp["bin_count"].to_numpy(),
                    np.full(len(dfp), n_total),
                    dfp["loss_bin"].to_numpy(dtype=float),
                ], axis=1)

                fig.add_trace(go.Scattergl(
                    x=dfp["hour"].to_numpy(),
                    y=dfp["loss"].to_numpy(dtype=float),
                    mode="markers",
                    name="Sensors (raw points)",
                    marker=dict(size=7, color="rgba(255,140,0,0.5)"),
                    customdata=custom_points,
                    hovertemplate=(
                        "Time Window: %{x|%Y-%m-%d %H:%M} to %{x|%H}:59<br>"
                        "Sensor=%{customdata[0]}<br>"
                        "Packet Loss (%)=%{y:.2f}<br>"
                        "Sensors in same bin=%{customdata[1]} / %{customdata[2]}<extra></extra>"
                    ),
                ))

        if show_overall_line:
            draw_overall_line(marker_size=10)

    elif view_mode == "Specific Sensors" and selected_sensors:
        for sensor in selected_sensors:
            if sensor in loss_mat.columns:
                y_sensor = loss_mat[sensor].to_numpy(dtype=float)
                fig.add_trace(go.Scatter(
                    x=hours,
                    y=y_sensor,
                    mode="lines+markers",
                    name=f"Sensor {sensor}",
                    hovertemplate=(
                        "Time Window: %{x|%Y-%m-%d %H:%M} to %{x|%H}:59<br>"
                        "Sensor " + str(sensor) + "<br>"
                        "Packet Loss (%)=%{y:.2f}<extra></extra>"
                    ),
                ))

    tick0 = pd.to_datetime(hours_index.min()).floor("d")
    tick_vals = pd.date_range(
        start=tick0,
        end=pd.to_datetime(hours_index.max()).ceil("h"),
        freq=f"{tick_every_hours}h",
    )
    tick_text = [
        f"{t:%H:%M}<br>{t:%Y-%m-%d}" if t.hour == 0 else f"{t:%H:%M}<br>"
        for t in tick_vals
    ]

    hmode = "closest" if view_mode in ["Raw Points", "Specific Sensors"] else "x unified"

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Hour Start Time",
        yaxis_title="Packet Loss (%)",
        margin=dict(t=30, b=90),
        yaxis=dict(range=[0, 100]),
        hovermode=hmode,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )

    fig.update_xaxes(
        type="date",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=0,
        automargin=True,
    )

    return fig


def plot_sensor_loss_distribution(df_s: pd.DataFrame, bin_size: float = 0.5) -> go.Figure:
    """Histogram of sensor-level packet loss."""
    fig = go.Figure()

    if df_s.empty:
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
# 4) Data-analysis health checks
# =========================================================
def analyze_battery(
    wide: pd.DataFrame,
    low_mv_threshold: float = 2700,
    last_n: int = 20,
    low_count_limit: int = 3,
) -> pd.DataFrame:
    """Battery health rule.

    Rule requested:
    - Take last 20 values.
    - If more than 3 are under 2700 mV -> LOW_BATTERY.
    - If there are only 3 or fewer timestamps to check -> flag if at least 1 is under 2700 mV.
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
        by=["status", "under_threshold_count", "min_last_values_mV"],
        ascending=[True, False, True],
    )


def longest_equal_run(
    series: pd.Series,
    decimals: int = 3,
    ignore_values: set[float] | None = None,
) -> dict:
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
    stuck_run_threshold: int = 4,
    stuck_round_decimals: int = 3,
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

        rows.append({
            "sensor": str(sensor),
            "status": "OK" if not issues else " / ".join(issues),
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
        by=["status", "minus_40_count", "max_stuck_run"],
        ascending=[True, False, False],
    )


def analyze_light(
    wide: pd.DataFrame,
    stuck_run_threshold: int = 4,
    stuck_round_decimals: int = 3,
    ignore_zero_for_light: bool = True,
) -> pd.DataFrame:
    """Light checks: stuck repeated values.

    By default, exact 0 is ignored because light sensors can stay at 0 at night.
    """
    rows = []
    ignore_values = {0.0} if ignore_zero_for_light else None

    for sensor in wide.columns:
        s = pd.to_numeric(wide[sensor], errors="coerce").dropna().sort_index()
        run_info = longest_equal_run(s, decimals=stuck_round_decimals, ignore_values=ignore_values)
        is_stuck = run_info["max_stuck_run"] >= stuck_run_threshold

        rows.append({
            "sensor": str(sensor),
            "status": "STUCK_VALUE" if is_stuck else "OK",
            "values_count": int(len(s)),
            "max_stuck_run": run_info["max_stuck_run"],
            "stuck_value": run_info["stuck_value"],
            "stuck_start_time": run_info["stuck_start_time"],
            "stuck_end_time": run_info["stuck_end_time"],
            "last_value": float(s.iloc[-1]) if len(s) else np.nan,
        })

    return pd.DataFrame(rows).sort_values(
        by=["status", "max_stuck_run"],
        ascending=[True, False],
    )


def run_data_health_check(
    df: pd.DataFrame,
    data_type: str,
    battery_threshold_mv: float,
    battery_last_n: int,
    battery_low_count_limit: int,
    stuck_run_threshold: int,
    stuck_round_decimals: int,
    ignore_zero_for_light: bool,
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
        return analyze_light(
            wide,
            stuck_run_threshold=stuck_run_threshold,
            stuck_round_decimals=stuck_round_decimals,
            ignore_zero_for_light=ignore_zero_for_light,
        )

    return pd.DataFrame()


def count_data_issues(result_df: pd.DataFrame) -> int:
    """Count rows that are not OK."""
    if result_df.empty or "status" not in result_df.columns:
        return 0
    return int((result_df["status"] != "OK").sum())


# =========================================================
# 5) Extra plotting for data analysis
# =========================================================
def plot_last_values_for_sensors(
    wide: pd.DataFrame,
    sensors: list[str],
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
# 6) Sidebar - Data Settings
# =========================================================
st.title("Field 4D: Sensor Data Analyzer")

st.sidebar.header("Data Settings")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV / CSVs",
    type=["csv"],
    accept_multiple_files=True,
    help="Upload one or several CSV files. Then choose which file to analyze below.",
)

FREQ_MINUTES = st.sidebar.number_input(
    "Expected sampling interval (minutes)",
    min_value=1,
    max_value=60,
    value=3,
    step=1,
)

keep_full = st.sidebar.checkbox(
    "Show full hours only",
    value=True,
    help="Ignores partial hours at the start or end of your dataset so percentages are based on full hourly windows.",
)

tick_every_hours = st.sidebar.slider(
    "X-axis tick every N hours",
    min_value=1,
    max_value=24,
    value=4,
    step=1,
)

if not uploaded_files:
    st.info("Upload one or more CSV files from the left sidebar to begin analysis.")
    st.stop()

# Unique labels even if two files have the same name.
file_labels = [f"{i + 1}. {file.name}" for i, file in enumerate(uploaded_files)]
selected_label = st.sidebar.selectbox("Choose CSV to analyze", file_labels)
selected_index = file_labels.index(selected_label)
selected_file = uploaded_files[selected_index]
selected_df = read_uploaded_csv(selected_file)

# Display selected dataset range in the sidebar.
try:
    selected_info = get_basic_file_info(selected_file)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Selected Dataset Range:**")
    st.sidebar.text(f"Start: {selected_info['start']:%Y-%m-%d %H:%M}")
    st.sidebar.text(f"End:   {selected_info['end']:%Y-%m-%d %H:%M}")
    st.sidebar.text(f"Sensors: {selected_info['sensors']}")
    st.sidebar.text(f"Auto type: {selected_info['auto_type']}")
except Exception as e:
    st.sidebar.error(f"Could not read selected file info: {e}")


# =========================================================
# 7) Tabs
# =========================================================
tab_summary, tab_packet_loss, tab_data_analysis = st.tabs([
    "Summary",
    "Packet Loss Analysis",
    "Data Analysis",
])


# =========================================================
# 7A) Summary tab - all uploaded files
# =========================================================
with tab_summary:
    st.subheader("Summary - All Uploaded CSVs")

    summary_rows = []

    # Default analysis settings for summary.
    summary_battery_threshold_mv = 2700
    summary_battery_last_n = 20
    summary_battery_low_count_limit = 3
    summary_stuck_run_threshold = 4
    summary_stuck_round_decimals = 3
    summary_ignore_zero_for_light = True

    for file in uploaded_files:
        row = {
            "file": file.name,
            "rows": np.nan,
            "sensors": np.nan,
            "start": pd.NaT,
            "end": pd.NaT,
            "auto_type": "Unknown",
            "overall_packet_loss_%": np.nan,
            "sensors_packet_loss_>5%": np.nan,
            "data_issues_count": np.nan,
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

            # Packet loss summary.
            rep_summary = packet_loss_hourly_sensor_matrix(
                raw,
                freq_minutes=int(FREQ_MINUTES),
                keep_full_hours_only=keep_full,
            )
            hourly = rep_summary["hourly_overall"]
            expected_total = hourly["expected_packets"].sum()
            lost_total = hourly["lost_packets"].sum()
            row["overall_packet_loss_%"] = (
                lost_total / expected_total * 100 if expected_total > 0 else np.nan
            )

            sensor_loss_df = sensor_overall_packet_loss(
                raw,
                freq_minutes=int(FREQ_MINUTES),
                keep_full_hours_only=keep_full,
            )
            row["sensors_packet_loss_>5%"] = int((sensor_loss_df["loss_pct"] > 5).sum())

            # Data issues summary according to auto-detected type.
            if auto_type in ["Battery", "Temperature", "Light"]:
                health_df = run_data_health_check(
                    raw,
                    data_type=auto_type,
                    battery_threshold_mv=summary_battery_threshold_mv,
                    battery_last_n=summary_battery_last_n,
                    battery_low_count_limit=summary_battery_low_count_limit,
                    stuck_run_threshold=summary_stuck_run_threshold,
                    stuck_round_decimals=summary_stuck_round_decimals,
                    ignore_zero_for_light=summary_ignore_zero_for_light,
                )
                row["data_issues_count"] = count_data_issues(health_df)
            else:
                row["data_issues_count"] = np.nan

        except Exception as e:
            row["status"] = f"ERROR: {e}"

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    st.dataframe(
        summary_df.style.format({
            "overall_packet_loss_%": "{:.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Uploaded files", len(uploaded_files))
    with c2:
        st.metric("Selected file", selected_file.name)
    with c3:
        ok_files = int((summary_df["status"] == "OK").sum())
        st.metric("Files read successfully", ok_files)
    with c4:
        total_issues = pd.to_numeric(summary_df["data_issues_count"], errors="coerce").sum()
        st.metric("Detected data issues", int(total_issues))

    st.info(
        "Summary uses automatic data-type detection. In the Data Analysis tab you can manually choose Battery, Temperature, Light, or Other for the selected file."
    )


# =========================================================
# 7B) Packet Loss Analysis tab - selected file
# =========================================================
with tab_packet_loss:
    st.subheader(f"Packet Loss Analysis - {selected_file.name}")

    try:
        rep = packet_loss_hourly_sensor_matrix(
            selected_df,
            freq_minutes=int(FREQ_MINUTES),
            keep_full_hours_only=keep_full,
        )

        df_sensor_overall = sensor_overall_packet_loss(
            selected_df,
            freq_minutes=int(FREQ_MINUTES),
            keep_full_hours_only=keep_full,
        )

        st.success("File loaded and packet-loss analysis completed successfully.")

        # --- High Packet Loss Sensors Alert ---
        st.markdown("### High Packet Loss Alert")
        high_loss_df = df_sensor_overall[df_sensor_overall["loss_pct"] > 5.0].sort_values(
            by="loss_pct",
            ascending=False,
        )

        if not high_loss_df.empty:
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
        else:
            st.success("✅ All sensors are operating at 5% or less overall packet loss.")

        st.markdown("---")
        st.markdown("### Hourly Packet Loss")
        st.info(
            "🕒 Data is grouped by hour. A point labeled 21:00 represents the interval from 21:00 to 21:59."
        )

        view_mode = st.radio(
            "Display Mode:",
            ["Overall Average", "Raw Points", "Specific Sensors"],
            horizontal=True,
        )

        selected_sensors = []
        show_raw = True
        show_overall = False

        if view_mode == "Specific Sensors":
            sensor_list = rep["hourly_sensor_loss"].columns.tolist()
            selected_sensors = st.multiselect(
                "Select sensors to display:",
                sensor_list,
                default=sensor_list,
            )

        elif view_mode == "Raw Points":
            col1, col2 = st.columns(2)
            with col1:
                show_raw = st.checkbox("Show Raw Sensor Points", value=True)
            with col2:
                show_overall = st.checkbox("Overlay Overall Average Line", value=True)

        fig_overall = plot_hourly_loss(
            rep,
            view_mode,
            selected_sensors=selected_sensors,
            show_raw_points=show_raw,
            show_overall_line=show_overall,
            tick_every_hours=int(tick_every_hours),
        )
        st.plotly_chart(fig_overall, use_container_width=True)

        st.markdown("---")
        fig_dist = plot_sensor_loss_distribution(df_sensor_overall)
        st.plotly_chart(fig_dist, use_container_width=True)

        with st.expander("Show full sensor packet-loss table"):
            st.dataframe(
                df_sensor_overall.style.format({"loss_pct": "{:.2f}%"}),
                hide_index=True,
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Error processing packet-loss analysis: {e}")


# =========================================================
# 7C) Data Analysis tab - selected file
# =========================================================
with tab_data_analysis:
    st.subheader(f"Data Analysis - {selected_file.name}")

    try:
        auto_type = detect_data_type(selected_file.name, selected_df)
        data_type_options = ["Battery", "Temperature", "Light", "Other"]
        default_type = auto_type if auto_type in data_type_options else "Other"

        c1, c2 = st.columns([1, 2])
        with c1:
            data_type = st.selectbox(
                "Choose data type for this CSV",
                data_type_options,
                index=data_type_options.index(default_type),
            )
        with c2:
            st.info(f"Auto-detected type: {auto_type}")

        # Settings for the checks.
        with st.expander("Data Analysis Settings", expanded=True):
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                battery_threshold_mv = st.number_input(
                    "Battery low threshold (mV)",
                    min_value=0,
                    max_value=5000,
                    value=2700,
                    step=50,
                )
            with col_b2:
                battery_last_n = st.number_input(
                    "Battery: check last N values",
                    min_value=1,
                    max_value=200,
                    value=20,
                    step=1,
                )
            with col_b3:
                battery_low_count_limit = st.number_input(
                    "Battery: allowed low values",
                    min_value=0,
                    max_value=20,
                    value=3,
                    step=1,
                )

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                stuck_run_threshold = st.number_input(
                    "Stuck-value run threshold",
                    min_value=2,
                    max_value=100,
                    value=4,
                    step=1,
                    help="Example: threshold 4 flags values like 23, 23, 23, 23.",
                )
            with col_s2:
                stuck_round_decimals = st.number_input(
                    "Round decimals for stuck check",
                    min_value=0,
                    max_value=6,
                    value=3,
                    step=1,
                )
            with col_s3:
                ignore_zero_for_light = st.checkbox(
                    "Ignore 0 for light stuck check",
                    value=True,
                    help="Useful because light can stay exactly 0 during the night.",
                )

        ts_col = _detect_timestamp_column(selected_df)
        wide_selected = _to_wide_timeseries(selected_df, ts_col, data_type=data_type)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Rows", len(selected_df))
        with m2:
            st.metric("Sensors", len(wide_selected.columns))
        with m3:
            st.metric("Start", f"{wide_selected.index.min():%Y-%m-%d %H:%M}")
        with m4:
            st.metric("End", f"{wide_selected.index.max():%Y-%m-%d %H:%M}")

        if data_type == "Other":
            st.warning(
                "Choose Battery, Temperature, or Light to run automatic checks. "
                "This file can still be viewed in the preview below."
            )
            with st.expander("Preview wide data"):
                st.dataframe(wide_selected.head(100), use_container_width=True)
            st.stop()

        result_df = run_data_health_check(
            selected_df,
            data_type=data_type,
            battery_threshold_mv=float(battery_threshold_mv),
            battery_last_n=int(battery_last_n),
            battery_low_count_limit=int(battery_low_count_limit),
            stuck_run_threshold=int(stuck_run_threshold),
            stuck_round_decimals=int(stuck_round_decimals),
            ignore_zero_for_light=bool(ignore_zero_for_light),
        )

        issue_df = result_df[result_df["status"] != "OK"].copy()

        st.markdown("---")
        st.markdown("### Sensor Health Results")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Sensors checked", len(result_df))
        with c2:
            st.metric("Sensors with issues", len(issue_df))
        with c3:
            st.metric("OK sensors", int((result_df["status"] == "OK").sum()))

        if issue_df.empty:
            st.success("✅ No data issues detected for the selected rules.")
        else:
            st.error(f"🚨 {len(issue_df)} sensor(s) have data issues.")
            st.dataframe(issue_df, hide_index=True, use_container_width=True)

        with st.expander("Show full health-check table", expanded=issue_df.empty):
            st.dataframe(result_df, hide_index=True, use_container_width=True)

        # Plot selected sensors.
        st.markdown("---")
        st.markdown("### Plot Last Values")

        if not issue_df.empty:
            default_plot_sensors = issue_df["sensor"].astype(str).head(10).tolist()
        else:
            default_plot_sensors = [str(c) for c in wide_selected.columns[:10]]

        plot_sensors = st.multiselect(
            "Choose sensors to plot",
            [str(c) for c in wide_selected.columns],
            default=default_plot_sensors,
        )

        # Convert columns to string only for plotting selection compatibility.
        wide_for_plot = wide_selected.copy()
        wide_for_plot.columns = wide_for_plot.columns.astype(str)

        if plot_sensors:
            fig_last = plot_last_values_for_sensors(
                wide_for_plot,
                sensors=plot_sensors,
                title=f"Last values - {data_type}",
                last_n=int(battery_last_n) if data_type == "Battery" else 100,
            )

            # Add battery threshold line when relevant.
            if data_type == "Battery":
                fig_last.add_hline(
                    y=float(battery_threshold_mv),
                    line_dash="dash",
                    annotation_text=f"Low threshold: {battery_threshold_mv:.0f} mV",
                )

            st.plotly_chart(fig_last, use_container_width=True)

        with st.expander("Preview wide data"):
            st.dataframe(wide_selected.head(100), use_container_width=True)

    except Exception as e:
        st.error(f"Error processing data analysis: {e}")
