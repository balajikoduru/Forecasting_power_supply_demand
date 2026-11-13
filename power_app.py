
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ==================================================
# 1. FILE PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

FORECAST_PATH = (
    PROJECT_DIR / "artifacts" / "power_demand_forecast.csv"
)


# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Electricity Demand Forecast",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Electricity Power Demand Forecasting")

st.caption(
    "Precomputed Linear Regression Forecast | "
    "30 Days | 720 Hourly Predictions"
)


# ==================================================
# 3. LOAD PRECOMPUTED FORECAST
# ==================================================

@st.cache_data
def load_forecast():
    """Load and validate the forecast exported by the notebook."""

    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {FORECAST_PATH}. "
            "Run the notebook through its final export cell first."
        )

    forecast = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["Datetime"]
    )

    required_columns = [
        "Datetime",
        "Forecast_PJMW_MW"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in forecast.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Forecast CSV is missing columns: {missing_columns}"
        )

    forecast = forecast[required_columns].copy()

    forecast["Datetime"] = pd.to_datetime(
        forecast["Datetime"],
        errors="coerce"
    )

    forecast["Forecast_PJMW_MW"] = pd.to_numeric(
        forecast["Forecast_PJMW_MW"],
        errors="coerce"
    )

    forecast = (
        forecast
        .dropna(subset=required_columns)
        .sort_values("Datetime")
        .drop_duplicates("Datetime", keep="last")
        .reset_index(drop=True)
    )

    if len(forecast) != 720:
        raise ValueError(
            f"Expected 720 hourly forecast rows, "
            f"but found {len(forecast)}. "
            "Re-run the notebook forecast and export cells."
        )

    if not forecast["Datetime"].is_monotonic_increasing:
        raise ValueError(
            "Forecast timestamps are not in chronological order."
        )

    if forecast["Datetime"].duplicated().any():
        raise ValueError(
            "Forecast contains duplicate timestamps."
        )

    intervals = forecast["Datetime"].diff().dropna()

    if not intervals.eq(pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Forecast timestamps are not consecutive hourly values."
        )

    return forecast


# ==================================================
# 4. LOAD DATA WITH ERROR HANDLING
# ==================================================

try:
    forecast_df = load_forecast()

except Exception as exc:

    st.error(
        f"Unable to load the precomputed forecast: {exc}"
    )

    st.info(
        "Open Power_Demand_Forecasting.ipynb, run it through "
        "the final export cell, and ensure the artifacts folder "
        "is located beside app.py."
    )

    st.stop()


# ==================================================
# 5. FORECAST PERIOD SLIDER
# ==================================================

st.subheader("Forecast Period")

forecast_days = st.slider(
    "Select forecast duration (days)",
    min_value=1,
    max_value=30,
    value=30,
    step=1,
)

selected_hours = forecast_days * 24

# Filter the forecast according to the slider
selected_forecast = (
    forecast_df
    .head(selected_hours)
    .copy()
)

st.caption(
    f"Selected forecast: {forecast_days} days "
    f"({selected_hours} hourly predictions)"
)


# ==================================================
# 6. KPI CALCULATIONS
# ==================================================

# Full 30-day forecast
all_values = forecast_df["Forecast_PJMW_MW"]

total_30day_energy = all_values.sum()

average_30day_demand = all_values.mean()

# Fixed first 7-day forecast
weekly_forecast = forecast_df.head(168)

total_7day_energy = (
    weekly_forecast["Forecast_PJMW_MW"].sum()
)

# Selected forecast period
selected_values = (
    selected_forecast["Forecast_PJMW_MW"]
)

selected_energy = selected_values.sum()

selected_average = selected_values.mean()

selected_minimum = selected_values.min()

selected_maximum = selected_values.max()


# ==================================================
# 7. MAIN FORECAST KPI CARDS
# ==================================================

st.subheader("Forecast Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "30-Day Forecast Energy",
    f"{total_30day_energy:,.0f} MWh"
)

col2.metric(
    "7-Day Forecast Energy",
    f"{total_7day_energy:,.0f} MWh"
)

col3.metric(
    "Selected Forecast Energy",
    f"{selected_energy:,.0f} MWh"
)

col4.metric(
    "Selected Forecast Duration",
    f"{forecast_days} Days"
)


# ==================================================
# 8. SELECTED PERIOD DEMAND METRICS
# ==================================================

st.subheader("Selected Period Demand")

col5, col6, col7 = st.columns(3)

col5.metric(
    "Average Demand",
    f"{selected_average:,.2f} MW"
)

col6.metric(
    "Minimum Demand",
    f"{selected_minimum:,.2f} MW"
)

col7.metric(
    "Maximum Demand",
    f"{selected_maximum:,.2f} MW"
)


st.divider()


# ==================================================
# 9. WEEKDAY-WISE FORECAST KPI CARDS
# ==================================================

st.subheader("Weekday-wise Forecast Demand")

st.caption(
    "Average predicted demand for each weekday "
    "within the selected forecast period."
)

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

# Extract weekday names
selected_forecast["Weekday"] = (
    selected_forecast["Datetime"].dt.day_name()
)

# Calculate average demand for each weekday
weekday_average = (
    selected_forecast
    .groupby("Weekday")["Forecast_PJMW_MW"]
    .mean()
    .reindex(weekday_order)
)


# First row: Monday to Thursday
weekday_cols_1 = st.columns(4)

for col, day in zip(
    weekday_cols_1,
    weekday_order[:4]
):

    value = weekday_average[day]

    with col:

        if pd.notna(value):
            st.metric(
                label=day,
                value=f"{value:,.2f} MW"
            )
        else:
            st.metric(
                label=day,
                value="N/A"
            )


# Second row: Friday to Sunday
weekday_cols_2 = st.columns(3)

for col, day in zip(
    weekday_cols_2,
    weekday_order[4:]
):

    value = weekday_average[day]

    with col:

        if pd.notna(value):
            st.metric(
                label=day,
                value=f"{value:,.2f} MW"
            )
        else:
            st.metric(
                label=day,
                value="N/A"
            )


st.divider()


# ==================================================
# 10. 30-DAY FORECAST CHART
# ==================================================

st.subheader("Electricity Demand Forecast")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=selected_forecast["Datetime"],
        y=selected_forecast["Forecast_PJMW_MW"],
        name="Forecast Demand",
        mode="lines",
        line=dict(width=2),
        hovertemplate=(
            "<b>Date:</b> %{x|%d %b %Y, %H:%M}<br>"
            "<b>Demand:</b> %{y:,.2f} MW"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="Datetime",
    yaxis_title="Forecast Demand (MW)",
    hovermode="x unified",
    height=500,
    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20
    ),
    template="plotly_white"
)

fig.update_xaxes(showgrid=True)

fig.update_yaxes(showgrid=True)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# 11. WEEKDAY-WISE DEMAND CHART
# ==================================================

st.subheader("Average Demand by Weekday")

weekday_chart = go.Figure()

weekday_chart.add_trace(
    go.Bar(
        x=weekday_order,
        y=weekday_average.values,
        name="Average Demand",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Average Demand: %{y:,.2f} MW"
            "<extra></extra>"
        )
    )
)

weekday_chart.update_layout(
    xaxis_title="Day of Week",
    yaxis_title="Average Demand (MW)",
    height=400,
    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20
    ),
    template="plotly_white"
)

st.plotly_chart(
    weekday_chart,
    use_container_width=True
)


st.divider()


# ==================================================
# 12. HOURLY FORECAST DATA TABLE
# ==================================================

st.subheader("Hourly Forecast Data")

st.caption(
    f"Records: {len(selected_forecast)} | "
    f"Start: "
    f"{selected_forecast['Datetime'].iloc[0]:%Y-%m-%d %H:%M} | "
    f"End: "
    f"{selected_forecast['Datetime'].iloc[-1]:%Y-%m-%d %H:%M}"
)

display_df = selected_forecast[
    ["Datetime", "Forecast_PJMW_MW"]
].copy()

display_df.columns = [
    "Datetime",
    "Forecast Demand (MW)"
]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 13. DOWNLOAD FORECAST CSV
# ==================================================

csv_data = display_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇ Download Forecast CSV",
    data=csv_data,
    file_name=f"power_demand_forecast_{forecast_days}_days.csv",
    mime="text/csv"
)


# ==================================================
# 14. FOOTER
# ==================================================

st.divider()

st.caption(
    "Electricity Power Demand Forecasting | "
    "Linear Regression | PJMW Dataset"
)

st.caption(
    "This dashboard displays the precomputed forecast "
    "generated and exported by the Jupyter Notebook. "
    "It does not train the model or generate new predictions."
)
