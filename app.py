from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math

import pandas as pd
import streamlit as st

from stock_dashboard.charts import drawdown_chart, forecast_chart, price_chart, returns_histogram, volume_chart
from stock_dashboard.data import (
    ColumnMap,
    available_tickers,
    date_bounds,
    infer_column_map,
    load_tabular,
    parquet_columns,
    standardize_prices,
)
from stock_dashboard.forecast import build_forecast
from stock_dashboard.metrics import compute_kpis, format_money, format_number, format_percent
from stock_dashboard.reporting import build_markdown_report


DEFAULT_DATA_PATH = Path("full_data.parquet")


st.set_page_config(
    page_title="Stock Forecast Intelligence",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
        [data-testid="stSidebar"] {background: #f7f8fa;}
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e6e8ec;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        [data-testid="stMetricLabel"] {color: #667085;}
        [data-testid="stMetricValue"] {font-size: 1.55rem;}
        .section-note {
            color: #667085;
            font-size: 0.92rem;
            margin-top: -0.65rem;
            margin-bottom: 0.6rem;
        }
        .report-box {
            border: 1px solid #e6e8ec;
            border-radius: 8px;
            padding: 18px;
            background: #ffffff;
        }
        div[data-testid="stDataFrame"] {border: 1px solid #e6e8ec; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_local_data(path: str, columns: tuple[str, ...] | None) -> pd.DataFrame:
    return load_tabular(Path(path), columns=list(columns) if columns else None)


def select_index(options: list[str], value: str | None) -> int:
    if value in options:
        return options.index(value)
    return 0


def metric_delta(value: float) -> str | None:
    if value is None or math.isnan(float(value)):
        return None
    return format_percent(float(value))


def render_metric(label: str, value: str, delta: float | None = None, inverse: bool = False) -> None:
    delta_text = metric_delta(delta) if delta is not None else None
    st.metric(label, value, delta=delta_text, delta_color="inverse" if inverse else "normal")


def build_column_map_controls(columns: list[str], inferred: ColumnMap) -> ColumnMap:
    options = [""] + columns
    with st.sidebar.expander("Column mapping", expanded=False):
        date_col = st.selectbox("Date", options, index=select_index(options, inferred.date))
        ticker_col = st.selectbox("Ticker", options, index=select_index(options, inferred.ticker))
        open_col = st.selectbox("Open", options, index=select_index(options, inferred.open))
        high_col = st.selectbox("High", options, index=select_index(options, inferred.high))
        low_col = st.selectbox("Low", options, index=select_index(options, inferred.low))
        close_col = st.selectbox("Close / Price", options, index=select_index(options, inferred.close))
        volume_col = st.selectbox("Volume", options, index=select_index(options, inferred.volume))
    return ColumnMap(
        date=date_col or None,
        ticker=ticker_col or None,
        open=open_col or None,
        high=high_col or None,
        low=low_col or None,
        close=close_col or None,
        volume=volume_col or None,
    )


def source_columns(uploaded_file, local_path: Path) -> tuple[list[str], pd.DataFrame | None]:
    if uploaded_file is not None:
        uploaded_df = load_tabular(uploaded_file)
        return [str(column) for column in uploaded_df.columns], uploaded_df

    if local_path.exists() and local_path.suffix.lower() in {".parquet", ".pq"}:
        columns = parquet_columns(local_path)
        if columns:
            return columns, None

    if local_path.exists():
        loaded = load_tabular(local_path)
        return [str(column) for column in loaded.columns], loaded

    return [], None


with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Dataset", type=["parquet", "pq", "csv", "tsv", "xlsx", "xls"])
    path_value = st.text_input("Local path", value=str(DEFAULT_DATA_PATH) if DEFAULT_DATA_PATH.exists() else "")
    local_path = Path(path_value) if path_value else DEFAULT_DATA_PATH

try:
    columns, uploaded_or_loaded_df = source_columns(uploaded, local_path)
except Exception as exc:
    st.error(f"Cannot read dataset: {exc}")
    st.stop()

if not columns:
    st.title("Stock Forecast Intelligence")
    st.info("Add a parquet/csv/xlsx dataset in the sidebar to start.")
    st.stop()

inferred_map = infer_column_map(columns)
column_map = build_column_map_controls(columns, inferred_map)
required_columns = tuple(column_map.selected_columns())

try:
    if uploaded_or_loaded_df is not None:
        raw_df = uploaded_or_loaded_df[list(required_columns)] if required_columns else uploaded_or_loaded_df
    else:
        raw_df = load_local_data(str(local_path), required_columns or None)
except Exception as exc:
    st.error(f"Cannot load selected columns: {exc}")
    st.stop()

tickers = available_tickers(raw_df, column_map.ticker)
with st.sidebar:
    if tickers:
        default_ticker = "AAPL" if "AAPL" in tickers else tickers[0]
        selected_ticker = st.selectbox("Ticker", tickers, index=tickers.index(default_ticker))
    else:
        selected_ticker = None
    min_date, max_date = date_bounds(raw_df, column_map.date)
    if min_date is not None and max_date is not None:
        picked_range = st.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
    else:
        picked_range = None

    st.header("Forecast")
    horizon = st.slider("Horizon", min_value=5, max_value=180, value=30, step=5)
    training_window = st.slider("Training window", min_value=60, max_value=1000, value=252, step=21)

start_date = end_date = None
if isinstance(picked_range, tuple) and len(picked_range) == 2:
    start_date, end_date = pd.Timestamp(picked_range[0]), pd.Timestamp(picked_range[1])

try:
    price_df = standardize_prices(raw_df, column_map, ticker=selected_ticker, start=start_date, end=end_date)
except Exception as exc:
    st.error(f"Cannot prepare price series: {exc}")
    st.stop()

if len(price_df) < 12:
    st.error("Need at least 12 clean observations after filtering.")
    st.stop()

forecast_result = build_forecast(price_df, horizon=horizon, training_window=training_window)
kpis = compute_kpis(price_df)
ticker_label = selected_ticker or "Portfolio / Series"
report_markdown = build_markdown_report(
    ticker=ticker_label,
    kpis=kpis,
    forecast_summary=forecast_result.summary,
    forecast=forecast_result.forecast,
    generated_at=datetime.now(),
)

st.title("Stock Forecast Intelligence")
st.caption("Professional equity dashboard with scenario forecast, risk profile, and downloadable research note.")

top_left, top_right = st.columns([0.72, 0.28])
with top_left:
    st.subheader(ticker_label)
    st.markdown(
        f"<div class='section-note'>{kpis['start_date']:%Y-%m-%d} to {kpis['end_date']:%Y-%m-%d} - "
        f"{format_number(kpis['observations'])} observations</div>",
        unsafe_allow_html=True,
    )
with top_right:
    st.download_button(
        "Download report",
        data=report_markdown,
        file_name=f"{ticker_label.replace('/', '_').replace(' ', '_').lower()}_forecast_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

metric_cols = st.columns(6)
with metric_cols[0]:
    render_metric("Latest close", format_money(kpis["last_close"]), kpis["daily_return"])
with metric_cols[1]:
    render_metric("Forecast target", format_money(forecast_result.summary["target_price"]), forecast_result.summary["expected_return"])
with metric_cols[2]:
    render_metric("1M return", format_percent(kpis["return_1m"]))
with metric_cols[3]:
    render_metric("Volatility", format_percent(kpis["volatility"]), inverse=True)
with metric_cols[4]:
    render_metric("Max drawdown", format_percent(kpis["max_drawdown"]), inverse=True)
with metric_cols[5]:
    render_metric("Backtest MAPE", format_percent(forecast_result.summary["backtest_mape"]), inverse=True)

overview_tab, forecast_tab, risk_tab, report_tab = st.tabs(["Overview", "Forecast", "Risk", "Report"])

with overview_tab:
    st.plotly_chart(price_chart(price_df), use_container_width=True)
    if "Volume" in price_df.columns:
        st.plotly_chart(volume_chart(price_df), use_container_width=True)

with forecast_tab:
    st.plotly_chart(forecast_chart(price_df, forecast_result.forecast), use_container_width=True)
    st.dataframe(
        forecast_result.forecast.assign(Date=lambda frame: frame["Date"].dt.date).tail(20),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Forecast": st.column_config.NumberColumn(format="$%.2f"),
            "Lower_80": st.column_config.NumberColumn(format="$%.2f"),
            "Upper_80": st.column_config.NumberColumn(format="$%.2f"),
            "Lower_95": st.column_config.NumberColumn(format="$%.2f"),
            "Upper_95": st.column_config.NumberColumn(format="$%.2f"),
            "Bear": st.column_config.NumberColumn(format="$%.2f"),
            "Bull": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

with risk_tab:
    risk_left, risk_right = st.columns(2)
    with risk_left:
        st.plotly_chart(drawdown_chart(price_df), use_container_width=True)
    with risk_right:
        st.plotly_chart(returns_histogram(price_df), use_container_width=True)

    quality_cols = st.columns(4)
    with quality_cols[0]:
        render_metric("Sharpe proxy", f"{kpis['sharpe']:.2f}" if not math.isnan(kpis["sharpe"]) else "n/a")
    with quality_cols[1]:
        render_metric("YTD return", format_percent(kpis["return_ytd"]))
    with quality_cols[2]:
        render_metric("52W high", format_money(kpis["high_52w"]))
    with quality_cols[3]:
        render_metric("Avg volume 30D", format_number(kpis["avg_volume_30d"]))

    if not forecast_result.backtest.empty:
        st.dataframe(
            forecast_result.backtest.assign(Date=lambda frame: frame["Date"].dt.date).tail(30),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Actual": st.column_config.NumberColumn(format="$%.2f"),
                "Predicted": st.column_config.NumberColumn(format="$%.2f"),
                "Error": st.column_config.NumberColumn(format="$%.2f"),
                "Absolute_Percent_Error": st.column_config.NumberColumn(format="%.2%"),
            },
        )

with report_tab:
    st.markdown("<div class='report-box'>", unsafe_allow_html=True)
    st.markdown(report_markdown)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Forecast is decision support only and is not investment advice.")
