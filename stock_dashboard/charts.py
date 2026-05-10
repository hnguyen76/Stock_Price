from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .metrics import drawdown_frame


PALETTE = {
    "ink": "#17202a",
    "muted": "#667085",
    "teal": "#0f8b8d",
    "amber": "#f59e0b",
    "coral": "#ef6351",
    "green": "#2ca58d",
    "line": "#344054",
    "band": "rgba(15, 139, 141, 0.14)",
    "band95": "rgba(239, 99, 81, 0.10)",
}


def _layout(fig: go.Figure, height: int = 460, yaxis_title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=24, r=24, t=34, b=24),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="Inter, Segoe UI, sans-serif", color=PALETTE["ink"]),
        yaxis_title=yaxis_title,
        xaxis_title=None,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#edf2f7")
    return fig


def price_chart(price_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if {"Open", "High", "Low", "Close"}.issubset(price_df.columns):
        fig.add_trace(
            go.Candlestick(
                x=price_df["Date"],
                open=price_df["Open"],
                high=price_df["High"],
                low=price_df["Low"],
                close=price_df["Close"],
                name="OHLC",
                increasing_line_color=PALETTE["green"],
                decreasing_line_color=PALETTE["coral"],
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=price_df["Date"],
                y=price_df["Close"],
                mode="lines",
                name="Close",
                line=dict(color=PALETTE["line"], width=2),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=price_df["Date"],
            y=price_df["Close"].rolling(20, min_periods=5).mean(),
            mode="lines",
            name="MA20",
            line=dict(color=PALETTE["teal"], width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=price_df["Date"],
            y=price_df["Close"].rolling(60, min_periods=10).mean(),
            mode="lines",
            name="MA60",
            line=dict(color=PALETTE["amber"], width=1.6),
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _layout(fig, yaxis_title="Price")


def forecast_chart(price_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_df["Date"],
            y=price_df["Close"],
            mode="lines",
            name="History",
            line=dict(color=PALETTE["line"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Upper_95"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Lower_95"],
            mode="lines",
            fill="tonexty",
            fillcolor=PALETTE["band95"],
            line=dict(width=0),
            name="95% range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Upper_80"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Lower_80"],
            mode="lines",
            fill="tonexty",
            fillcolor=PALETTE["band"],
            line=dict(width=0),
            name="80% range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecast"],
            mode="lines",
            name="Base forecast",
            line=dict(color=PALETTE["teal"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Bear"],
            mode="lines",
            name="Bear",
            line=dict(color=PALETTE["coral"], width=1.4, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Bull"],
            mode="lines",
            name="Bull",
            line=dict(color=PALETTE["amber"], width=1.4, dash="dot"),
        )
    )
    return _layout(fig, height=500, yaxis_title="Price")


def drawdown_chart(price_df: pd.DataFrame) -> go.Figure:
    drawdown = drawdown_frame(price_df)
    fig = go.Figure(
        go.Scatter(
            x=drawdown["Date"],
            y=drawdown["Drawdown"],
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color=PALETTE["coral"], width=2),
        )
    )
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, height=320, yaxis_title="Drawdown")


def returns_histogram(price_df: pd.DataFrame) -> go.Figure:
    returns = price_df["Close"].pct_change().dropna()
    fig = go.Figure(
        go.Histogram(
            x=returns,
            nbinsx=60,
            marker_color=PALETTE["teal"],
            opacity=0.82,
            name="Daily return",
        )
    )
    fig.update_xaxes(tickformat=".1%")
    return _layout(fig, height=320, yaxis_title="Frequency")


def volume_chart(price_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=price_df["Date"],
            y=price_df["Volume"],
            name="Volume",
            marker_color=PALETTE["muted"],
            opacity=0.72,
        )
    )
    return _layout(fig, height=260, yaxis_title="Volume")

