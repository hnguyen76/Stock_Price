from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import BinaryIO, Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class ColumnMap:
    date: str | None = None
    ticker: str | None = None
    open: str | None = None
    high: str | None = None
    low: str | None = None
    close: str | None = None
    volume: str | None = None

    def selected_columns(self) -> list[str]:
        values = [self.date, self.ticker, self.open, self.high, self.low, self.close, self.volume]
        return list(dict.fromkeys([value for value in values if value]))


DATE_CANDIDATES = (
    "date",
    "datetime",
    "timestamp",
    "time",
    "pricedate",
    "trade_date",
    "tradedate",
    "day",
)
TICKER_CANDIDATES = (
    "ticker",
    "symbol",
    "stock",
    "asset",
    "name",
    "code",
    "security",
)
OPEN_CANDIDATES = ("open", "openingprice", "openprice", "o")
HIGH_CANDIDATES = ("high", "highprice", "h")
LOW_CANDIDATES = ("low", "lowprice", "l")
CLOSE_CANDIDATES = (
    "close",
    "adjclose",
    "adjustedclose",
    "closeprice",
    "last",
    "lastprice",
    "price",
    "settle",
    "value",
    "c",
)
VOLUME_CANDIDATES = ("volume", "vol", "shares", "turnover", "tradingvolume", "v")


def normalize_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _pick(columns: Sequence[str], candidates: Iterable[str]) -> str | None:
    normalized = {normalize_name(column): column for column in columns}
    for candidate in candidates:
        exact = normalize_name(candidate)
        if exact in normalized:
            return normalized[exact]
    for candidate in candidates:
        needle = normalize_name(candidate)
        for column in columns:
            if needle and needle in normalize_name(column):
                return column
    return None


def infer_column_map(columns: Sequence[str]) -> ColumnMap:
    return ColumnMap(
        date=_pick(columns, DATE_CANDIDATES),
        ticker=_pick(columns, TICKER_CANDIDATES),
        open=_pick(columns, OPEN_CANDIDATES),
        high=_pick(columns, HIGH_CANDIDATES),
        low=_pick(columns, LOW_CANDIDATES),
        close=_pick(columns, CLOSE_CANDIDATES),
        volume=_pick(columns, VOLUME_CANDIDATES),
    )


def parquet_columns(path: str | Path) -> list[str] | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None

    parquet_file = pq.ParquetFile(path)
    return [str(name) for name in parquet_file.schema_arrow.names]


def parse_date_values(values: object) -> pd.Series:
    series = pd.Series(values)
    non_null = series.dropna()
    if non_null.empty:
        return pd.to_datetime(series, errors="coerce")

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    if pd.api.types.is_numeric_dtype(series):
        sample = non_null.astype("int64")
        if sample.between(19000101, 22001231).mean() >= 0.95:
            return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

    text_sample = non_null.astype(str).str.strip()
    digit_share = text_sample.str.fullmatch(r"\d{8}").mean()
    if digit_share >= 0.95:
        return pd.to_datetime(series.astype(str).str.strip(), format="%Y%m%d", errors="coerce")

    return pd.to_datetime(series, errors="coerce")


def source_name(source: str | Path | BinaryIO) -> str:
    if isinstance(source, (str, Path)):
        return str(source)
    return str(getattr(source, "name", "uploaded_data"))


def load_tabular(source: str | Path | BinaryIO, columns: Sequence[str] | None = None) -> pd.DataFrame:
    name = source_name(source).lower()
    selected = list(columns) if columns else None

    if name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(source, columns=selected)
    if name.endswith(".csv"):
        return pd.read_csv(source, usecols=selected)
    if name.endswith(".tsv"):
        return pd.read_csv(source, sep="\t", usecols=selected)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(source, usecols=selected)

    raise ValueError("Unsupported file type. Use parquet, csv, tsv, xlsx, or xls.")


def available_tickers(df: pd.DataFrame, ticker_col: str | None, limit: int = 20000) -> list[str]:
    if not ticker_col or ticker_col not in df.columns:
        return []

    values = df[ticker_col].dropna().astype(str).unique().tolist()
    return sorted(values)[:limit]


def standardize_prices(
    df: pd.DataFrame,
    columns: ColumnMap,
    ticker: str | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if columns.close is None:
        raise ValueError("A close/price column is required for forecasting.")

    work = df.copy()
    if columns.ticker and ticker and columns.ticker in work.columns:
        work = work[work[columns.ticker].astype(str) == str(ticker)]

    if columns.date and columns.date in work.columns:
        dates = parse_date_values(work[columns.date])
    elif isinstance(work.index, pd.DatetimeIndex):
        dates = pd.Series(work.index, index=work.index)
    else:
        raise ValueError("A date column is required for time-series forecasting.")

    output = pd.DataFrame({"Date": dates})
    rename_map = {
        "Open": columns.open,
        "High": columns.high,
        "Low": columns.low,
        "Close": columns.close,
        "Volume": columns.volume,
    }
    for target, source in rename_map.items():
        if source and source in work.columns:
            output[target] = pd.to_numeric(work[source], errors="coerce")

    if "Close" not in output.columns:
        raise ValueError("Close column could not be converted to numeric values.")

    output = output.dropna(subset=["Date", "Close"]).sort_values("Date")
    if start is not None:
        output = output[output["Date"] >= pd.Timestamp(start)]
    if end is not None:
        output = output[output["Date"] <= pd.Timestamp(end)]

    if output.empty:
        return output

    aggregations = {}
    if "Open" in output.columns:
        aggregations["Open"] = "first"
    if "High" in output.columns:
        aggregations["High"] = "max"
    if "Low" in output.columns:
        aggregations["Low"] = "min"
    aggregations["Close"] = "last"
    if "Volume" in output.columns:
        aggregations["Volume"] = "sum"

    output = output.groupby("Date", as_index=False).agg(aggregations)
    return output.sort_values("Date").reset_index(drop=True)


def date_bounds(df: pd.DataFrame, date_col: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not date_col or date_col not in df.columns:
        return None, None
    dates = parse_date_values(df[date_col]).dropna()
    if dates.empty:
        return None, None
    return dates.min(), dates.max()
