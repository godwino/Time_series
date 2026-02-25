import pandas as pd


def _resolve_column(df: pd.DataFrame, expected: str):
    # Support both plain columns and yfinance MultiIndex columns.
    if expected in df.columns:
        return expected

    if isinstance(df.columns, pd.MultiIndex):
        exact_level0 = [col for col in df.columns if str(col[0]).lower() == expected.lower()]
        if exact_level0:
            return exact_level0[0]

    fuzzy = [col for col in df.columns if expected.lower() in str(col).lower()]
    if fuzzy:
        return fuzzy[0]

    raise KeyError(f"Column '{expected}' not found. Available columns: {list(df.columns)}")


def prepare_time_series(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Basic time-series cleanup for modeling."""
    actual_date_col = _resolve_column(df, date_col)
    actual_target_col = _resolve_column(df, target_col)

    out = df[[actual_date_col, actual_target_col]].copy()
    out.columns = [date_col, target_col]
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col)
    out = out.set_index(date_col).asfreq("D")
    out[target_col] = out[target_col].ffill().bfill()
    return out
