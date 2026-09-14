import gzip
import json
import os
from glob import glob

import numpy as np
import pandas as pd

PAIRS = {"A": "ETH-USD", "B": "BTC-USD", "C": "ETH-BTC"}
VENUE_ID = 1  # Coinbase
GRID_SECONDS = 60
MAX_STALE_SECONDS = 300
RAW_DIR = "data/raw"
OUT_PATH = "data/processed/features.parquet"


def load_pair(pair):
    """Read trades_hist_<pair>.jsonl.gz and keep only the target venue and valid prints."""
    rows = []
    for path in glob(f"{RAW_DIR}/trades_hist_{pair}.jsonl.gz"):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                o = json.loads(line)
                rows.append((o["price"], o["size"], o["participant_timestamp"] // 1_000_000, o["exchange"]))
    df = pd.DataFrame(rows, columns=["price", "size", "t", "x"])
    df = df[(df["x"] == VENUE_ID) & (df["price"] > 0) & (df["size"] > 0)]
    return df.sort_values("t").reset_index(drop=True)


def tick_rule_signs(price):
    """Lee-Ready tick-rule sign per trade: +1 up, -1 down, carries forward when unchanged."""
    diff = np.sign(np.diff(price)).astype(float)
    diff[diff == 0] = np.nan
    signs = pd.Series(diff).ffill().fillna(0.0).to_numpy()
    return np.concatenate([[0.0], signs])


def window_features(trades, grid, window_ms):
    """Compute the as-of log price and signed trade-flow imbalance for each grid step."""
    sign = tick_rule_signs(trades["price"].to_numpy())
    size = trades["size"].to_numpy()
    t = trades["t"].to_numpy()

    k = np.searchsorted(grid, t, side="left")
    keep = (k > 0) & (k < len(grid)) & (t > grid[np.clip(k, 0, len(grid) - 1)] - window_ms)
    win = pd.DataFrame({"_k": k[keep], "signed": (sign * size)[keep], "size": size[keep]})
    agg = win.groupby("_k").agg(net=("signed", "sum"), vol=("size", "sum"), n=("signed", "size")).reset_index()
    agg["z"] = agg["net"] / agg["vol"]

    grid_df = pd.DataFrame({"tau": grid, "_k": np.arange(len(grid))})
    asof = pd.merge_asof(grid_df, trades.rename(columns={"t": "trade_t"})[["trade_t", "price"]],
                         left_on="tau", right_on="trade_t", direction="backward")
    asof["logmid"] = np.log(asof["price"])
    asof.loc[(asof["tau"] - asof["trade_t"]) > MAX_STALE_SECONDS * 1000, "logmid"] = np.nan

    return asof[["_k", "tau", "logmid"]].merge(agg[["_k", "z", "n", "vol"]], on="_k", how="left")


def build_features():
    """Join all three legs onto one grid and derive the regression variables."""
    step_ms = GRID_SECONDS * 1000
    legs = {leg: load_pair(pair) for leg, pair in PAIRS.items()}
    t0 = min(df["t"].min() for df in legs.values())
    t1 = max(df["t"].max() for df in legs.values())
    grid = np.arange((t0 // step_ms + 1) * step_ms, t1 + 1, step_ms)

    wide = None
    for leg, trades in legs.items():
        f = window_features(trades, grid, step_ms).rename(
            columns={"logmid": f"logmid_{leg}", "z": f"z_{leg}", "n": f"n_{leg}", "vol": f"vol_{leg}"})
        wide = f if wide is None else wide.merge(f.drop(columns="tau"), on="_k", how="left")
    wide = wide.sort_values("_k").reset_index(drop=True)

    wide["d_k"] = wide["logmid_A"] - wide["logmid_B"] - wide["logmid_C"]
    wide["R_C_next"] = wide["logmid_C"].shift(-1) - wide["logmid_C"]
    wide["R_C_lag"] = wide["logmid_C"] - wide["logmid_C"].shift(1)

    k = wide["_k"].to_numpy()
    consecutive = np.r_[False, np.diff(k) == 1]
    wide["_lag_ok"] = consecutive
    wide["_next_ok"] = np.r_[consecutive[1:], False]

    hour_of_day = (wide["tau"] / 1000 % 86400) / 3600
    wide["sin_tod"] = np.sin(2 * np.pi * hour_of_day / 24)
    wide["cos_tod"] = np.cos(2 * np.pi * hour_of_day / 24)
    wide["utc_hour"] = (wide["tau"] // 1000 // 3600).astype("int64")

    has_data = (wide[["n_A", "n_B", "n_C"]].fillna(0) > 0).all(axis=1)
    wide["usable"] = (has_data & wide["_lag_ok"] & wide["_next_ok"]
                      & wide[["d_k", "R_C_next", "R_C_lag", "z_A", "z_B", "z_C"]].notna().all(axis=1))

    os.makedirs("data/processed", exist_ok=True)
    wide.to_parquet(OUT_PATH)
    return wide


if __name__ == "__main__":
    df = build_features()
    print(f"{len(df):,} windows, {int(df['usable'].sum()):,} usable -> {OUT_PATH}")
