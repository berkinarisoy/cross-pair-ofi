import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

FEATURES_PATH = "data/processed/features.parquet"
RESULTS_PATH = "results/results.md"
TRAIN_FRACTION = 0.70
HAC_LAGS = 10
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_BLOCK_SECONDS = 300
GRID_SECONDS = 60
SEED = 20260910

CONTROLS = ["R_C_lag", "z_C", "d_k", "n_A", "n_B", "n_C", "vol_A", "vol_B", "vol_C", "sin_tod", "cos_tod"]
CORR_FACTORS = ["s_k", "g_k", "zt_A", "zt_B", "z_C"]


def split_hours(df):
    hours = np.sort(df["utc_hour"].unique())
    n_train = max(1, min(int(round(TRAIN_FRACTION * len(hours))), len(hours) - 1))
    return df["utc_hour"].isin(hours[:n_train])


def standardize(df, is_train):
    """Z-score z_A/z_B/z_C on train-only mean/std, then derive the relative factor s_k and common factor g_k."""
    df = df.copy()
    for leg in "ABC":
        mu, sd = df.loc[is_train, f"z_{leg}"].mean(), df.loc[is_train, f"z_{leg}"].std(ddof=0) or 1.0
        df[f"zt_{leg}"] = (df[f"z_{leg}"] - mu) / sd
    df["s_k"] = df["zt_A"] - df["zt_B"]
    df["g_k"] = (df["zt_A"] + df["zt_B"]) / 2
    df["z_C"] = df["zt_C"]
    return df


def design(df, extended):
    """Build the OLS design matrix for the baseline or extended specification."""
    cols = CONTROLS + (["g_k", "s_k"] if extended else [])
    X = sm.add_constant(df[cols].to_numpy(), has_constant="add")
    return X, df["R_C_next"].to_numpy(), ["const"] + cols


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def block_bootstrap(train, iters):
    """Return the stationary block bootstrap distribution of beta_cross."""
    X, y, names = design(train, extended=True)
    j = names.index("s_k")
    n = len(y)
    p = GRID_SECONDS / BOOTSTRAP_BLOCK_SECONDS
    rng = np.random.default_rng(SEED)
    betas = np.empty(iters)
    for i in range(iters):
        idx, filled = np.empty(n, dtype=np.int64), 0
        while filled < n:
            start, length = rng.integers(0, n), rng.geometric(p)
            for s in range(min(length, n - filled)):
                idx[filled] = (start + s) % n
                filled += 1
        betas[i] = ols(X[idx], y[idx])[j]
    return betas


def correlations(df):
    """Correlate each factor with the contemporaneous and predictive returns."""
    return {c: {"contemp": df[c].corr(df["R_C_lag"]), "predictive": df[c].corr(df["R_C_next"])}
            for c in CORR_FACTORS}


def evaluate():
    """Fit baseline vs extended on train, test beta_cross, and score both models on the holdout."""
    df = pd.read_parquet(FEATURES_PATH)
    df = df[df["usable"]].sort_values("_k").reset_index(drop=True)
    is_train = split_hours(df)
    df = standardize(df, is_train)
    corr = correlations(df)
    train, holdout = df[is_train].reset_index(drop=True), df[~is_train].reset_index(drop=True)

    X, y, names = design(train, extended=True)
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    j = names.index("s_k")
    beta_cross, se = float(hac.params[j]), float(hac.bse[j])

    ci = np.percentile(block_bootstrap(train, BOOTSTRAP_ITERS), [2.5, 97.5])

    scores = {}
    for tag, extended in (("baseline", False), ("extended", True)):
        Xtr, ytr, _ = design(train, extended)
        Xho, yho, _ = design(holdout, extended)
        pred = Xho @ ols(Xtr, ytr)
        sse = np.sum((yho - pred) ** 2)
        sst = np.sum((yho - yho.mean()) ** 2)
        scores[tag] = {"mse": sse / len(yho), "r2": 1 - sse / sst}

    supported = beta_cross > 0 and ci[0] > 0 and scores["extended"]["mse"] < scores["baseline"]["mse"] \
        and scores["extended"]["r2"] > scores["baseline"]["r2"]

    return {
        "n_train": len(train), "n_holdout": len(holdout),
        "beta_cross": beta_cross, "se": se, "ci": ci, "scores": scores, "supported": supported,
        "corr": corr,
    }


def write_results(r):
    """Render the headline numbers from evaluate() to RESULTS_PATH."""
    os.makedirs("results", exist_ok=True)
    corr_rows = "\n".join(
        f"| {c} | {v['contemp']:+.4f} | {v['predictive']:+.4f} |" for c, v in r["corr"].items())
    with open(RESULTS_PATH, "w") as f:
        f.write(f"""# Model results

windows: train {r['n_train']:,} / holdout {r['n_holdout']:,}

beta_cross = {r['beta_cross']:.4g} (HAC se {r['se']:.4g})
bootstrap 95% CI = [{r['ci'][0]:.4g}, {r['ci'][1]:.4g}]

| model | holdout MSE | OOS R2 |
|---|---|---|
| baseline | {r['scores']['baseline']['mse']:.4g} | {r['scores']['baseline']['r2']:.4g} |
| extended | {r['scores']['extended']['mse']:.4g} | {r['scores']['extended']['r2']:.4g} |

hypothesis_supported = {r['supported']}

| factor | contemporaneous r | predictive r |
|---|---|---|
{corr_rows}
""")


if __name__ == "__main__":
    result = evaluate()
    write_results(result)
    print(f"beta_cross={result['beta_cross']:.4g} ci={result['ci']} supported={result['supported']}")
    print(f"-> {RESULTS_PATH}")
