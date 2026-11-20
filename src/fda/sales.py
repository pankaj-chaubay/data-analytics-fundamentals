"""Task 1: retail sales with linear algebra.

* Regression solved three ways: normal equations, QR, and least squares (SVD).
  All three must agree; the condition number shows when the normal equations
  become unreliable.
* SVD of the region x month sales matrix: how much of the pattern one shared
  seasonal curve explains.
* Forecast of the next six months from trend, seasonality and marketing spend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIONS = ["North", "South", "East", "West"]


def generate(seed: int = 1, months: int = 36) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.period_range("2022-01", periods=months, freq="M")
    rows = []
    level = {"North": 120, "South": 95, "East": 80, "West": 60}  # thousand EUR
    for r in REGIONS:
        for i, d in enumerate(dates):
            season = 1 + 0.18 * np.sin(2 * np.pi * (d.month - 3) / 12) + (0.25 if d.month == 12 else 0)
            marketing = rng.uniform(4, 12) * level[r] / 100
            sales = (level[r] * (1 + 0.012 * i) * season + 2.4 * marketing) * rng.normal(1, 0.04)
            units = int(sales * 1000 / rng.normal(38, 1.5))
            rows.append({"month": str(d), "region": r, "sales_k": round(sales, 2), "units": units, "marketing_k": round(marketing, 2)})
    return pd.DataFrame(rows)


def design(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    m = pd.PeriodIndex(df["month"], freq="M")
    t = np.array([(p - m.min()).n for p in m], float)
    cols = [np.ones(len(df)), t, df["marketing_k"].to_numpy(), np.sin(2 * np.pi * m.month / 12), np.cos(2 * np.pi * m.month / 12), (m.month == 12).astype(float)]
    names = ["intercept", "trend_per_month", "marketing", "season_sin", "season_cos", "december"]
    for r in REGIONS[1:]:
        cols.append((df["region"] == r).astype(float).to_numpy())
        names.append(f"region_{r}")
    return np.column_stack(cols), names


def solve(X: np.ndarray, y: np.ndarray) -> dict:
    normal = np.linalg.solve(X.T @ X, X.T @ y)
    q, r = np.linalg.qr(X)
    qr = np.linalg.solve(r, q.T @ y)
    lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
    return {"normal_equations": normal, "qr": qr, "lstsq": lstsq, "condition_number": float(np.linalg.cond(X))}


def r_squared(X, y, beta) -> float:
    resid = y - X @ beta
    return float(1 - resid @ resid / ((y - y.mean()) @ (y - y.mean())))


def seasonal_svd(df: pd.DataFrame) -> dict:
    M = df.pivot(index="region", columns="month", values="sales_k").to_numpy()
    centred = M - M.mean(axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    share = s**2 / (s**2).sum()
    return {"explained_share": share.round(3).tolist(), "first_component": vt[0], "months": list(df["month"].unique())}


def forecast(df: pd.DataFrame, beta: np.ndarray, months: int = 6, marketing_change: float = 0.0) -> pd.DataFrame:
    last = pd.Period(df["month"].max(), "M")
    future = []
    avg_mkt = df.groupby("region")["marketing_k"].mean()
    for r in REGIONS:
        for h in range(1, months + 1):
            future.append({"month": str(last + h), "region": r, "marketing_k": avg_mkt[r] * (1 + marketing_change)})
    f = pd.DataFrame(future)
    full = pd.concat([df[["month", "region", "marketing_k"]], f], ignore_index=True)
    X, _ = design(full)
    f["forecast_sales_k"] = (X[len(df):] @ beta).round(1)
    return f


def backtest(df: pd.DataFrame, holdout: int = 6) -> dict:
    months = sorted(df["month"].unique())
    train, test = df[df["month"].isin(months[:-holdout])], df[df["month"].isin(months[-holdout:])]
    both = pd.concat([train, test], ignore_index=True)  # one design matrix, so the trend counts from the same start
    X, _ = design(both)
    beta = np.linalg.lstsq(X[: len(train)], train["sales_k"].to_numpy(), rcond=None)[0]
    pred = X[len(train) :] @ beta
    err = np.abs(pred - test["sales_k"].to_numpy())
    return {"mape_pct": round(float(100 * (err / test["sales_k"].to_numpy()).mean()), 2), "holdout_months": holdout}
