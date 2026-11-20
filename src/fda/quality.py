"""Task 2: which production factors affect product quality.

* One-way ANOVA per factor (temperature band, humidity band, shift, line),
  with effect size (eta squared) so a significant but tiny effect is not
  mistaken for an important one.
* Welch t-test for the two-group comparison (day vs night shift).
* Tukey HSD to show which bands differ.
* Linear regression with temperature as a curve, to find the best setting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def generate(seed: int = 2, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    temp = rng.normal(21, 3, n)
    humidity = rng.uniform(30, 70, n)
    shift = rng.choice(["day", "night"], n, p=[0.6, 0.4])
    line = rng.choice(["L1", "L2", "L3"], n)
    time = rng.normal(60, 5, n)
    quality = (
        85
        - 0.9 * (temp - 21.5) ** 2          # best around 21-22 degrees
        - 0.12 * np.clip(humidity - 50, 0, None) * 2
        - 2.0 * (shift == "night")
        + 0.0 * (line == "L2")              # no real line effect
        + 0.05 * (time - 60)
        + rng.normal(0, 3, n)
    )
    return pd.DataFrame(
        {
            "batch_id": np.arange(1, n + 1),
            "temperature_c": temp.round(1),
            "humidity_pct": humidity.round(1),
            "shift": shift,
            "line": line,
            "production_time_min": time.round(1),
            "quality_score": quality.round(2),
        }
    )


def bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temperature_band"] = pd.cut(out["temperature_c"], [-np.inf, 18, 20, 23, 25, np.inf], labels=["<18", "18-20", "20-23", "23-25", ">25"])
    out["humidity_band"] = pd.cut(out["humidity_pct"], [0, 40, 50, 60, 100], labels=["30-40", "40-50", "50-60", "60-70"])
    return out


def anova_table(df: pd.DataFrame, factors=("temperature_band", "humidity_band", "shift", "line")) -> pd.DataFrame:
    rows = []
    y = df["quality_score"]
    total_ss = ((y - y.mean()) ** 2).sum()
    for f in factors:
        groups = [g["quality_score"].to_numpy() for _, g in df.groupby(f, observed=True)]
        F, p = stats.f_oneway(*groups)
        between = sum(len(g) * (g.mean() - y.mean()) ** 2 for g in groups)
        rows.append({"factor": f, "groups": len(groups), "F": round(float(F), 2), "p_value": float(p), "eta_squared": round(float(between / total_ss), 3)})
    t = pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)
    t["significant_at_5pct"] = t["p_value"] < 0.05
    t["significant_after_bonferroni"] = t["p_value"] < 0.05 / len(factors)  # four tests, so 0.0125 each
    t["p_value"] = t["p_value"].map(lambda v: f"{v:.2e}")
    return t


def shift_ttest(df: pd.DataFrame) -> dict:
    d = df.loc[df["shift"] == "day", "quality_score"]
    n = df.loc[df["shift"] == "night", "quality_score"]
    t, p = stats.ttest_ind(d, n, equal_var=False)
    pooled = np.sqrt((d.var() + n.var()) / 2)
    return {"day_mean": round(d.mean(), 2), "night_mean": round(n.mean(), 2), "t": round(float(t), 2), "p_value": f"{p:.2e}", "cohens_d": round(float((d.mean() - n.mean()) / pooled), 2)}


def tukey_temperature(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby("temperature_band", observed=True)["quality_score"]
    names = list(groups.groups.keys())
    res = stats.tukey_hsd(*[g.to_numpy() for _, g in groups])
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rows.append({"band_a": str(names[i]), "band_b": str(names[j]), "mean_diff": round(float(res.statistic[i, j]), 2), "p_value": round(float(res.pvalue[i, j]), 4)})
    return pd.DataFrame(rows)


def best_temperature(df: pd.DataFrame) -> dict:
    t = df["temperature_c"].to_numpy()
    X = np.column_stack([np.ones_like(t), t, t**2, df["humidity_pct"], (df["shift"] == "night").astype(float)])
    beta = np.linalg.lstsq(X, df["quality_score"].to_numpy(), rcond=None)[0]
    best = -beta[1] / (2 * beta[2])
    return {"best_temperature_c": round(float(best), 1), "night_shift_effect": round(float(beta[4]), 2), "humidity_effect_per_pct": round(float(beta[3]), 3)}


def ols_table(df: pd.DataFrame) -> pd.DataFrame:
    """All factors in one regression, with standard errors from (X'X)^-1.

    Looking at each factor alone can mislead: temperature varies so much that it hides the
    shift effect, and four separate tests at 5% will often flag one factor by chance.
    """
    t = df["temperature_c"].to_numpy()
    tc = t - t.mean()  # centred, so the linear and squared terms are not strongly correlated
    cols = {
        "intercept": np.ones(len(df)),
        "temperature": tc,
        "temperature_squared": tc**2,
        "humidity_above_50": np.clip(df["humidity_pct"] - 50, 0, None),
        "night_shift": (df["shift"] == "night").astype(float),
        "line_L2": (df["line"] == "L2").astype(float),
        "line_L3": (df["line"] == "L3").astype(float),
        "production_time": df["production_time_min"] - 60,
    }
    X = np.column_stack(list(cols.values()))
    y = df["quality_score"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    tval = beta / se
    p = 2 * stats.t.sf(np.abs(tval), dof)
    out = pd.DataFrame({"term": list(cols), "estimate": beta.round(3), "std_error": se.round(3), "t": tval.round(2), "p_value": p})
    out["significant_at_5pct"] = out["p_value"] < 0.05
    out["p_value"] = out["p_value"].map(lambda v: f"{v:.2e}")
    return out
