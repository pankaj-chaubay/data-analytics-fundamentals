"""Task 3: flag suspicious card transactions.

Each card's transactions become feature vectors (amount, hour, distance from
the card's usual amount, time since the last transaction, category). Three
detectors are compared against planted fraud:

* robust z-score of the amount per card (median and MAD),
* Mahalanobis distance on features measured against each card's own history (uses the covariance matrix),
* Isolation Forest.

Card numbers are never used as features; they are replaced by a card key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_score, recall_score

CATEGORIES = ["grocery", "fuel", "restaurant", "online", "travel", "electronics"]


def generate(seed: int = 3, cards: int = 300, days: int = 90, fraud_rate: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2024-01-01")
    for c in range(cards):
        typical = rng.lognormal(3.4, 0.5)
        habit = rng.dirichlet(np.ones(len(CATEGORIES)) * 2)
        n = rng.poisson(days * 0.8)
        times = np.sort(rng.uniform(0, days * 24, n))
        for h in times:
            ts = start + pd.Timedelta(hours=float(h))
            hour = ts.hour
            if hour < 7 and rng.random() < 0.8:  # most people do not shop at night
                ts += pd.Timedelta(hours=9)
            rows.append({"card_key": f"C{c:04d}", "timestamp": ts, "amount": round(float(typical * rng.lognormal(0, 0.45)), 2),
                         "category": CATEGORIES[rng.choice(len(CATEGORIES), p=habit)], "is_fraud": 0})
    df = pd.DataFrame(rows)
    # plant fraud: large amounts, odd hours, online/electronics, bursts
    k = int(len(df) * fraud_rate)
    idx = rng.choice(df.index, k, replace=False)
    df.loc[idx, "amount"] = (df.loc[idx, "amount"] * rng.uniform(4, 12, k)).round(2)
    df.loc[idx, "category"] = rng.choice(["online", "electronics", "travel"], k)
    df.loc[idx, "timestamp"] = df.loc[idx, "timestamp"].dt.normalize() + pd.to_timedelta(rng.integers(0, 5, k), unit="h")
    df.loc[idx, "is_fraud"] = 1
    df = df.sort_values(["card_key", "timestamp"]).reset_index(drop=True)
    df.insert(0, "transaction_id", np.arange(1, len(df) + 1))
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("card_key")["amount"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median()) * 1.4826
    out = pd.DataFrame(index=df.index)
    out["log_amount"] = np.log1p(df["amount"])
    out["amount_vs_card_median"] = np.log1p(df["amount"]) - np.log1p(med)
    out["robust_z"] = (df["amount"] - med) / mad.replace(0, np.nan)
    hour = df["timestamp"].dt.hour
    out["night"] = (hour < 6).astype(float)
    out["hours_since_last"] = df.groupby("card_key")["timestamp"].diff().dt.total_seconds().div(3600).fillna(48).clip(upper=48)
    share = df.groupby(["card_key", "category"])["amount"].transform("size") / df.groupby("card_key")["amount"].transform("size")
    out["category_rarity"] = 1 - share
    return out.fillna(0)


def mahalanobis(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    inv = np.linalg.pinv(cov)
    d = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))


def scores(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    f = features(df)
    cols = ["amount_vs_card_median", "night", "hours_since_last", "category_rarity"]
    X = f[cols].to_numpy()
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=seed).fit(X)
    return pd.DataFrame(
        {
            "robust_z": f["robust_z"].abs(),
            "mahalanobis": mahalanobis(X),
            "isolation_forest": -iso.score_samples(X),
        },
        index=df.index,
    )


def evaluate(df: pd.DataFrame, sc: pd.DataFrame, review_share: float = 0.02) -> pd.DataFrame:
    """Flag the top review_share of transactions per method, the size of a manual review queue."""
    rows = []
    y = df["is_fraud"].to_numpy()
    for m in sc.columns:
        s = sc[m].to_numpy()
        cut = np.quantile(s, 1 - review_share)
        flag = s >= cut
        rows.append({"method": m, "flagged": int(flag.sum()), "precision": round(precision_score(y, flag), 3),
                     "recall": round(recall_score(y, flag), 3), "average_precision": round(average_precision_score(y, s), 3)})
    return pd.DataFrame(rows).sort_values("average_precision", ascending=False).reset_index(drop=True)
