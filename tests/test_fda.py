import numpy as np
import pytest

from fda import fraud, quality, sales
from fda.__main__ import main


def test_three_solvers_agree():
    s = sales.generate()
    X, _ = sales.design(s)
    sol = sales.solve(X, s["sales_k"].to_numpy())
    assert np.allclose(sol["normal_equations"], sol["lstsq"], atol=1e-8)
    assert np.allclose(sol["qr"], sol["lstsq"], atol=1e-8)


def test_regression_recovers_marketing_effect():
    s = sales.generate()
    X, names = sales.design(s)
    beta = sales.solve(X, s["sales_k"].to_numpy())["lstsq"]
    assert beta[names.index("marketing")] == pytest.approx(2.4, abs=0.8)  # planted value 2.4


def test_one_seasonal_curve_dominates():
    assert sales.seasonal_svd(sales.generate())["explained_share"][0] > 0.8


def test_backtest_error_is_small():
    assert sales.backtest(sales.generate())["mape_pct"] < 10


def test_joint_regression_finds_planted_effects_and_ignores_line():
    q = quality.bands(quality.generate())
    t = quality.ols_table(q).set_index("term")
    assert t.loc["night_shift", "estimate"] == pytest.approx(-2.0, abs=0.6)
    assert bool(t.loc["night_shift", "significant_at_5pct"])
    assert not bool(t.loc["line_L2", "significant_at_5pct"]) and not bool(t.loc["line_L3", "significant_at_5pct"])


def test_best_temperature_near_planted_optimum():
    assert quality.best_temperature(quality.bands(quality.generate()))["best_temperature_c"] == pytest.approx(21.5, abs=0.7)


def test_bonferroni_is_stricter():
    a = quality.anova_table(quality.bands(quality.generate()))
    assert (a["significant_after_bonferroni"] <= a["significant_at_5pct"]).all()


def test_mahalanobis_matches_definition():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    d = fraud.mahalanobis(X)
    mu, inv = X.mean(0), np.linalg.inv(np.cov(X, rowvar=False))
    assert d[5] == pytest.approx(np.sqrt((X[5] - mu) @ inv @ (X[5] - mu)))


def test_detectors_beat_random():
    f = fraud.generate(cards=80)
    ev = fraud.evaluate(f, fraud.scores(f))
    assert (ev["average_precision"] > 5 * f["is_fraud"].mean()).all()


def test_card_numbers_are_not_in_the_data():
    assert "card_number" not in [c.lower() for c in fraud.generate(cards=5).columns]


def test_cli_writes_report(tmp_path):
    main(["--data", str(tmp_path / "d"), "--out", str(tmp_path / "r")])
    assert "Task 3" in (tmp_path / "r" / "results.md").read_text()
