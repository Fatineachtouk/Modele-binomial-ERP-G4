"""Tests S2 (Yassine) : coherence financiere de l'arbre CRR (crr_numpy)."""
import math

import pytest

from src.pricing import crr_numpy

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


def _cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(s0, k, t, r, sigma, kind):
    """Black-Scholes de reference (ecrit ici pour que le test soit autonome)."""
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if kind == "call":
        return s0 * _cdf(d1) - k * math.exp(-r * t) * _cdf(d2)
    return k * math.exp(-r * t) * _cdf(-d2) - s0 * _cdf(-d1)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_put_call_parity(strike):
    c = crr_numpy(S0, strike, T, R, SIGMA, 500, "call", False)
    p = crr_numpy(S0, strike, T, R, SIGMA, 500, "put", False)
    assert c - p == pytest.approx(S0 - strike * math.exp(-R * T), abs=1e-6)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_american_call_equals_european_call_without_dividend(strike):
    eu = crr_numpy(S0, strike, T, R, SIGMA, 300, "call", False)
    am = crr_numpy(S0, strike, T, R, SIGMA, 300, "call", True)
    assert am == pytest.approx(eu, abs=1e-8)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_american_put_at_least_european_put(strike):
    eu = crr_numpy(S0, strike, T, R, SIGMA, 300, "put", False)
    am = crr_numpy(S0, strike, T, R, SIGMA, 300, "put", True)
    assert am >= eu - 1e-12


def test_american_put_strictly_above_european_when_in_the_money():
    eu = crr_numpy(S0, 120.0, T, R, SIGMA, 300, "put", False)
    am = crr_numpy(S0, 120.0, T, R, SIGMA, 300, "put", True)
    assert am > eu


def test_american_put_never_below_intrinsic_value():
    am = crr_numpy(80.0, K, T, R, SIGMA, 300, "put", True)
    assert am >= K - 80.0 - 1e-12


def test_deep_in_the_money_american_put_is_exercised_immediately():
    am = crr_numpy(40.0, K, T, R, SIGMA, 200, "put", True)
    assert am == pytest.approx(K - 40.0, abs=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_convergence_to_black_scholes(kind):
    ref = bs_price(S0, K, T, R, SIGMA, kind)
    assert crr_numpy(S0, K, T, R, SIGMA, 1000, kind, False) == pytest.approx(ref, abs=0.01)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_error_is_smaller_with_more_steps(kind):
    ref = bs_price(S0, K, T, R, SIGMA, kind)
    err_small = abs(crr_numpy(S0, K, T, R, SIGMA, 20, kind, False) - ref)
    err_large = abs(crr_numpy(S0, K, T, R, SIGMA, 500, kind, False) - ref)
    assert err_large < err_small


def test_prices_stay_within_no_arbitrage_bounds():
    call = crr_numpy(S0, K, T, R, SIGMA, 300, "call", False)
    put = crr_numpy(S0, K, T, R, SIGMA, 300, "put", False)
    assert max(S0 - K * math.exp(-R * T), 0.0) - 1e-9 <= call <= S0
    assert 0.0 <= put <= K


def test_call_price_increases_with_volatility():
    low = crr_numpy(S0, K, T, R, 0.10, 300, "call", False)
    mid = crr_numpy(S0, K, T, R, 0.20, 300, "call", False)
    high = crr_numpy(S0, K, T, R, 0.40, 300, "call", False)
    assert low < mid < high


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        crr_numpy(S0, K, T, R, SIGMA, 50, "straddle", False)


def test_arbitrage_violation_raises():
    with pytest.raises(ValueError):
        crr_numpy(S0, K, 1.0, 1.0, 0.01, 1, "call", False)
