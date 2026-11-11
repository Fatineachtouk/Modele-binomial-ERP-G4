"""Tests unitaires du modèle binomial CRR (src/pricing.py).

Les tests s'exécutent sur toutes les implémentations disponibles :
crr_list (version listes, S1) et crr_numpy (version vectorisée, S2).
"""
import numpy as np
import pytest
from scipy.stats import norm

pricing = pytest.importorskip("src.pricing")

IMPLEMENTATIONS = [name for name in ("crr_list", "crr_numpy") if hasattr(pricing, name)]
S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2


def bs_reference(S, K, T, r, sigma, kind):
    """Black-Scholes de référence, indépendant de src/ pour valider la convergence."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


@pytest.fixture(params=IMPLEMENTATIONS)
def crr(request):
    return getattr(pricing, request.param)


def test_module_exposes_crr():
    assert IMPLEMENTATIONS, "src.pricing doit définir crr_list ou crr_numpy"


# --- Paramètres de l'arbre ---------------------------------------------------

def test_non_arbitrage_violation_raises():
    with pytest.raises(ValueError):
        pricing.crr_params(0.5, 0.01, 1, 1)


@pytest.mark.parametrize("N, T_, sigma_", [(0, 1, 0.2), (2.5, 1, 0.2), (10, 0, 0.2), (10, 1, 0)])
def test_invalid_parameters_raise(N, T_, sigma_):
    with pytest.raises(ValueError):
        pricing.crr_params(r, sigma_, T_, N)


def test_invalid_kind_raises(crr):
    with pytest.raises(ValueError):
        crr(S0, K, T, r, sigma, 10, "Call")


@pytest.mark.parametrize("N", [1, 10, 100, 500])
def test_parameters_consistent(N):
    dt, u, d, p = pricing.crr_params(r, sigma, T, N)
    assert dt == pytest.approx(T / N)
    assert u * d == pytest.approx(1)
    assert 0 < p < 1
    assert d < np.exp(r * dt) < u


# --- Valorisation ------------------------------------------------------------

def test_one_step_by_hand(crr):
    dt, u, d, p = pricing.crr_params(r, sigma, T, 1)
    expected = np.exp(-r * dt) * (p * max(S0 * u - K, 0) + (1 - p) * max(S0 * d - K, 0))
    assert crr(S0, K, T, r, sigma, 1, "call") == pytest.approx(expected)


@pytest.mark.parametrize("s0", [80, 100, 120])
def test_put_call_parity_european(crr, s0):
    call = crr(s0, K, T, r, sigma, 200, "call")
    put = crr(s0, K, T, r, sigma, 200, "put")
    assert call - put == pytest.approx(s0 - K * np.exp(-r * T), abs=1e-8)


def test_prices_within_no_arbitrage_bounds(crr):
    call = crr(S0, K, T, r, sigma, 200, "call")
    put = crr(S0, K, T, r, sigma, 200, "put")
    assert max(S0 - K * np.exp(-r * T), 0) <= call <= S0
    assert max(K * np.exp(-r * T) - S0, 0) <= put <= K


@pytest.mark.parametrize("kind", ["call", "put"])
def test_convergence_to_black_scholes(crr, kind):
    bs = bs_reference(S0, K, T, r, sigma, kind)
    errors = [abs(crr(S0, K, T, r, sigma, N, kind) - bs) for N in (50, 400)]
    assert errors[1] < 0.01
    assert errors[1] < errors[0]


def test_call_price_increases_with_sigma(crr):
    prices = [crr(S0, K, T, r, s, 200, "call") for s in (0.1, 0.2, 0.3, 0.4)]
    assert prices == sorted(prices)


# --- Options américaines -----------------------------------------------------

@pytest.mark.parametrize("s0", [80, 100, 120])
def test_american_put_at_least_european(crr, s0):
    eu = crr(s0, K, T, r, sigma, 200, "put", False)
    am = crr(s0, K, T, r, sigma, 200, "put", True)
    assert am >= eu - 1e-10
    assert am >= K - s0 - 1e-10


def test_early_exercise_premium_deep_itm_put(crr):
    eu = crr(50, K, T, r, sigma, 200, "put", False)
    am = crr(50, K, T, r, sigma, 200, "put", True)
    assert am > eu + 1


def test_american_call_equals_european_without_dividend(crr):
    eu = crr(S0, K, T, r, sigma, 200, "call", False)
    am = crr(S0, K, T, r, sigma, 200, "call", True)
    assert am == pytest.approx(eu, abs=1e-10)


# --- Cohérence entre implémentations (S2) -------------------------------------

@pytest.mark.skipif(len(IMPLEMENTATIONS) < 2, reason="crr_numpy pas encore disponible")
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("american", [False, True])
def test_list_equals_numpy(kind, american):
    for s0 in (80, 100, 120):
        a = pricing.crr_list(s0, K, T, r, sigma, 60, kind, american)
        b = pricing.crr_numpy(s0, K, T, r, sigma, 60, kind, american)
        assert a == pytest.approx(b, abs=1e-8)
