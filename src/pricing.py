import numpy as np
from scipy.stats import norm


def crr_params(r, sigma, T, N):
    """Paramètres de l'arbre Cox-Ross-Rubinstein."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    growth = np.exp(r * dt)
    if not (d < growth < u):
        raise ValueError("Condition de non-arbitrage violée : d < e^(r dt) < u")
    p = (growth - d) / (u - d)
    return dt, u, d, p


def crr_list(S0, K, T, r, sigma, N, kind="put", american=False):
    """Arbre CRR avec des listes (version pédagogique)."""
    dt, u, d, p = crr_params(r, sigma, T, N)
    disc = np.exp(-r * dt)
    if kind == "call":
        payoff = lambda s: max(s - K, 0)
    else:
        payoff = lambda s: max(K - s, 0)

    # Valeurs aux noeuds terminaux
    ST = [S0 * u**j * d**(N - j) for j in range(N + 1)]
    V = [payoff(s) for s in ST]

    # Induction rétrograde de i = N-1 à 0
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * V[j + 1] + (1 - p) * V[j])
            if american:
                S_ij = S0 * u**j * d**(i - j)
                cont = max(cont, payoff(S_ij))
            V[j] = cont
    return V[0]