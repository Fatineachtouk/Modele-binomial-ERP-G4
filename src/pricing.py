import numpy as np
from scipy.stats import norm


def crr_params(r, sigma, T, N):
    """Paramètres de l'arbre Cox-Ross-Rubinstein."""
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N doit être un entier >= 1")
    if T <= 0 or sigma <= 0:
        raise ValueError("T et sigma doivent être strictement positifs")
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
    if kind not in ("call", "put"):
        raise ValueError("kind doit valoir 'call' ou 'put'")
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

def crr_numpy(S0, K, T, r, sigma, N, kind="put", american=False):
    """Arbre CRR vectorise NumPy (S2) : O(N^2) en temps, O(N) en memoire."""
    if kind not in ("call", "put"):
        raise ValueError("kind doit etre 'call' ou 'put'")        
    import numpy as np
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    growth = np.exp(r * dt)
    if not (d < growth < u):
        raise ValueError("Condition de non-arbitrage violee : d < e^(r dt) < u")
    p = (growth - d) / (u - d)
    disc = np.exp(-r * dt)
    sign = 1 if kind == "call" else -1

    # Noeuds terminaux
    j = np.arange(N + 1)
    ST = S0 * u**j * d**(N - j)
    V = np.maximum(sign * (ST - K), 0)

    # Induction retrograde
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[1:i + 2] + (1 - p) * V[0:i + 1])
        if american:
            jj = np.arange(i + 1)
            S = S0 * u**jj * d**(i - jj)
            V = np.maximum(V, np.maximum(sign * (S - K), 0))
    return float(V[0])
