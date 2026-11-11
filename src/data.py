import numpy as np
import yfinance as yf


def _close(ticker, period):
    px = yf.download(ticker, period=period, auto_adjust=True, progress=False)["Close"]
    return px.squeeze()


def historical_vol(ticker, period="1y"):
    """Volatilité historique annualisée (rendements logarithmiques, 252 jours)."""
    px = _close(ticker, period)
    ret = np.log(px / px.shift(1)).dropna()
    return float(ret.std() * np.sqrt(252))


def last_price(ticker):
    """Dernier cours de clôture."""
    return float(_close(ticker, "5d").iloc[-1])