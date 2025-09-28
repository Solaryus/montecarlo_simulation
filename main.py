import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def download_stock_data(ticker, start_date, end_date):
    """Télécharge les données boursières via yfinance."""
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if stock_data.empty:
        raise ValueError("Aucune donnée récupérée. Vérifiez le ticker et les dates.")
    return stock_data


def estimate_parameters(stock_data, window=None):
    """
    Estime mu (rendement moyen) et sigma (volatilité) à partir des log-rendements.
    window : si None → toute la période, sinon prend les n derniers jours.
    """
    log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1)).dropna()
    if window:
        log_returns = log_returns.tail(window)
    return float(log_returns.mean()), float(log_returns.std())


def monte_carlo_simulation(
    ticker, start_date, end_date,
    num_simulations=10, time_horizon=252,
    window=None
):
    """Simule des trajectoires de prix avec Monte Carlo."""
    stock_data = download_stock_data(ticker, start_date, end_date)
    mu, sigma = estimate_parameters(stock_data, window)
    S0 = stock_data['Close'].iloc[-1]

    # Génération des chocs aléatoires
    z = np.random.standard_normal((time_horizon, num_simulations))
    drift = (mu - 0.5 * sigma**2)
    diffusion = sigma * z

    # Simulation des trajectoires
    log_returns = drift + diffusion
    log_returns[0, :] = 0  # Jour 0 = prix initial
    price_paths = S0 * np.exp(np.cumsum(log_returns, axis=0))

    # Mise en DataFrame pour manipulation facile
    future_dates = pd.date_range(start=stock_data.index[-1], periods=time_horizon, freq="B")
    simulations = pd.DataFrame(price_paths, index=future_dates, columns=[f"Sim_{i+1}" for i in range(num_simulations)])

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label=f"Historique {ticker}", color="blue")
    plt.plot(simulations, alpha=0.5)
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.title(f"Simulation Monte Carlo : {ticker}")
    plt.legend(["Historique", "Simulations"], loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.show()

    return simulations


# Exemple d’utilisation
monte_carlo_simulation("TSLA", "2024-01-01", "2025-01-01", num_simulations=20, window=60)
