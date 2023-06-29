import pandas as pd
from functools import cache
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf


def average_dicts(dict1, dict2):
    if set(dict1.keys()) != set(dict2.keys()):
        raise ValueError("Dictionaries must have the same keys")

    average_dict = {}
    for key in dict1:
        average_dict[key] = (dict1[key] + dict2[key]) / 2

    return average_dict


def get_data(stocks, start, end):
    stock_data = yf.download(stocks, start=start, end=end)['Close']
    returns = stock_data.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix


# stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
# stockList = ['ADBE', 'AAPL', 'CMG', 'ETSY', 'META', 'LULU', 'NFLX', 'PINS', 'PTON', 'SHOP']
stockList = ['4755', '7201', '7269', '8306', '8035', '7733', '6857', '5802', '4543', '7211']
number_of_assets = len(stockList)

# International Modes
# stocks = [stock + '.AX' for stock in stockList]
stocks = [stock + '.T' for stock in stockList]

# stocks = stockList
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)
mean_returns, covMatrix = get_data(stocks, startDate, endDate)


def mc():
    asset_weights = np.random.random(len(mean_returns))
    asset_weights /= np.sum(asset_weights)  # Normalize Random Weights
    weights_dict = dict(zip(stockList, [round(weight, 2) for weight in asset_weights]))
    print(weights_dict)

    mc_sims = 400  # number of simulations
    T = 100  # timeframe in days

    mean_returns_matrix = np.full(shape=(T, len(asset_weights)), fill_value=mean_returns)
    mean_returns_matrix = mean_returns_matrix.T  # transpose

    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

    initialPortfolio = 100  # 100 percent or x monies

    for monte_carlo_sim_count in range(0, mc_sims):
        Z = np.random.normal(size=(T, len(asset_weights)))  # Uncorrelated Random Values with Normal Dist
        L = np.linalg.cholesky(covMatrix)  # Cholesky decomposition to Lower Triangular Matrix
        daily_returns = mean_returns_matrix + np.inner(L, Z)  # Correlated daily returns for individual stocks
        """np.inner(asset_weights, dailyReturns.T): This calculates the inner dot product between the weights and the 
        transposed matrix of daily returns. The weights represent the allocation of investment in different assets, 
        and the daily returns matrix contains the returns of each asset for each day in the timeframe. 

        np.cumprod(...): This calculates the cumulative product of the inner dot product result. The cumulative 
        product is taken to simulate the compounding effect of returns over time. This multiplication of returns over 
        time is a common way to calculate the growth of a portfolio. 

        +1: This adds 1 to the result of the inner dot product and cumulative product. This is done because the inner 
        dot product represents the daily returns, which are usually expressed as a percentage change. Adding 1 
        accounts for the fact that a return of 0 corresponds to no change in value. 

        *initialPortfolio: This multiplies the result by the initial portfolio value. It scales the simulated 
        portfolio value to reflect the initial investment amount. 
        
        portfolio_sims is a 2-dimensional NumPy array that represents the simulated portfolio values over a given 
        timeframe and multiple simulations. It has dimensions (T, mc_sims), where T is the number of days in the 
        timeframe, and mc_sims is the number of Monte Carlo simulations. 
        
        Therefore, portfolio_sims[:, m] selects the entire m-th column of portfolio_sims. This means it extracts the 
        values of the simulated portfolio for a specific simulation m over the entire timeframe. 
        """
        portfolio_sims[:, monte_carlo_sim_count] = np.cumprod(
            np.inner(asset_weights, daily_returns.T) + 1) * initialPortfolio

    print(portfolio_sims)
    print()

    """
    plt.clf()
    plt.plot(portfolio_sims)
    plt.ylabel('Portfolio Value (%)')
    plt.xlabel('Days')
    plt.title('MC simulation of a stock portfolio')
    plt.figtext(0.5, 0.01, str(weights_dict), ha="center", fontsize=6,
                bbox={"facecolor": "orange", "alpha": 0.5, "pad": 1})
    """

    time_x = np.arange(0, T, 1)
    indv_sim_performance = portfolio_sims.T

    slope_average = None
    intercept_average = None
    for sim in indv_sim_performance:
        coefficients = np.polyfit(time_x, sim, 1)  # Linear regression with degree 1
        slope = coefficients[0]
        intercept = coefficients[1]
        if slope_average:
            slope_average = (slope_average + slope) / 2
        else:
            slope_average = slope

        if intercept_average:
            intercept_average = (intercept_average + intercept) / 2
        else:
            intercept_average = intercept

    """
    plt.plot(time_x, slope_average * time_x + intercept_average, color='black')
    plt.savefig("../res/NIKKEI225T10-1.png")
    """

    print(f'slope_average: {slope_average}')
    print(f'weights_dict: {weights_dict}')
    return slope_average, weights_dict


if __name__ == "__main__":
    max_slope_average = None
    max_weights_average = None

    for i in range(500):
        curr_slope_average, curr_weights_dict = mc()

        if curr_slope_average > 0.25:
            if max_slope_average:
                max_slope_average = (max_slope_average + curr_slope_average) / 2
            else:
                max_slope_average = curr_slope_average

            if max_weights_average:
                max_weights_average = average_dicts(max_weights_average, curr_weights_dict)
            else:
                max_weights_average = curr_weights_dict

    for k, v in max_weights_average.items():
        max_weights_average[k] = round(v, 4)

    print()
    print(f'max_slope_average: {max_slope_average}')
    print(f'max_weights_average: {max_weights_average}')
