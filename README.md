# Monte Carlo Stock Portfolio Weighting

Utilizes Monte Carlo Method to simulate future value of stocks within a portfolio, accounting for previous historical movements and covariance (100-day window). Linear regresses average direction of the movements into a generalized trend. 

Ensembles multiple simulations to then find the greatest returning distribution of weights for the given portfolio. 

Example of NIKKEI225 index top 10 performers simulated:



Best weighting example by {ticker, percentage of investment}: 

{'4755': 0.1149, '7201': 0.0892, '7269': 0.1346, '8306': 0.1294, '8035': 0.0437, '7733': 0.0739, '6857': 0.1381, '5802': 0.0679, '4543': 0.0693, '7211': 0.1374}

Code Available here