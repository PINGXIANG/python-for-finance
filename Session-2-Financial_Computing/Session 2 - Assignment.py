import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from scipy.stats import norm


# Computing the compounded money
def discrete_compounding(Pi: float, r: float, n: int, T: int):
    return np.multiply(Pi, np.power((1. + r / n), np.multiply(n, T)))


# as n goes to infinity
Pi, r, T = 1, 0.03, 10.
n_array = np.array([1., 2., 3., 6., 12., 126., 252., 2520.])

# Draw the graph one by one
for n in n_array:
    # The x-axis
    tt = np.linspace(start=0, stop=T, num=int(np.multiply(n, T) + 1))

    # The y-axis
    discrete_compounding_vec = np.vectorize(discrete_compounding)
    pp = discrete_compounding_vec(Pi, r, n, tt)
    data = pandas.DataFrame({"t": tt, "return": pp})
    # Draw the plot
    sns.lineplot(data=data, x="t", y='return', label='n= %d' % int(n), drawstyle="steps-post")

# The converge function
n = 2520.
tt = np.linspace(start=0, stop=T, num=int(n + 1))
plt.plot(tt, [np.multiply(Pi, np.exp(np.multiply(r, t))) for t in tt], label='exp(rt)')

plt.xlabel("time/day, T = 1 Yr")
plt.legend(loc="best")
plt.show()

S0, K, sigma, t, T = 100., 100., 0.2, 0., 1.


def BS_call(St, K, sigma, t, T, r=0):
    deltaT = T - t
    sigmaT = np.multiply(sigma, np.sqrt(deltaT))
    lnSK = np.log(St / K)
    d1 = (lnSK + (r + (np.power(sigma, 2.)) / 2.)) / sigmaT
    d2 = d1 - sigmaT

    return np.multiply(St, norm.cdf(d1)) - np.multiply(K, norm.cdf(d2)) * np.exp(-r * deltaT)


print("Option price in BS model:", BS_call(S0, K, sigma, t, T))


def BS_simulation(S0, K, T, sigma, N):
    W_T = np.random.normal(0., 1., N) * np.sqrt(T)
    ST = S0*np.exp((-0.5*np.power(sigma, 2)*T + sigma*W_T))
    returns = np.maximum(ST-K, 0.)
    return np.mean(returns)


nn = np.arange(500, 100000, 1000)
BS_simulation_vec = np.vectorize(BS_simulation)
bb = BS_simulation_vec(S0, K, T, sigma, nn)
plt.plot(nn, bb)
