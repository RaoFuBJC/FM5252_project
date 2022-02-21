import numpy as np
from scipy.stats import binom
from scipy.stats import norm
from math import log, sqrt, exp

def American_call(N, S0, sigma, T, r, K):
    """
    N = number of binomial iterations
    S0 = initial stock price
    r = risk free interest rate per annum
    K = strike price
    """
    t = T/N
    u = np.exp(sigma*np.sqrt(t))
    d = 1 / u
    p = (1 + r - d) / (u - d)
    q = 1 - p

    k = np.arange(0,N+1)
    svals = S0*u**(2*k - N)
    payoffs = np.maximum(svals -K, 0)
    discount = np.exp(-r*t)
    def loop(N, discount, K, payoffs):
      if N > 1:
        N=N-1
        k = np.arange(0,N+1)
        svals = S0*u**(2*k - N)
        payoffs[:N+1] = discount*(p * payoffs[1:N+2] + q * payoffs[0:N+1])
        payoffs = payoffs[:-1]
        payoffs = np.maximum(payoff, svals-K)
        return loop(N, discount, K, payoffs)
      elif N == 1:
        k = np.arange(0,N+1)
        svals = S0*u**(2*k - N)
        payoffs[:N+1] = discount*(p * payoffs[1:N+2] + q * payoffs[0:N+1])
        payoffs = payoffs[:-1]
        payoffs = np.maximum(payoff, svals-K)
        return payoffs[0]

    ACall_result = loop(N, discount, K, payoffs)
    return ACall_result

def American_put(N, S0, sigma, T, r, K):
    """
    N = number of binomial iterations
    S0 = initial stock price
    r = risk free interest rate per annum
    K = strike price
    """
    t = T/N
    u = np.exp(sigma*np.sqrt(t))
    d = 1 / u
    p = (1 + r - d) / (u - d)
    q = 1 - p

    k = np.arange(0,N+1)
    svals = S0*u**(2*k - N)
    payoffs = np.maximum(svals -K, 0)
    discount = np.exp(-r*t)
    def loop(N, discount, K, payoffs):
      if N > 1:
        N=N-1
        k = np.arange(0,N+1)
        svals = S0*u**(2*k - N)
        payoffs[:N+1] = discount*(p * payoffs[1:N+2] + q * payoffs[0:N+1])
        payoffs = payoffs[:-1]
        payoffs = np.maximum(payoffs, K-svals)
        return loop(N, discount, K, payoffs)
      elif N == 1:
        k = np.arange(0,N+1)
        svals = S0*u**(2*k - N)
        payoffs[:N+1] = discount*(p * payoffs[1:N+2] + q * payoffs[0:N+1])
        payoffs = payoffs[:-1]
        payoffs = np.maximum(payoffs, K-svals)
        return payoffs[0]

    APut_result = loop(N, discount, K, payoffs)
    return APut_result


def European_call(N, S0, sigma, T, r, K):
    """
    N = number of binomial iterations
    S0 = initial stock price
    r = risk free interest rate per annum
    K = strike price
    """
    t = T/N
    u = np.exp(sigma*np.sqrt(t))
    d = 1 / u
    p = (1 + r - d) / (u - d)

    k = np.arange(0,N+1)
    svals = S0*u**(2*k - N)
    payoffs = np.maximum(svals -K, 0)
    probs = binom.pmf(k,n=N, p=p)
    callvalue = (probs @ payoffs) * np.exp(-N*r*t)
    return callvalue



def European_put(N, S0, sigma, T, r, K):
    """
    N = number of binomial iterations
    S0 = initial stock price
    r = risk free interest rate per annum
    K = strike price
    """
    t = T/N
    u = np.exp(sigma*np.sqrt(t))
    d = 1 / u
    p = (1 + r - d) / (u - d)

    k = np.arange(0,N+1)
    svals = S0*u**(2*k - N)
    payoffs = np.maximum(K - svals, 0)
    probs = binom.pmf(k,n=N, p=p)
    putvalue = (probs @ payoffs) * np.exp(-N*r*t)
    return putvalue



def call_greeks(S,K,T,r,sigma):
    d1 = lambda S, K, T, r, sigma: (log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * sqrt(T))
    d2 = lambda S, K, T, r, sigma: d1(S, K, T, r, sigma) - sigma * sqrt(T)
    call_delta = lambda S, K, T, r, sigma: norm.cdf(d1(S, K, T, r, sigma))
    gamma = lambda S, K, T, r, sigma: norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * sqrt(T))
    vega = lambda S, K, T, r, sigma: (S * norm.pdf(d1(S, K, T, r, sigma)) * sqrt(T))
    call_theta = lambda S, K, T, r, sigma: (-(S * norm.pdf(d1(S, K, T, r, sigma)) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma)))
    call_rho = lambda S, K, T, r, sigma: (K * T * exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma)))
    return call_delta(S, K, T, r, sigma),gamma(S, K, T, r, sigma),vega(S, K, T, r, sigma),call_theta(S, K, T, r, sigma),call_rho(S, K, T, r, sigma)

def put_greeks(S, K, T, r, sigma):
    d1 = lambda S, K, T, r, sigma: (log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * sqrt(T))
    d2 = lambda S, K, T, r, sigma: d1(S, K, T, r, sigma) - sigma * sqrt(T)
    put_delta = lambda S, K, T, r, sigma: -norm.cdf(-d1(S, K, T, r, sigma))
    gamma = lambda S, K, T, r, sigma: norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * sqrt(T))
    vega = lambda S, K, T, r, sigma: (S * norm.pdf(d1(S, K, T, r, sigma)) * sqrt(T))
    put_theta = lambda S, K, T, r, sigma: (-(S * norm.pdf(d1(S, K, T, r, sigma)) * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma)))
    put_rho = lambda S, K, T, r, sigma: (-K * T * exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma)))
    return put_delta(S, K, T, r, sigma), gamma(S, K, T, r, sigma), vega(S, K, T, r, sigma), put_theta(S, K, T, r,sigma), put_rho(S, K, T, r, sigma)



