#Part 1(bisection method)
from math import log, sqrt, exp, pi
from scipy.stats import norm
import scipy
import yfinance as yf
def Callprice(S, K, T, r, sigma):
    d1 = (log(S/K)+(r +sigma**2/2.)*T)/sigma*sqrt(T)
    d2 = d1 - sigma * sqrt(T)
    cprice = S*norm.cdf(d1)-K*exp(-r*T)*norm.cdf(d2)
    return cprice

def Putprice(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * (T)) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    pprice = K*exp(-r*T)-S+Callprice(S,K,T,r,sigma)
    return pprice

def imp_vol(type, market_price, S, K, r, T):
    precision = 0.00001
    upper_vol = 50.0
    lower_vol = 0.0001
    iteration = 0

    while 1:
        iteration += 1
        mid_vol = (upper_vol + lower_vol) / 2.0

        if type == 'c':
            price = Callprice(S, K, T, r, mid_vol)
            lower_price = Callprice(S, K, T, r, lower_vol)
            if (lower_price - market_price) * (price - market_price) > 0:
                lower_vol = mid_vol
            else:
                upper_vol = mid_vol
            if abs(price - market_price) < precision: break

        elif type == 'p':
            price = Putprice(S, K, T, r, mid_vol)
            upper_price = Putprice(S, K, T, r, upper_vol)

            if (upper_price - market_price) * (price - market_price) > 0:
                upper_vol = mid_vol
            else:
                lower_vol = mid_vol

            if abs(price - market_price) < precision: break

            if iteration > 100: break

    return mid_vol

print(imp_vol('p', 5.57, 100, 100, 0.05, 1))

#Part 1(Newton's method)

def fun_vega(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * sqrt(T))
    vega = (S * norm.pdf(d1) * sqrt(T))
    return vega

def newton_vol_call(C, S, K, T, r, sigma, tol = 0.00001):
    max_iter = 200 #max number 0f iteration

    for k in range(max_iter):
        diff = Callprice(S, K, T, r, sigma) - C
        vega = fun_vega(S, K, T, r, sigma)
        vol_old = sigma
        vol_new = vol_old - diff/vega

        if abs(vol_old - vol_new) < tol:
            break
    implied_vol = vol_new
    return implied_vol
print(newton_vol_call(3.7,30,28,0.5,0.025,0.3))

def newton_vol_put(P, S, K, T, r, sigma, tol = 0.00001):
    max_iter = 200 #max number 0f iteration
    vega = fun_vega(S, K, T, r, sigma)

    for k in range(max_iter):
        diff = Putprice(S, K, T, r, sigma) - P
        vol_old = sigma
        vol_new = vol_old - diff/vega

        if abs(vol_old - vol_new) < tol:
            break
    implied_vol = vol_new
    return implied_vol
print(newton_vol_put(7,25,20,1,0.05,0.25))



def get_vols_skew(ticker, date):
    underly = yf.Ticker(ticker)
    opt = underly.option_chain(date)
    call_data = opt.calls
    put_data = opt.puts
    effective_vol_call = []
    strike_call = []
    effective_vol_put = []
    strike_put = []
    for i in call_data.iterrows():
        if i[1][11] == True and i[1][10] > 0.00005 and i[1][10] < 1.5:

            effective_vol_call.append(i[1][10])
            strike_call.append(i[1][2])

    for i in put_data.iterrows():
        if i[1][11] == True and i[1][10] > 0.00005 and i[1][10] < 1.5:

            effective_vol_put.append(i[1][10])
            strike_put.append(i[1][2])
    current_price = (min(strike_put) + max(strike_call))/2
    effective_vol = effective_vol_put + effective_vol_call
    strike = strike_put + strike_call
    return effective_vol, strike, current_price

vol = get_vols_skew("AAPL","2022-06-17")

strike = vol[1]
impliedvol = vol[0]
underlyprice = vol[2]

def sviCurve (x, a, b, rho, m,sigma):

    #if b > 0 and np.abs(rho) < 0 and sigma>0:
    result = a+b*(rho*(x-m)+sqrt((x-m)**2 + sigma**2))
    std = sqrt(result)
    return std

def sviFit (impliedvol,strikes):
    o = scipy.optimize.curve_fit(sviCurve, strikes, impliedvol, maxfev= 1000000)
    return o
print("section 2")

def runfunc_2(ticker,date):
    vol = get_vols_skew(ticker, date)

    strike = vol[1]
    impliedvol = vol[0]
    underlyprice = vol[2]
    param = sviFit(impliedvol, strike)[0]
    print(f"The a is {param[0]}, b is {param[1]}, rho is {param[2]}, m is {param[3]}, sigma is {param[4]}")


runfunc_2("AAPL", "2022-06-17")
# colab with Hu