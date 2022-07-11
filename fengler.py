
from turtle import clone
import numpy as np
from qpsolvers import solve_qp
import thinplate as tps
import py_vollib.black, py_vollib.black.implied_volatility

def calcFenglerPreSmoothedPrices(strikes, expiries, \
    impl_vols, forwards, interest_rates, from_log_moneyness=-1e100, to_log_moneyness=1e100, \
    fwd_moneyness_step=1e-2, presmoother_lambda=0.):
    # This function calculates the pre-smoothed call option prices
    # In the following we assume:
    #   N: number of strikes
    #   M: number of maturities
    #
    # Parameters
    #   kappa_: Numpy vector of equally spaced forward moneyness
    #   fwd_moneyness_: Actual forward moneyness vector
    #   expiries: numpy array of M time-to-expiries (must be in increasing order)
    #   impl_vols: numpy matrix M x N of implied volatilities
    #   impl_forward_ [vector]: Implied forwards vector
    #   interest_rate_ [vector]: Interest rates vector
    # Out
    #   pre_smooth_call_price [vector]: Pre-smoothed call prices vector
    #   kappa [vector]: Forward moneyness vector
    #   expiries [vector]: Time-to-maturities vector
    #   forward [vector]: Implied forwards vector
    #   interest_rate [vector]: Interest rates vector
    # Source
    #   based on https://www.mathworks.com/matlabcentral/fileexchange/ \
    #   46253-arbitrage-free-smoothing-of-the-implied-volatility-surface

    # check that input is consistent
    if len(expiries) != len(forwards):
        raise Exception('expiries and forwards must be same length')
    if len(expiries) != len(interest_rates):
        raise Exception('expiries and interest_rates must be same length')
    if np.size(impl_vols, axis=0) != len(expiries):
        raise Exception('impl_vols rows must be same length as expiries')
    if np.size(impl_vols, axis=1) != len(strikes):
        raise Exception('impl_vols columns must be same length as strikes')

    # forward moneyness grid
    fwd_moneyness = np.array([np.array(forwards)]).transpose()/np.array(strikes)
    kappa = np.arange(np.floor(max(np.exp(from_log_moneyness), np.min(fwd_moneyness))*10.)/10., np.ceil(min(np.exp(to_log_moneyness), np.max(fwd_moneyness))*10.)/10., fwd_moneyness_step)

    # thin-plate spline
    exp_matrix = np.array([expiries]).transpose()*np.ones(np.size(impl_vols, axis=1))
    x = np.array([fwd_moneyness.flatten(), exp_matrix.flatten(), impl_vols.flatten()]).transpose()
    x = x[~np.isnan(x).any(axis=1)]
    thin_plate_spline = tps.TPS.fit(x, presmoother_lambda)

    # fit implied vols
    grid_moneyness, grid_expiry = np.meshgrid(kappa,expiries,indexing='xy')
    grid=np.array([grid_moneyness.flatten(), grid_expiry.flatten()]).transpose()
    impl_vols_interpolated = tps.TPS.z(grid, x, thin_plate_spline)
    pre_smooth_call_prices = np.zeros(np.size(impl_vols_interpolated))

    # calculation of call prices
    grid_moneyness_f, grid_forwards = np.meshgrid(kappa,forwards,indexing='xy')
    gridf=np.array([grid_moneyness_f.flatten(), grid_forwards.flatten()]).transpose()
    grid_moneyness_r, grid_rates = np.meshgrid(kappa,interest_rates,indexing='xy')
    gridr=np.array([grid_moneyness_r.flatten(), grid_rates.flatten()]).transpose()
    for i in range(0, np.size(grid, axis=0)):
        pre_smooth_call_prices[i] = py_vollib.black.black('c', gridf[i,1], gridf[i,1]/grid[i,0], grid[i,1], gridr[i,1], impl_vols_interpolated[i])

    impl_vols_interpolated = impl_vols_interpolated.reshape(np.size(grid_moneyness, axis=0), np.size(grid_moneyness, axis=1))
    pre_smooth_call_prices = pre_smooth_call_prices.reshape(np.size(grid_moneyness, axis=0), np.size(grid_moneyness, axis=1))

    return kappa, grid_moneyness, grid_expiry, impl_vols_interpolated, pre_smooth_call_prices

def solveFenglerQuadraticProgram(u, h, y, A, b, lb, ub, lambd=1e-2):
    # Function to solve the quadratic program of Fenlger's implied volatility
    # surface smoothing algorithm
    # In
    #   u [vector]: Moneyness of nodes
    #   y [vector]: Call prices at nodes
    #   A [matrix]: Matrix for linear inequality (Ax <= b)
    #   b [vector]: Vector of inequality values
    #   lb [vector]: Lower bound
    #   ub [vector]: Upper bound
    #   lambd [float]: Smoothing parameter
    # Out
    #   g [vector]: Vector of smoothed call option prices
    #   gamma [vector]: Vector of second derivatives at nodes
    # Source
    #   based on https://www.mathworks.com/matlabcentral/fileexchange/ ...
    #   46253-arbitrage-free-smoothing-of-the-implied-volatility-surface

    n = len(u)

    # check that input is consistent
    if len(y) != len(u):
        raise Exception('length of y is incorrect')

    lb = np.where(lb>ub,ub,lb)

    # set up estimation and restriction matrices
    #h = np.diff(u,1)

    Q = np.zeros([n, n-2])
    for j in range(1, n-1):
        Q[j-1, j-1] = 1./h[j-1]
        Q[j, j-1] = -1./h[j-1] - 1./h[j]
        Q[j+1, j-1] = 1./h[j]

    R = np.zeros([n-2, n-2])
    for i in range(1, n-1):
        R[i-1, i-1] = (h[i-1]+h[i])/3.
        if i < n-2:
            R[i-1, i] = h[i]/6.
            R[i, i-1] = h[i]/6.

    # set-up problem min_x -y'x + 0.5 x'Bx

    # linear term
    y = np.concatenate([y, np.zeros(n-2)], axis=None)

    #quadratic term
    B = np.vstack([np.hstack([np.diag(np.ones(n)), np.zeros([n, np.size(R, axis=1)])]), np.hstack([np.zeros([np.size(R, axis=0), n]), lambd*R])])

    # initial guess
    x0 = np.copy(y)
    x0[n:] = 1e-3

    # equality constraint Aeq x = beq
    Aeq = np.hstack([Q.transpose(), -R])
    beq = np.zeros(np.size(Aeq,axis=0))

    # estimate the quadratic program
    x = solve_qp(B, -y, A, b, Aeq, beq, lb, ub, solver='quadprog', initvals=x0)

    if x is not None:
        # First n values of x are the points g_i
        g = x[0:n]

        # Remaining values of x are the second derivatives
        gamma = np.concatenate([0., x[n:2*n-2], 0.], axis=None)

        # Gamma has to be greater equal zero
        # Spline values have to be greater equal zero
        if np.any(g<-1e-8, axis=None) or np.any(gamma<-1e-8, axis=None):
            g = np.full(n, np.nan)
            gamma = np.full(n, np.nan)
    else:
        g = np.full(n, np.nan)
        gamma = np.full(n, np.nan)

    return g, gamma

def calcFenglerSpline(v, u, g, gamma):
    # Function to evaluate smoothing spline based on g and gamma
    # Parameters
    #   v [vector]: Vector of strikes where to evaluate smoothing spline
    #   u [vector]: Vector of strikes
    #   g [vector]: Vector of prices
    #   gamma [vector]: Vector of second derivatives
    # Outputs
    #   smooth_prices [vector]: Vector of interpolated prices
    # Source
    #   based on https://www.mathworks.com/matlabcentral/fileexchange/ ...
    #   46253-arbitrage-free-smoothing-of-the-implied-volatility-surface

    n = len(u)
    m = len(v)
    smooth_prices = np.zeros(m)

    for s in range(0, m):
        for i in range(0, n-1):
            h = u[i+1] - u[i]
            if u[i] <= v[s] and v[s] <= u[i+1]:
                smooth_prices[s] = ((v[s]-u[i])*g[i+1] + (u[i+1]-v[s])*g[i])/h \
                        - (v[s]-u[i])*(u[i+1]-v[s])/6. \
                        * ((1.+(v[s]-u[i])/h)*gamma[i+1] \
                        + (1.+(u[i+1]-v[s])/h)*gamma[i])
      
        if v[s] < np.min(u):
            dg = (g[1]-g[0])/(u[1]-u[0]) - (u[1]-u[0])/6.*gamma[1]
            smooth_prices[s] = g[0] - (u[0] - v[s])*dg
        
        if v[s] > np.max(u):
            dg = (g[n]-g[n-1])/(u[n]-u[n-1]) + (u[n]-u[n-1])/6.*gamma[n-1]
            smooth_prices[s] = g[n] + (v[s] - u[n])*dg

    return smooth_prices

def calcFenglerSmoothIvQs(strikes, forwards, expiries, interest_rates, impl_vols, \
    from_log_moneyness=-1e100, to_log_moneyness=1e100, fwd_moneyness_step=1e-2, presmoother_lambda=0.):
    # Function to calculate nodes of Fengler's smoothing spline
    # In the following we assume:
    #   N: number of strikes
    #   M: number of maturities
    #
    # Parameters
    #   strikes: numpy array of N strike prices
    #   forwards: numpy array of M forwards
    #   expiries: numpy array of M time-to-expiries (must be in increasing order)
    #   interest_rates: numpy array of M interest rates
    #   impl_vols: numpy 2d array M x N of implied volatilities. Must be NaN where implied vol is not defined
    # Out
    #   kappa: numpy array of forward moneyness vector
    #   smooth_call_price: numpy 2d array of smooth call prices
    #   smooth_impl_vol: numpy 2d array of smooth implied vols
    #   smooth_total_variance: numpy 2d array of smooth total variance

    # step 1: pre-smoother
    kappa, grid_moneyness, grid_expiry, impl_vols_interpolated, pre_smooth_call_prices = \
         calcFenglerPreSmoothedPrices(strikes, expiries, impl_vols, forwards, interest_rates, \
        from_log_moneyness, to_log_moneyness, fwd_moneyness_step, presmoother_lambda)

    # step2: iterative smoothing of pricing surface
    T = len(expiries)
    K = len(kappa)
    g = np.zeros([T,K])
    gamma = np.zeros([T,K])
    u = np.zeros([T,K])
    for t in range(T-1, -1, -1):
        u[t] = kappa*forwards[t]
        y = pre_smooth_call_prices[t]
        n = len(u[t])
        h = np.diff(u[t])
        # inequality constraints A x <= b
        # -(g_2 - g_1)/h_1 + h_1/6 gamma(2) <= e^(-expiries*r)
        #  (g_n - g_(n-1))/h_(n-1) + h_(n-1)/6 gamma(n-1) <= 0
        A = np.array([np.concatenate([1./h[0], -1./h[0], np.zeros(n-2), h[0]/6., np.zeros(n-3)], axis=None), \
            np.concatenate([np.zeros(n-2), -1./h[n-2], 1./h[n-2], np.zeros(n-3), h[n-2]/6.], axis=None)])
        b = np.array([np.exp(-expiries[t]*interest_rates[t]), 0])
        # set-up lower bound
        lb = np.concatenate([np.maximum(np.exp(-interest_rates[t]*expiries[t])*(forwards[t]-u[t].transpose()), 0.), np.zeros(n-2)], axis=None)
        # set-up upper bound
        if t == T-1:
            ub = np.concatenate([np.exp(-interest_rates[t]*expiries[t])*forwards[t], np.full(2*n-3, np.inf)], axis=None)
        else:
            ub = np.concatenate([np.exp(interest_rates[t+1]*expiries[t+1]-interest_rates[t]*expiries[t])*forwards[t]/forwards[t+1]*g[t+1].transpose(), np.full(n-2, np.inf)], axis=None)
        g[t], gamma[t] = solveFenglerQuadraticProgram(u[t], h, y, A, b, lb, ub)

    # calculate smooth call price and implied vol
    S = len(strikes)
    smooth_call_price = np.full([T,S], np.nan)
    smooth_impl_vol = np.full([T,S], np.nan)
    smooth_total_variance = np.full([T,S], np.nan)
    for t in range(0,T):
        if not np.any(np.isnan(g[t])) and not np.any(np.isnan(gamma[t])):
            smooth_call_price[t] = calcFenglerSpline(strikes, u[t], g[t], gamma[t])
            for i in range(0, S):
                try:
                    smooth_impl_vol[t][i] = py_vollib.black.implied_volatility.implied_volatility(smooth_call_price[t][i], forwards[t], strikes[i], interest_rates[t], expiries[t], 'c')
                except:
                    smooth_impl_vol[t][i] = np.nan
        smooth_total_variance[t] = (smooth_impl_vol[t]**2)*expiries[t]

    grid_moneyness = np.array([np.array(forwards)]).transpose()/np.array(strikes)
    return grid_moneyness, smooth_call_price, smooth_impl_vol, smooth_total_variance
