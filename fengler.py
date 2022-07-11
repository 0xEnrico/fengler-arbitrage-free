
from turtle import clone
import numpy as np
from qpsolvers import solve_qp
import thinplate as tps
import py_vollib.black

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

    # [~, idx] = ismember(Y,expiries);
    # option = makeVanillaOption(forward(idx).*X', Y', ones(size(idx))');
    # bs_model = makeBsModel(impl_volsinterpolated);
    # mkt_data = makeMarketData(NaN, forward(idx), interest_rate(idx), NaN);
    # pre_smooth_call_price = calcBsPriceAnalytic(bs_model, option, mkt_data, \
    #     'forward');

    # # ensure that output dimensions are correct, each column is one smile
    # pre_smooth_call_price = reshape(pre_smooth_call_price, length(expiries), length(kappa)) 

    # return pre_smooth_call_price

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
    Aeq = np.vstack([Q, -R.transpose()])
    beq = np.zeros(np.size(Aeq,axis=0))

    # estimate the quadratic program
    x = solve_qp(B, -y, A, b, Aeq, beq, lb, ub, solver='quadprog', initvals=x0)

    # First n values of x are the points g_i
    g = x[0:n-1]

    # Remaining values of x are the second derivatives
    gamma = np.concatenate([0., x[n:2*n-3], 0.], axis=None)

    return g, gamma

def calibFenglerSplineNodes(strikes, forwards, expiries, interest_rates, impl_vols, \
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
        A = np.matrix([np.concatenate([1./h[0], -1./h[0], np.zeros(n-2), h[0]/6., np.zeros(n-3)], axis=None), \
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
