
from turtle import clone
import numpy as np
from qpsolvers import solve_qp
import thinplate as tps

def calcFenglerPreSmoothedPrices(kappa, fwd_moneyness, expiries, \
    impl_vols, forwards, interest_rates):
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

    # thin-plate spline
    x = np.array([fwd_moneyness, expiries, impl_vols]).transpose()
    thin_plate_spline = tps.TPS.fit(x, 1.)

    grid = np.meshgrid(kappa,expiries)
    impl_volsinterpolated = tps.TPS.z(grid, x, thin_plate_spline)

    # # calculation of call prices
    # [~, idx] = ismember(Y,expiries);
    # option = makeVanillaOption(forward(idx).*X', Y', ones(size(idx))');
    # bs_model = makeBsModel(impl_volsinterpolated);
    # mkt_data = makeMarketData(NaN, forward(idx), interest_rate(idx), NaN);
    # pre_smooth_call_price = calcBsPriceAnalytic(bs_model, option, mkt_data, \
    #     'forward');

    # # ensure that output dimensions are correct, each column is one smile
    # pre_smooth_call_price = reshape(pre_smooth_call_price, length(expiries), length(kappa)) 

    # return pre_smooth_call_price

def solveFenglerQuadraticProgram(u, h, y, A, b, lb, ub, lambda_=1e-2):
    # Function to solve the quadratic program of Fenlger's implied volatility
    # surface smoothing algorithm
    # In
    #   u [vector]: Moneyness of nodes
    #   y [vector]: Call prices at nodes
    #   A [matrix]: Matrix for linear inequality (Ax <= b)
    #   b [vector]: Vector of inequality values
    #   lb [vector]: Lower bound
    #   ub [vector]: Upper bound
    #   lambda_ [float]: Smoothing parameter
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
    B = np.vstack([np.hstack([np.diag(np.ones(n)), np.zeros([n, np.size(R, axis=1)])]), np.hstack([np.zeros([np.size(R, axis=0), n]), lambda_*R])])

    # initial guess
    x0 = y.clone()
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

def calibFenglerSplineNodes(strikes, forwards, expiries, interest_rates, impl_vols):
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
    #   impl_vols: numpy matrix M x N of implied volatilities

    # step 1: pre-smoother
    fwd_moneyness = np.matrix(np.array(forwards)).transpose()*np.array(strikes)
    kappa = range(np.floor(np.min(fwd_moneyness*10.))/10., np.ceil(np.max(fwd_moneyness*10.))/10., 0.01)
    pre_smooth_call_price = \
        calcFenglerPreSmoothedPrices(kappa, fwd_moneyness, expiries, \
        impl_vols, forwards, interest_rates)
    # step2: iterative smoothing of pricing surface
    T = len(expiries)
    K = len(kappa)
    g = np.zeros([K,T])
    gamma = np.zeros([K,T])
    u = np.zeros([K,T])
    for t in range(T-1, -1, -1):
        u[t] = kappa*forwards[t]
        y = pre_smooth_call_price[t]
        n = len(u[t])
        h = np.diff(u[t])
        # inequality constraints A x <= b
        # -(g_2 - g_1)/h_1 + h_1/6 gamma(2) <= e^(-expiries*r)
        #  (g_n - g_(n-1))/h_(n-1) + h_(n-1)/6 gamma(n-1) <= 0
        A = np.matrix([np.concatenate([1./h[0], -1./h[0], np.zeros(n-2), h[0]/6., np.zeros(n-3)], axis=None), \
            np.concatenate([np.zeros(n-2), -1./h[n-2], 1./h[n-2], np.zeros(n-3), h[n-2]/6.], axis=None)])
        b = np.array([np.exp(-expiries[t]*interest_rates[t]), 0])
        # set-up lower bound
        lb = np.concatenate([np.max(np.exp(-interest_rates[t]*expiries[t])*(forwards[t]-u[t].transpose()), 0.), np.zeros(n-2)], axis=None)
        # set-up upper bound
        if t == T-1:
            ub = np.concatenate([np.exp(-interest_rates[t]*expiries[t])*forwards[t], np.full(2*n-3, np.inf)], axis=None)
        else:
            ub = np.concatenate([np.exp(interest_rates[t+1]*expiries[t+1]-interest_rates[t]*expiries[t])*forwards[t]/forwards[t+1]*g[t+1].transpose(), np.full(n-2, np.inf)], axis=None)
        g[t], gamma[t] = solveFenglerQuadraticProgram(u[t], h, y, A, b, lb, ub)
