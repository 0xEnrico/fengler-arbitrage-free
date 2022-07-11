import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from fengler import *
import matplotlib.pyplot as plt
import matplotlib as mpl

# This example reads SPX_2022_03_04_10_01_00.parquet file which contain SPX options data for 04 Mar 2022 10:01
# and applies Fengler's arbitrage free interpolation [1], then plots the data
#
# [1] Arbitrage-Free Smoothing of the Implied Volatility Surface, Matthias R. Fengler, 2005 (https://core.ac.uk/reader/6978470)

if __name__ == "__main__":

    # Read option data from parquet file, excluding expiries with zero forward (not enough data)
    option_data = pq.read_table("SPX_2022_03_04_10_01_00.parquet")
    option_data = option_data.filter(pa.compute.greater(option_data['F'], 0.))
    strikes = np.unique(np.array(option_data["K"]))
    expiries = np.unique(np.array(option_data["T"]))
    impl_vols_bid = np.zeros([len(expiries),len(strikes)])
    impl_vols_ask = np.zeros([len(expiries),len(strikes)])
    impl_vols = np.zeros([len(expiries),len(strikes)])
    forwards = []
    interest_rates = []
    i = 0
    for T in expiries:
        option_data_single_expiry = option_data.filter(pa.compute.equal(option_data["T"], T))
        impl_vols_bid[i] = np.array(option_data_single_expiry["IV_Bid"])
        impl_vols_ask[i] = np.array(option_data_single_expiry["IV_Ask"])
        impl_vols[i] = np.array(option_data_single_expiry["IV"])
        forwards.append(option_data_single_expiry["F"][0].as_py())
        interest_rates.append(option_data_single_expiry["R"][0].as_py())
        i+=1

    # In the dataset, implied vols < 0 are undefined, we need to convert them to NaN before calling calibFenglerSplineNodes
    impl_vols_bid = np.where(impl_vols_bid<=0,np.nan,impl_vols_bid)
    impl_vols_ask = np.where(impl_vols_ask<=0,np.nan,impl_vols_ask)
    impl_vols = np.where(impl_vols<=0,np.nan,impl_vols)

    calibFenglerSplineNodes(strikes, forwards, expiries, interest_rates, impl_vols)

    # Plot option data as total implied variance
    mpl.rcParams['lines.linewidth'] = 1
    i = 0
    for T in expiries:
        plt.scatter(strikes, impl_vols_bid[i]*impl_vols_bid[i]*T, marker=mpl.markers.CARETUP, c='b')
        plt.scatter(strikes, impl_vols_ask[i]*impl_vols_ask[i]*T, marker=mpl.markers.CARETDOWN, c='b')
        plt.scatter(strikes, impl_vols[i]*impl_vols[i]*T, marker='.', c='b')
        i+=1
    plt.show()
