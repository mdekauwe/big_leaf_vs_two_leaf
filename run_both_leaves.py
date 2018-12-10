#!/usr/bin/env python
"""
For a synthetic 48 half hour time window, solve 30-minute coupled A-gs(E) using
a big-leaf and 2-leaf approximation and make a comparison plot

"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from big_leaf import CoupledModel as BigLeaf
#from big_leaf_depFarq import CoupledModel as BigLeaf
import pandas as pd
from two_leaf import CoupledModel as TwoLeaf
import constants as c
import parameters as p
from get_days_met_forcing import get_met_data
from utils import calc_esat

__author__  = "Martin De Kauwe"
__version__ = "1.0 (09.11.2018)"
__email__   = "mdekauwe@gmail.com"


def main():

    doy = 180.
    #
    ## Met data ...
    #
    (par, tair, vpd) = get_met_data(p.lat, p.lon, doy)

    # more realistic VPD
    rh = 40.
    esat = calc_esat(tair)
    ea = rh / 100. * esat
    vpd = (esat - ea) * c.PA_2_KPA
    vpd = np.where(vpd < 0.05, 0.05, vpd)

    ##
    ### Run Big-leaf
    ##

    B = BigLeaf(p.g0, p.g1, p.D0, p.gamma, p.Vcmax25, p.Jmax25, p.Rd25,
                p.Eaj, p.Eav, p.deltaSj, p.deltaSv, p.Hdv, p.Hdj, p.Q10,
                p.leaf_width, p.SW_abs, gs_model="medlyn")

    An_bl = np.zeros(48)
    gsw_bl = np.zeros(48)
    et_bl = np.zeros(48)
    tcan_bl = np.zeros(48)

    hod = 0
    for i in range(len(par)):

        (An_bl[i], gsw_bl[i],
         et_bl[i], tcan_bl[i]) = B.main(tair[i], par[i], vpd[i],
                                        p.wind, p.pressure, p.Ca, doy, hod,
                                        p.lat, p.lon, p.LAI)

        hod += 1
    ##
    ### Run 2-leaf
    ##

    T = TwoLeaf(p.g0, p.g1, p.D0, p.gamma, p.Vcmax25, p.Jmax25, p.Rd25,
                p.Eaj, p.Eav, p.deltaSj, p.deltaSv, p.Hdv, p.Hdj, p.Q10,
                p.leaf_width, p.SW_abs, gs_model="medlyn")

    An_tl = np.zeros(48)
    gsw_tl = np.zeros(48)
    et_tl = np.zeros(48)
    tcan_tl = np.zeros(48)

    hod = 0
    for i in range(len(par)):

        (An_tl[i], gsw_tl[i],
         et_tl[i], tcan_tl[i],
         __,__,__,__,
         __,__) = T.main(tair[i], par[i], vpd[i], p.wind, p.pressure, p.Ca,
                         doy, hod, p.lat, p.lon, p.LAI)

        hod += 1


    fig = plt.figure(figsize=(16,4))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color'] = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor'] = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.plot(np.arange(48)/2., An_bl, label="Big leaf")
    ax1.plot(np.arange(48)/2., An_tl, label="Two leaf")
    ax1.legend(numpoints=1, loc="best")
    ax1.set_ylabel("$A_{\mathrm{n}}$ ($\mathrm{\mu}$mol m$^{-2}$ s$^{-1}$)")

    ax2.plot(np.arange(48)/2., et_bl * c.MOL_TO_MMOL, label="Big leaf")
    ax2.plot(np.arange(48)/2., et_tl * c.MOL_TO_MMOL, label="Two leaf")
    ax2.set_ylabel("E (mmol m$^{-2}$ s$^{-1}$)")
    ax2.set_xlabel("Hour of day")


    ax3.plot(np.arange(48)/2., tcan_bl, label="Tcanopy$_{1leaf}$")
    ax3.plot(np.arange(48)/2., tcan_tl, label="Tcanopy$_{2leaf}$")
    ax3.plot(np.arange(48)/2., tair, label="Tair")
    ax3.set_ylabel("Temperature (deg C)")
    ax3.legend(numpoints=1, loc="best")

    ax1.locator_params(nbins=6, axis="y")
    ax2.locator_params(nbins=6, axis="y")

    plt.show()
    fig.savefig("/Users/%s/Desktop/A_E_Tcan.pdf" % (os.getlogin()),
                bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    main()
