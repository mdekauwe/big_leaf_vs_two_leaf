#!/usr/bin/env python
"""
For a synthetic 48 half hour time window, solve 30-minute coupled A-gs(E) using
a big-leaf and 2-leaf approximation and make a comparison plot

"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, exp, sqrt, acos, asin
import random
from big_leaf import CoupledModel as BigLeaf
#from big_leaf_depFarq import CoupledModel as BigLeaf

from two_leaf import CoupledModel as TwoLeaf
import constants as c
from get_days_met_forcing import get_met_data

__author__  = "Martin De Kauwe"
__version__ = "1.0 (09.11.2018)"
__email__   = "mdekauwe@gmail.com"


def main():

    lat = -23.575001
    lon = 152.524994
    doy = 180.0
    #
    ## Met data ...
    #
    (par, tair, vpd) = get_met_data(lat, lon, doy)
    wind = 2.5
    pressure = 101325.0
    Ca = 400.0

    #
    ## Parameters
    #
    g0 = 0.001
    g1 = 2.0
    D0 = 1.5 # kpa
    Vcmax25 = 60.0
    Jmax25 = Vcmax25 * 1.67
    Rd25 = 2.0
    Eaj = 30000.0
    Eav = 60000.0
    deltaSj = 650.0
    deltaSv = 650.0
    Hdv = 200000.0
    Hdj = 200000.0
    Q10 = 2.0
    gamma = 0.0
    leaf_width = 0.02
    LAI = 1.5

    # Cambell & Norman, 11.5, pg 178
    # The solar absorptivities of leaves (-0.5) from Table 11.4 (Gates, 1980)
    # with canopies (~0.8) from Table 11.2 reveals a surprising difference.
    # The higher absorptivityof canopies arises because of multiple reflections
    # among leaves in a canopy and depends on the architecture of the canopy.
    SW_abs = 0.8 # use canopy absorptance of solar radiation

    ##
    ### Run Big-leaf
    ##

    B = BigLeaf(g0, g1, D0, gamma, Vcmax25, Jmax25, Rd25, Eaj, Eav,
                deltaSj, deltaSv, Hdv, Hdj, Q10, leaf_width, SW_abs,
                gs_model="medlyn")

    An_bl = np.zeros(48)
    gsw_bl = np.zeros(48)
    et_bl = np.zeros(48)
    tcan_bl = np.zeros(48)

    hod = 0
    for i in range(len(par)):

        (An_bl[i], gsw_bl[i],
         et_bl[i], tcan_bl[i]) = B.main(tair[i], par[i], vpd[i],
                                        wind, pressure, Ca, doy, hod,
                                        lat, lon, LAI)

        hod += 1
    ##
    ### Run 2-leaf
    ##

    T = TwoLeaf(g0, g1, D0, gamma, Vcmax25, Jmax25, Rd25, Eaj, Eav,
                deltaSj, deltaSv, Hdv, Hdj, Q10, leaf_width, SW_abs,
                gs_model="medlyn")

    An_tl = np.zeros(48)
    gsw_tl = np.zeros(48)
    et_tl = np.zeros(48)
    tcan_tl = np.zeros(48)

    hod = 0
    for i in range(len(par)):

        (An_tl[i], gsw_tl[i],
         et_tl[i], tcan_tl[i]) = T.main(tair[i], par[i], vpd[i],
                                        wind, pressure, Ca, doy, hod,
                                        lat, lon, LAI)

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
