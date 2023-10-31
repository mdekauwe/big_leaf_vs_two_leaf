#!/usr/bin/env python
"""
For a synthetic 48 half hour time window, solve 30-minute coupled A-gs(E) using
a big-leaf and 2-leaf approximation and make a comparison plot

"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import constants as c
import parameters as p
from utils import calc_esat
from big_leaf import Canopy as BigLeaf
from two_leaf import Canopy as TwoLeaf
from get_days_met_forcing import get_met_data

__author__  = "Martin De Kauwe"
__version__ = "1.0 (09.11.2018)"
__email__   = "mdekauwe@gmail.com"


def main():


    ##
    ### Run 2-leaf
    ##
    T = TwoLeaf(p, gs_model="medlyn", gs_vpd_sens=0.5)


    doy = 180.
    lat = 50.6
    lon = 0.2

    ndays = 365
    n_halfhrs = 48
    An_hlfhr = np.zeros(n_halfhrs)
    et_hlfhr = np.zeros(n_halfhrs)
    tcan_hlfhr = np.zeros(n_halfhrs)
    An_doy = np.zeros(ndays)
    et_doy = np.zeros(ndays)
    tcan_doy = np.zeros(ndays)

    for doy in range(ndays):

        (par, tair, vpd) = get_met_data(lat, lon, doy)

        # more realistic VPD
        rh = 40.
        esat = calc_esat(tair)
        ea = rh / 100. * esat
        vpd = (esat - ea) * c.PA_2_KPA
        vpd = np.where(vpd < 0.05, 0.05, vpd)

        #
        ##  Fixed met stuff
        #
        wind = 2.5
        pressure = 101325.0
        Ca = 400.0
        LAI = p.LAI

        hod = 0
        tsoil = np.mean(tair)
        for i in range(len(par)):

            (An, et, Tcan,
             apar, lai_leaf) = T.main(tair[i], par[i], vpd[i], wind,
                                      pressure, Ca, doy, hod/2., LAI, tsoil)

            sun_frac = lai_leaf[c.SUNLIT] / np.sum(lai_leaf)
            sha_frac = lai_leaf[c.SHADED] / np.sum(lai_leaf)
            An_hlfhr[i] = np.sum(An)
            et_hlfhr[i] = np.sum(et)
            tcan_hlfhr[i] = (Tcan[c.SUNLIT] * sun_frac) + (Tcan[c.SHADED] * sha_frac)

            hod += 1

        An_doy[doy] = np.mean(An_hlfhr)
        et_doy[doy] = np.mean(et_hlfhr)
        tcan_doy[doy] = np.mean(tcan_hlfhr)


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

    ax1.plot(np.arange(365), An_doy, label="Two leaf")
    ax1.legend(numpoints=1, loc="best")
    ax1.set_ylabel("$A_{\mathrm{n}}$ ($\mathrm{\mu}$mol m$^{-2}$ s$^{-1}$)")

    ax2.plot(np.arange(365), et_doy * c.MOL_TO_MMOL, label="Two leaf")
    ax2.set_ylabel("E (mmol m$^{-2}$ s$^{-1}$)")
    ax2.set_xlabel("Hour of day")


    ax3.plot(np.arange(365), tcan_doy, label="Tcanopy$_{2leaf}$")
    #ax3.plot(np.arange(365), tair, label="Tair")
    ax3.set_ylabel("Temperature (deg C)")
    ax3.legend(numpoints=1, loc="best")

    ax1.locator_params(nbins=6, axis="y")
    ax2.locator_params(nbins=6, axis="y")

    plt.show()
    fig.savefig("/Users/%s/Desktop/A_E_Tcan.pdf" % (os.getlogin()),
                bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    main()
