#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
For a synthetic 48 half hour time window, solve 30-minute coupled A-gs(E) using
a big-leaf and 2-leaf approximation and make a comparison plot

"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, cos, sin, exp, sqrt, acos, asin
import random
from big_leaf import CoupledModel as BigLeaf
from utils import calc_esat
#from big_leaf_depFarq import CoupledModel as BigLeaf

from two_leaf import CoupledModel as TwoLeaf
from two_leaf_opt import CoupledModel as TwoLeafOpt
import constants as c
from get_days_met_forcing import get_met_data
from radiation import calculate_solar_geometry

# first make sure that own modules from parent dir can be loaded
#script_dir = '/srv/ccrc/data15/z5153939/two_leaf_optimisation'
script_dir = '/Users/mdekauwe/src/python/two_leaf_optimisation'
sys.path.append(os.path.abspath(script_dir))

from OptModel.CH2OCoupler import profit_psi
from OptModel.Utils.default_params import default_params
from OptModel.PlantModel import absorbed_radiation_2_leaves

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

    # more realistic VPD
    rh = 40.
    esat = calc_esat(tair)
    ea = rh / 100. * esat
    vpd = (esat - ea) * c.PA_2_KPA
    vpd = np.where(vpd < 0.05, 0.05, vpd)

    #plt.plot(vpd)
    #plt.show()
    #sys.exit()
    
    ## Parameters
    #
    g0 = 0.001
    g1 = 1.5635 # Puechabon
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
    LAI = 3.

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

    ##
    ### Run 2-leaf opt
    ##

    T = TwoLeafOpt(g0, g1, D0, gamma, Vcmax25, Jmax25, Rd25, Eaj, Eav,
                deltaSj, deltaSv, Hdv, Hdj, Q10, leaf_width, SW_abs,
                gs_model="medlyn")

    An_tlo = np.zeros(48)
    gsw_tlo = np.zeros(48)
    et_tlo = np.zeros(48)
    tcan_tlo = np.zeros(48)

    hod = 0
    for i in range(len(par)):

        (An_tlo[i], gsw_tlo[i],
         et_tlo[i], tcan_tlo[i]) = T.main(tair[i], par[i], vpd[i],
                                        wind, pressure, Ca, doy, hod,
                                        lat, lon, LAI)

        hod += 1


    ##
    ### Run 2-leaf Manon
    ##

    Ao = np.zeros(48)
    gso = np.zeros(48)
    Eo = np.zeros(48)

    hod = 0
    for i in range(len(par)):

        cos_zenith = calculate_solar_geometry(doy, hod, lat, lon)
        zenith_angle = np.rad2deg(np.arccos(cos_zenith))
        elevation = 90.0 - zenith_angle
        if elevation > 0.0 and par[i] > 50.0:


            p = declared_params()

            p.PPFD = par[i]
            p.sw_rad_day = par[i] * c.PAR_2_SW
            p.LAI = LAI
            p.coszen = cos_zenith
            p.VPD = vpd[i]
            p.precip = 0
            p.Tair = tair[i]
            p.Vmax25 = Vcmax25
            p.gamstar25 = 0.436 # CO2 compensation point @ 25 degC (Pa)
            p.g1 = g1
            p.CO2 = Ca / 101.25
            p.JV = 1.67
            p.Rlref = Rd25
            p.Ej = Eaj
            p.Ev = Eav
            p.deltaSv = deltaSv
            p.deltaSj = deltaSj
            p.max_leaf_width = leaf_width
            p.gamstar25 = 0.422222  # 42.75 / 101.25 umol m-2 s-1
            p.Kc25 = 41.0       # 404.9 umol m-2 s-1
            p.Ko25 = 28202.0    # 278.4 mmol mol-1
            p.O2 = 20.670000    # 210 mmol mol-1
            p.alpha = 0.24
            p.Egamstar= 37830.0
            p.Ec = 79430.0
            p.Eo = 36380.0
            p.P50 = 2.02 # Puechabon
            p.P88 = 4.17
            p.kmax = 0.862457122856143

            _, _, fscale2can = absorbed_radiation_2_leaves(p)
            p = p.append(pd.Series([np.nansum(fscale2can)], index=['fscale']))


            #for i in p:
            #    print (p)
            #sys.exit()
            try:
                fstom_opt_psi, Eo[i], gso[i], Ao[i], _, _ = profit_psi(p,
                                                               photo='Farquhar',
                                                               res='med',
                                                               case=2)
                p = p.append(pd.Series([fstom_opt_psi],
                             index=['fstom_opt_psi']))

            except (ValueError, AttributeError):
                (Eo[i], gso[i], Ao[i]) = (0., 0., 0.)

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

    #ax1.plot(np.arange(48)/2., An_bl, label="Big leaf")
    ax1.plot(np.arange(48)/2., An_tl, label="Two leaf")
    ax1.plot(np.arange(48)/2., An_tlo, label="Two leaf Opt")
    ax1.plot(np.arange(48)/2., Ao, label="Two leaf Manon")
    ax1.legend(numpoints=1, loc="best")
    ax1.set_ylabel("$A_{\mathrm{n}}$ ($\mathrm{\mu}$mol m$^{-2}$ s$^{-1}$)")

    #ax2.plot(np.arange(48)/2., et_bl * c.MOL_TO_MMOL, label="Big leaf")
    ax2.plot(np.arange(48)/2., et_tl * c.MOL_TO_MMOL, label="Two leaf")
    ax2.plot(np.arange(48)/2., et_tlo * c.MOL_TO_MMOL, label="Two leaf opt")
    ax2.plot(np.arange(48)/2., Eo , label="Two leaf Manon")
    ax2.set_ylabel("E (mmol m$^{-2}$ s$^{-1}$)")
    ax2.set_xlabel("Hour of day")


    #ax3.plot(np.arange(48)/2., tcan_bl, label="Tcanopy$_{1leaf}$")
    ax3.plot(np.arange(48)/2., tcan_tl, label="Tcanopy$_{2leaf}$")
    ax3.plot(np.arange(48)/2., tair, label="Tair")
    ax3.set_ylabel("Temperature (deg C)")
    ax3.legend(numpoints=1, loc="best")

    ax1.locator_params(nbins=6, axis="y")
    ax2.locator_params(nbins=6, axis="y")

    plt.show()
    fig.savefig("/Users/%s/Desktop/A_E_Tcan.pdf" % (os.getlogin()),
                bbox_inches='tight', pad_inches=0.1)


def declared_params():

    # make the default param class a pandas object
    p = default_params()
    attrs = vars(p)
    p = {item[0]: item[1] for item in attrs.items()}
    p = pd.Series(p)

    # add missing params
    p = p.append(pd.Series([25., 1500., 1., ], index=['Tair', 'PPFD', 'VPD']))

    # deal with the radiation component
    p.LAI = 1.
    p = p.append(pd.Series([0.6], index=['coszen']))

    return p



if __name__ == "__main__":

    main()
