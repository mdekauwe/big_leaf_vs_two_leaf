#!/usr/bin/env python
"""
For a synthetic 48 half hour time window, calculate GPP/E using a big-leaf and
2-leaf approximation.

This is the wrapper script that calls both leaves and makes a plot ...
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, exp, sqrt, acos, asin
import random
from weather_generator import WeatherGenerator
from big_leaf import CoupledModel as BigLeaf

__author__  = "Martin De Kauwe"
__version__ = "1.0 (09.11.2018)"
__email__   = "mdekauwe@gmail.com"


def main():

    #
    ## Met data ...
    #
    (par, tair, vpd) = get_met_data()
    wind = 2.5
    pressure = 101325.0
    Ca = 400.0

    #
    ## Parameters
    #
    g0 = 0.001
    g1 = 4.0
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
    ### Run Big-lead
    ##

    C = BigLeaf(g0, g1, D0, gamma, Vcmax25, Jmax25, Rd25, Eaj, Eav,
                deltaSj, deltaSv, Hdv, Hdj, Q10, leaf_width, SW_abs,
                gs_model="medlyn")

    An = np.zeros(48)
    gsw = np.zeros(48)
    et = np.zeros(48)

    for i in range(len(par)):

        (An[i], gsw[i], et[i]) = C.main(tair[i], par[i], vpd[i],
                                                  wind, pressure, Ca)


    plt.plot(An * LAI)
    plt.show()


def get_met_data():

    lat = -23.575001
    lon = 152.524994
    sw_rad_day = 20.5 # mj m-2 d-1
    tmin = 2.0
    tmax = 24.0
    doy = 180.0
    lat = 50.0
    rain = 10.0
    vpd09 = 1.4
    vpd09_next = 1.8
    vpd15 = 2.3
    vpd15_prev = 3.4

    hours = np.arange(48) / 2.0

    WG = WeatherGenerator(lat, lon)

    # MJ m-2 d-1 -> J m-2 s-1 = W m-2 -> umol m-2 s-1 -> MJ m-2 d-1 #
    #par_day = sw_rad_day * MJ_TO_J * DAY_2_SEC * SW_2_PAR * \
    #          UMOL_TO_J * J_TO_MJ * SEC_2_DAY
    par_day = sw_rad_day * WG.SW_2_PAR_MJ
    par = WG.estimate_dirunal_par(par_day, doy)

    tair = WG.estimate_diurnal_temp(doy, tmin, tmax)

    vpd = WG.estimate_diurnal_vpd(vpd09, vpd15, vpd09_next, vpd15_prev)

    return (par, tair, vpd)


if __name__ == "__main__":

    main()
