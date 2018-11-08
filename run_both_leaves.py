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

__author__  = "Martin De Kauwe"
__version__ = "1.0 (09.11.2018)"
__email__   = "mdekauwe@gmail.com"


def main():

    (par, tair, vpd) = get_met_data()

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

    plt.plot(hours, vpd, "r-")

    plt.show()

    return (par, tair, vpd)


if __name__ == "__main__":

    main()
