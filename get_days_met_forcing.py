#!/usr/bin/env python
"""
Generate a days timeseries of met forcing to run the model.
"""

import numpy as np
from weather_generator import WeatherGenerator

__author__  = "Martin De Kauwe"
__version__ = "1.0 (09.11.2018)"
__email__   = "mdekauwe@gmail.com"

def get_met_data(lat, lon, doy):

    sw_rad_day = 25.5 # mj m-2 d-1
    tmin = 2.0
    tmax = 24.0
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
