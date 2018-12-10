#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply the two-leaf model to the WTC experiments.

"""
import os
import sys
import numpy as np
import math
import pandas as pd

import constants as c
from radiation import calculate_absorbed_radiation
from two_leaf import CoupledModel as TwoLeaf

__author__  = "Martin De Kauwe"
__version__ = "1.0 (07.12.2018)"
__email__   = "mdekauwe@gmail.com"


def run_treatment(T, df, footprint):

    wind = 5.0
    pressure = 101325.0
    Ca = 400.0

    days = df.doy
    hod = df.hod
    ndays = int(len(days) / 24.)

    dummy = np.ones(ndays) * np.nan
    out = pd.DataFrame({'year':dummy, 'doy':dummy, 'An':dummy, 'E':dummy,
                        'An_obs':dummy, 'E_obs':dummy})

    An_store = np.zeros(ndays)
    E_store = np.zeros(ndays)
    An_store_obs = np.zeros(ndays)
    E_store_obs = np.zeros(ndays)

    an_conv = c.UMOL_TO_MOL * c.MOL_C_TO_GRAMS_C * c.SEC_TO_HR
    et_conv = c.MOL_WATER_2_G_WATER * c.G_TO_KG * c.SEC_TO_HR

    # Add an LAI field, i.e. converting from per tree to m2 m-2
    df = df.assign(LAI = lambda x: x.leafArea/footprint)

    i = 0
    j = 0
    while i < len(df):
        year = df.index.year[i]
        doy = df.doy[i]

        Anx = 0.0
        Ex = 0.0
        An_obs = 0.0
        Ex_obs = 0.0
        hod = 0
        for k in range(24):

            (An, gsw,
             et, tcan,
             _,_,_,_,_,_) = T.main(df.tair[i], df.par[i], df.vpd[i], wind,
                                   pressure, Ca, doy, hod, lat, lon,
                                   df.LAI[i])

            Anx += An * an_conv
            Ex += et * et_conv

            # Convert from per tree to m-2
            An_obs += df.FluxCO2[i] * c.MMOL_2_UMOL * an_conv / footprint
            Ex_obs += df.FluxH2O[i] * et_conv / footprint

            hod += 1
            i += 1

        out.year[i] = year
        out.doy[i] = doy
        out.An[i] = Anx
        out.E[i] = Ex
        out.An_obs[i] = An_obs
        out.E_obs[i] = Ex_obs
        j += 1

    return (out)


if __name__ == "__main__":


    fpath = "/Users/mdekauwe/Downloads/"
    fname = "met_data.csv"
    fn = os.path.join(fpath, fname)
    df = pd.read_csv(fn)
    df = df.drop(df.columns[0], axis=1)
    df.index = pd.to_datetime(df.DateTime)

    #
    ## Parameters - Dushan to set these ...
    #
    lat = -33.617778 # Ellsworth 2017, NCC
    lon = 150.740278
    g0 = 1E-09
    g1 = 3.8
    D0 = 1.5 # kpa # Not used so ignore ...
    Vcmax25 = 81.706
    Jmax25 = Vcmax25 * 1.67
    Rd25 = 2.0  # Need to discuss what you want here, "None" -> Vcmax = 0.015 Rd
    Eaj = 30000.0
    Eav = 60000.0
    deltaSj = 650.0
    deltaSv = 650.0
    Hdv = 200000.0
    Hdj = 200000.0
    Q10 = 2.0
    gamma = 0.0
    leaf_width = 0.02
    SW_abs = 0.8 # use canopy absorptance of solar radiation, not used anyway...

    diameter = 3.25 # chamber
    footprint = np.pi * (diameter / 2.)**2 # to convert from tree to m2

    T = TwoLeaf(g0, g1, D0, gamma, Vcmax25, Jmax25, Rd25, Eaj, Eav, deltaSj,
                deltaSv, Hdv, Hdj, Q10, leaf_width, SW_abs, gs_model="medlyn")

    # Not sure which treatments Dushan wants to run, so will just use this one
    # Easy to edit as I'm passing to a func
    dfx = df[(df.T_treatment == "ambient") &
             (df.Water_treatment == "control") &
             (df.chamber == "C01")]


    (out) = run_treatment(T, dfx, footprint)



    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16,4))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(out.An, label="Model")
    ax1.plot(out.An_obs, label="Observations")
    ax1.set_ylabel("GPP (g C m$^{-2}$ d$^{-1}$)")
    ax1.set_xlabel("Days", position=(1.1, 0.5))
    ax1.legend(numpoints=1, loc="best")

    ax2.plot(out.E, label="Model")
    ax2.plot(out.E_obs, label="Observations")
    ax2.set_ylabel("E (mm d$^{-1}$)")

    ax1.locator_params(nbins=6, axis="y")
    ax2.locator_params(nbins=6, axis="y")

    plt.show()
