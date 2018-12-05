#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Solve 30-minute coupled A-gs(E) using a two-leaf approximation roughly following
Wang and Leuning.

References:
----------
* Wang & Leuning (1998) Agricultural & Forest Meterorology, 91, 89-111.
* Dai et al. (2004) Journal of Climate, 17, 2281-2299.
* De Pury & Farquhar (1997) PCE, 20, 537-557.

"""
import os
import sys
import numpy as np
from math import pi, cos, sin, exp, sqrt, acos, asin
import random
import math
import pandas as pd

import constants as c
from farq import FarquharC3
from penman_monteith_leaf import PenmanMonteith
from radiation import calculate_solar_geometry, spitters
from radiation import calculate_absorbed_radiation
from big_leaf import CoupledModel as BigLeaf
from two_leaf import CoupledModel as TwoLeaf

__author__  = "Martin De Kauwe"
__version__ = "1.0 (09.11.2018)"
__email__   = "mdekauwe@gmail.com"


lat = 61.8474
lon = 24.2948




fpath = "/Users/mdekauwe/Downloads/"
fname = "FI-Hyy_met_and_plant_data_drought_2003.csv"
fn = os.path.join(fpath, fname)
df = pd.read_csv(fn, skiprows=range(1,2))

par = df.PPFD
tair = df.Tair
vpd = df.VPD
wind = df.u
pressure = 101325.0
Ca = 400.0
LAI = df.LAI
days = df.doy

#
## Parameters
#
g0 = 1E-09
g1 = df.g1[0]
D0 = 1.5 # kpa
Vcmax25 = df.Vmax25[0]
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

# Cambell & Norman, 11.5, pg 178
# The solar absorptivities of leaves (-0.5) from Table 11.4 (Gates, 1980)
# with canopies (~0.8) from Table 11.2 reveals a surprising difference.
# The higher absorptivityof canopies arises because of multiple reflections
# among leaves in a canopy and depends on the architecture of the canopy.
SW_abs = 0.8 # use canopy absorptance of solar radiation

##
### Run 2-leaf
##

T = TwoLeaf(g0, g1, D0, gamma, Vcmax25, Jmax25, Rd25, Eaj, Eav,
                 deltaSj, deltaSv, Hdv, Hdj, Q10, leaf_width, SW_abs,
                 gs_model="medlyn")

B = BigLeaf(g0, g1, D0, gamma, Vcmax25, Jmax25, Rd25, Eaj, Eav,
            deltaSj, deltaSv, Hdv, Hdj, Q10, leaf_width, SW_abs,
            gs_model="medlyn")

An_store = np.zeros(365)
E_store = np.zeros(365)
AnB_store = np.zeros(365)
EB_store = np.zeros(365)
gpp_obs = np.zeros(365)
lai_obs = np.zeros(365)

et_conv = c.MOL_WATER_2_G_WATER * c.G_TO_KG * 1800.
an_conv = c.UMOL_TO_MOL * c.MOL_C_TO_GRAMS_C * 1800.

cnt = 0
for doy in range(364):
    print(doy)
    hod = 0

    Aobsx = 0.0
    Anx = 0.0
    Ex = 0.0
    Anxb = 0.0
    Exb = 0.0
    Lobsx = 0.0
    for i in range(48):

        (An, gsw,
         et, tcan,
         _,_,_,_,_,_) = T.main(tair[cnt], par[cnt], vpd[cnt],
                                        wind[cnt], pressure, Ca, doy, hod,
                                        lat, lon, LAI[cnt])

        (Anb, gswb,
         etb, tcanb) = B.main(tair[cnt], par[cnt], vpd[cnt],
                                        wind[cnt], pressure, Ca, doy, hod,
                                        lat, lon, LAI[cnt])



        Anx += An * an_conv
        Ex += et * et_conv
        Anxb += Anb * an_conv
        Exb += etb * et_conv
        Aobsx += df.GPP[cnt] * an_conv
        Lobsx += df.LAI[cnt]

        hod += 1
        cnt += 1

    An_store[doy] = Anx
    E_store[doy] = Ex
    AnB_store[doy] = Anxb
    EB_store[doy] = Exb
    gpp_obs[doy] = Aobsx
    lai_obs[doy] = Lobsx / 48


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


ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.plot(gpp_obs, label="Obs")
ax1.plot(AnB_store, label="Big-leaf")
ax1.plot(An_store, label="2-leaf")
ax1.set_ylabel("GPP (g C m$^{-2}$ d$^{-1}$)")
ax1.legend(numpoints=1, loc="best")

ax2.plot(E_store, label="Big leaf")
ax2.plot(EB_store, label="Big leaf")
ax2.set_ylabel("E (mm d$^{-1}$)")
ax2.set_xlabel("Day of year")

ax3.plot(lai_obs)
ax3.set_ylabel("LAI (m$^{2}$ m$^{-2}$)")


ax1.locator_params(nbins=6, axis="y")
ax2.locator_params(nbins=6, axis="y")

plt.show()
