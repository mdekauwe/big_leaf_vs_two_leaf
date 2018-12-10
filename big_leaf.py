#!/usr/bin/env python
"""
Solve 30-minute coupled A-gs(E) using a big-leaf approximation, i.e. simply
multiplying leaf-level fluxes by LAI.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, exp, sqrt, acos, asin
import random
import math
import numpy as np

import constants as c
import parameters as p
from farq import FarquharC3
from penman_monteith_leaf import PenmanMonteith
from radiation import spitters
from radiation import calculate_absorbed_radiation
from radiation import calculate_cos_zenith, calc_leaf_to_canopy_scalar
from utils import calc_esat

__author__  = "Martin De Kauwe"
__version__ = "1.0 (09.11.2018)"
__email__   = "mdekauwe@gmail.com"


class CoupledModel(object):
    """Iteratively solve leaf temp, Ci, gs and An."""

    def __init__(self, g0, g1, D0, gamma, Vcmax25, Jmax25, Rd25, Eaj, Eav,
                 deltaSj, deltaSv, Hdv, Hdj, Q10, leaf_width, SW_abs,
                 gs_model, alpha=None, iter_max=100):

        # set params
        self.g0 = g0
        self.g1 = g1
        self.D0 = D0
        self.gamma = gamma
        self.Vcmax25 = Vcmax25
        self.Jmax25 = Jmax25
        self.Rd25 = Rd25
        self.Eaj = Eaj
        self.Eav = Eav
        self.deltaSj = deltaSj
        self.deltaSv = deltaSv
        self.Hdv = Hdv
        self.Hdj = Hdj
        self.Q10 = Q10
        self.leaf_width = leaf_width
        self.alpha = alpha
        self.SW_abs = SW_abs # leaf abs of solar rad [0,1]
        self.gs_model = gs_model
        self.iter_max = iter_max

        self.emissivity_leaf = 0.99   # emissivity of leaf (-)
        self.k = 0.5 # light extinction coefficient

    def main(self, tair, par, vpd, wind, pressure, Ca, doy, hod, lat, lon,
             LAI, rnet=None):
        """
        Parameters:
        ----------
        tair : float
            air temperature (deg C)
        par : float
            Photosynthetically active radiation (umol m-2 s-1)
        vpd : float
            Vapour pressure deficit (kPa, needs to be in Pa, see conversion
            below)
        wind : float
            wind speed (m s-1)
        pressure : float
            air pressure (using constant) (Pa)
        Ca : float
            ambient CO2 concentration
        doy : float
            day of day
        hod : float
            hour of day
        lat : float
            latitude
        lon : float
            longitude
        lai : floar
            leaf area index

        Returns:
        --------
        An : float
            net leaf assimilation (umol m-2 s-1)
        gs : float
            stomatal conductance (mol m-2 s-1)
        et : float
            transpiration (mol H2O m-2 s-1)
        """

        F = FarquharC3(theta_J=0.85, peaked_Jmax=True, peaked_Vcmax=True,
                       model_Q10=True, gs_model=self.gs_model,
                       gamma=self.gamma, g0=self.g0,
                       g1=self.g1, D0=self.D0, alpha=self.alpha)
        P = PenmanMonteith(self.leaf_width, self.SW_abs)

        # set initial values
        dleaf = vpd
        Cs = Ca
        Tleaf = tair
        Tleaf_K = Tleaf + c.DEG_2_KELVIN

        cos_zenith = calculate_cos_zenith(doy, lat, hod)
        zenith_angle = np.rad2deg(np.arccos(cos_zenith))
        elevation = 90.0 - zenith_angle

        # Is the sun up?
        if elevation > 0.0 and par > 50.0:

            iter = 0
            while True:
                (An,
                 gsc) = F.photosynthesis(Cs=Cs, Tleaf=Tleaf_K, Par=par,
                                         Jmax25=self.Jmax25,
                                         Vcmax25=self.Vcmax25, Q10=self.Q10,
                                         Eaj=self.Eaj, Eav=self.Eav,
                                         deltaSj=self.deltaSj,
                                         deltaSv=self.deltaSv,
                                         Rd25=self.Rd25, Hdv=self.Hdv,
                                         Hdj=self.Hdj, vpd=dleaf)

                # Scale leaf to canopy fluxes, assuming that the photosynthetic
                # capacity is assumed to decline exponentially through the
                # canopy, in proportion to the incident radiation estimated by
                # Beer’s Law
                An *= (1.0 - np.exp(-self.k * LAI)) / self.k
                gsc *= (1.0 - np.exp(-self.k * LAI)) / self.k

                # Calculate new Tleaf, dleaf, Cs
                (new_tleaf, et,
                 le_et, gbH, gw) = self.calc_leaf_temp(P, Tleaf, tair, gsc,
                                                       par, vpd, pressure, wind,
                                                       rnet=rnet)

                gbc = gbH * c.GBH_2_GBC
                if gbc > 0.0 and An > 0.0:
                    Cs = Ca - An / gbc # boundary layer of leaf
                else:
                    Cs = Ca

                if np.isclose(et, 0.0) or np.isclose(gw, 0.0):
                    dleaf = vpd
                else:
                    dleaf = (et * pressure / gw) * c.PA_2_KPA # kPa

                # Check for convergence...?
                if math.fabs(Tleaf - new_tleaf) < 0.02:
                    break

                if iter > self.iter_max:
                    #raise Exception('No convergence: %d' % (iter))
                    An = 0.0
                    gsc = 0.0
                    et = 0.0
                    break

                # Update temperature & do another iteration
                Tleaf = new_tleaf
                Tleaf_K = Tleaf + c.DEG_2_KELVIN

                iter += 1

            an_canopy = An
            gsw_canopy = gsc * c.GSC_2_GSW
            et_canopy = et
            tcanopy = Tleaf
        else:
            an_canopy = 0.0
            gsw_canopy = 0.0
            et_canopy = 0.0
            tcanopy = tair

        return (an_canopy, gsw_canopy, et_canopy, tcanopy)

    def calc_leaf_temp(self, P=None, tleaf=None, tair=None, gsc=None, par=None,
                       vpd=None, pressure=None, wind=None, rnet=None):
        """
        Resolve leaf temp

        Parameters:
        ----------
        P : object
            Penman-Montheith class instance
        tleaf : float
            leaf temperature (deg C)
        tair : float
            air temperature (deg C)
        gs : float
            stomatal conductance (mol m-2 s-1)
        par : float
            Photosynthetically active radiation (umol m-2 s-1)
        vpd : float
            Vapour pressure deficit (kPa, needs to be in Pa, see conversion
            below)
        pressure : float
            air pressure (using constant) (Pa)
        wind : float
            wind speed (m s-1)

        Returns:
        --------
        new_Tleaf : float
            new leaf temperature (deg C)
        et : float
            transpiration (mol H2O m-2 s-1)
        gbH : float
            total boundary layer conductance to heat for one side of the leaf
        gw : float
            total leaf conductance to water vapour (mol m-2 s-1)
        """
        tleaf_k = tleaf + c.DEG_2_KELVIN
        tair_k = tair + c.DEG_2_KELVIN

        air_density = pressure / (c.RSPECIFC_DRY_AIR * tair_k)

        # convert from mm s-1 to mol m-2 s-1
        cmolar = pressure / (c.RGAS * tair_k)

        # W m-2 = J m-2 s-1
        if rnet is None:
            rnet = P.calc_rnet(par, tair, tair_k, tleaf_k, vpd, pressure)

        (grn, gh, gbH, gw) = P.calc_conductances(tair_k, tleaf, tair,
                                                 wind, gsc, cmolar)
        if np.isclose(gsc, 0.0):
            et = 0.0
            le_et = 0.0
        else:
            (et, le_et) = P.calc_et(tleaf, tair, vpd, pressure, wind, par,
                                    gh, gw, rnet)

        # D6 in Leuning. NB I'm doubling conductances, see note below E5.
        # Leuning isn't explicit about grn but I think this is right
        # NB the units or grn and gbH are mol m-2 s-1 and not m s-1, but it
        # cancels.
        Y = 1.0 / (1.0 + (2.0 * grn) / (2.0 * gbH))

        # sensible heat exchanged between leaf and surroundings
        H = Y * (rnet - le_et)

        # leaf-air temperature difference recalculated from energy balance.
        # NB. I'm using gh here to include grn and the doubling of conductances
        new_Tleaf = tair + H / (c.CP * air_density * (gh / cmolar))

        return (new_Tleaf, et, le_et, gbH, gw)


if __name__ == "__main__":

    from get_days_met_forcing import get_met_data

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

    C = CoupledModel(p.g0, p.g1, p.D0, p.gamma, p.Vcmax25, p.Jmax25, p.Rd25,
                     p.Eaj, p.Eav, p.deltaSj, p.deltaSv, p.Hdv, p.Hdj, p.Q10,
                     p.leaf_width, p.SW_abs, gs_model="medlyn")

    An_bl = np.zeros(48)
    gsw_bl = np.zeros(48)
    et_bl = np.zeros(48)
    tcan_bl = np.zeros(48)

    hod = 0
    for i in range(len(par)):

        (An_bl[i], gsw_bl[i],
         et_bl[i], tcan_bl[i]) = C.main(tair[i], par[i], vpd[i],
                                        p.wind, p.pressure, p.Ca, doy, hod,
                                        p.lat, p.lon, p.LAI)

        hod += 1

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

    ax1.plot(np.arange(48)/2, An_bl)
    ax1.set_ylabel("$A_{\mathrm{n}}$ ($\mathrm{\mu}$mol m$^{-2}$ s$^{-1}$)")

    ax2.plot(np.arange(48)/2, et_bl * c.MOL_TO_MMOL, label="Big leaf")
    ax2.set_ylabel("E (mmol m$^{-2}$ s$^{-1}$)")
    ax2.set_xlabel("Hour of day")

    ax3.plot(np.arange(48)/2., tair, label="Tair")
    ax3.plot(np.arange(48)/2., tcan_bl, label="Tcanopy")
    ax3.set_ylabel("Temperature (deg C)")
    ax3.legend(numpoints=1, loc="best")

    ax1.locator_params(nbins=6, axis="y")
    ax2.locator_params(nbins=6, axis="y")

    plt.show()
