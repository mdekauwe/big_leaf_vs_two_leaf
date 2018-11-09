#!/usr/bin/env python
"""
Two-leaf approximation
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin, exp, sqrt, acos, asin
import random
import math

import constants as c
from farq import FarquharC3
from penman_monteith_leaf import PenmanMonteith
from radiation import calculate_solar_geometry, spitters
from radiation import calculate_absorbed_radiation

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


    def main(self, tair, par, vpd, wind, pressure, Ca, doy, hod, lat, lon,
             lai, rnet=None):
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

        An = np.zeros(2) # sunlit, shaded
        gsc = np.zeros(2)  # sunlit, shaded
        Ci = np.zeros(2) # sunlit, shaded
        et = np.zeros(2) # sunlit, shaded

        cos_zenith = calculate_solar_geometry(doy, hod, lat, lon)
        zenith_angle = np.rad2deg(np.arccos(cos_zenith))
        elevation = 90.0 - zenith_angle
        sw_rad = par * c.PAR_2_SW # W m-2

        # get diffuse/beam frac
        (diffuse_frac, direct_frac) = spitters(doy, sw_rad, cos_zenith)

        # Is the sun up?
        if elevation > 0.0 and par > 20.0:

            (apar_leaf,
             lai_leaf, kb) = calculate_absorbed_radiation(par, cos_zenith, lai,
                                                      direct_frac,
                                                      diffuse_frac)

            cscalar = calc_leaf_to_canopy_scalar(lai_leaf, kb)

            # sunlit / shaded loop
            for ileaf in range(2):

                # initialise values of Tleaf, Cs, dleaf at the leaf surface
                dleaf = vpd
                dair = vpd
                Cs = Ca
                Tleaf = tair
                Tleaf_K = Tleaf + c.DEG_2_KELVIN



                iter = 0
                while True:
                    (An[ileaf],
                     gsc[ileaf],
                     Ci[ileaf]) = F.calc_photosynthesis(Cs=Cs, Tleaf=Tleaf_K,
                                                     Par=apar_leaf[ileaf],
                                                     Jmax25=self.Jmax25,
                                                     Vcmax25=self.Vcmax25,
                                                     Q10=self.Q10, Eaj=self.Eaj,
                                                     Eav=self.Eav,
                                                     deltaSj=self.deltaSj,
                                                     deltaSv=self.deltaSv,
                                                     Rd25=self.Rd25,
                                                     Hdv=self.Hdv,
                                                     Hdj=self.Hdj, vpd=dleaf)

                    # Calculate new Tleaf, dleaf, Cs
                    (new_tleaf, et[ileaf],
                     le_et, gbH, gw) = self.calc_leaf_temp(P, Tleaf, tair,
                                                           gsc[ileaf],
                                                           apar_leaf[ileaf],
                                                           vpd, pressure,
                                                           wind, rnet=rnet)

                    gbc = gbH * c.GBH_2_GBC
                    if gbc > 0.0 and An[ileaf] > 0.0:
                        Cs = Ca - An[ileaf] / gbc # boundary layer of leaf
                    else:
                        Cs = Ca

                    if math.isclose(et[ileaf], 0.0) or math.isclose(gw, 0.0):
                        dleaf = dair
                    else:
                        dleaf = (et[ileaf] * pressure / gw) * c.PA_2_KPA # kPa

                    # Check for convergence...?
                    if math.fabs(Tleaf - new_tleaf) < 0.02:
                        break

                    if iter > self.iter_max:
                        raise Exception('No convergence: %d' % (iter))

                    # Update temperature & do another iteration
                    Tleaf = new_tleaf
                    Tleaf_K = Tleaf + c.DEG_2_KELVIN

                    iter += 1

            # scale leaf to canopy
            an_canopy = np.sum(An)
            gsw_canopy = np.sum(gsc) * c.GSC_2_GSW
            et_canopy = np.sum(et)
        else:
            an_canopy = 0.0
            gsw_canopy = 0.0
            et_canopy = 0.0

        return (an_canopy, gsw_canopy, et_canopy)


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
        if math.isclose(gsc, 0.0):
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

def calc_leaf_to_canopy_scalar(lai_leaf, kb):
    """
    Calculate scalar to transform leaf Vcmax and Jmax values to big leaf
    values. Following Wang & Leuning, as long as sunlit and shaded
    leaves are treated seperately, values of parameters in the coupled
    model for the two big leaves can be closely approximated by
    integrating values for individual leaves.
    - Inserting eqn C6 & C7 into B5
    per unit ground area

    Parameters:
    ----------
    canopy_wk : structure
        various canopy values: in this case the sunlit or shaded LAI &
        cos_zenith angle.
    scalar_sun : float
        scalar for sunlit leaves, values returned in unit ground area
        (returned)
    scalar_sha : float
        scalar for shaded leaves, values returned in unit ground area
        (returned)

    References:
    ----------
    * Wang and Leuning (1998) AFm, 91, 89-111; particularly the Appendix.
    """

    cscalar = np.zeros(2)
    # extinction coefficient of nitrogen in the canopy, assumed to be 0.3 by
    # default which comes half Belinda's head and is supported by fig 10 in
    # Lloyd et al. Biogeosciences, 7, 1833–1859, 2010
    kn = 0.3

    lai_sun = lai_leaf[c.SUNLIT]
    lai_sha = lai_leaf[c.SHADED]

    cscalar[c.SUNLIT] = (1.0 - np.exp(-(kb + kn) * lai_sun)) / (kb + kn)
    cscalar[c.SHADED] = (1.0 - np.exp(-kn * lai_sha)) / kn - cscalar[c.SUNLIT]

    return (cscalar)

if __name__ == "__main__":

    from get_days_met_forcing import get_met_data

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
    ### Run Big-leaf
    ##

    C = CoupledModel(g0, g1, D0, gamma, Vcmax25, Jmax25, Rd25, Eaj, Eav,
                     deltaSj, deltaSv, Hdv, Hdj, Q10, leaf_width, SW_abs,
                     gs_model="medlyn")

    An_tl = np.zeros(48)
    gsw_tl = np.zeros(48)
    et_tl = np.zeros(48)

    hod = 0
    for i in range(len(par)):

        (An_tl[i], gsw_tl[i], et_tl[i]) = C.main(tair[i], par[i], vpd[i],
                                                 wind, pressure, Ca, doy, hod,
                                                 lat, lon, LAI)

        hod += 1

    fig = plt.figure(figsize=(14,5))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.3)
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

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(An_tl)
    ax1.set_ylabel("$A_{\mathrm{n}}$ ($\mathrm{\mu}$mol m$^{-2}$ s$^{-1}$)")
    ax1.set_xlabel("Hour of day")

    ax2.plot(et_tl * c.MOL_TO_MMOL, label="Big leaf")
    ax2.set_ylabel("E (mmol m$^{-2}$ s$^{-1}$)")
    ax2.set_xlabel("Hour of day")

    plt.show()
