#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Various radiation funcs needed for the two-leaf approximation
"""

import sys
import numpy as np
import math
import constants as c

__author__  = "Martin De Kauwe"
__version__ = "1.0 (09.11.2018)"
__email__   = "mdekauwe@gmail.com"


def spitters(doy, sw_rad, cos_zenith):

    """
    Spitters algorithm to estimate the diffuse component from the measured
    irradiance.

    Eqn 20a-d.

    Parameters:
    ----------
    doy : int
        day of year
    sw_rad : double
        total incident radiation [J m-2 s-1]

    Returns:
    -------
    diffuse : double
        diffuse component of incoming radiation (returned in cw structure)

    References:
    ----------
    * Spitters, C. J. T., Toussaint, H. A. J. M. and Goudriaan, J. (1986)
      Separating the diffuse and direct component of global radiation and
      its implications for modeling canopy photosynthesis. Part I.
      Components of incoming radiation. Agricultural Forest Meteorol.,
      38:217-229.
    """

    solcon = 1370.0

    fbeam = 0.0
    tmpr = 0.847 + cos_zenith * (1.04 * cos_zenith - 1.61)
    tmpk = (1.47 - tmpr) / 1.66

    if cos_zenith > 1.0e-10 and sw_rad > 10.0:
        tmprat = sw_rad / (solcon * (1.0 + 0.033 * \
                    np.cos(2. * np.pi * (doy-10.0) / 365.0)) * cos_zenith)
    else:
        tmprat = 0.0

    if tmprat > 0.22:
        fbeam = 6.4 * ( tmprat - 0.22 )**2

    if tmprat > 0.35:
        fbeam = min( 1.66 * tmprat - 0.4728, 1.0 )

    if tmprat > tmpk:
        fbeam = max( 1.0 - tmpr, 0.0 )

    if cos_zenith < 1.0e-2:
        fbeam = 0.0

    diffuse_frac = 1.0 - fbeam

    return (diffuse_frac, fbeam)

def calculate_absorbed_radiation(par, cos_zenith, lai, direct_frac,
                                 diffuse_frac, doy, sw_rad, tair, refl, tau):
    """
    Calculate absorded irradiance of sunlit and shaded fractions of
    the canopy.

    This logic matches CABLE

    References:
    -----------
    * Wang and Leuning (1998) AFm, 91, 89-111. B3b and B4, the answer is
                              identical de P & F
    * De Pury & Farquhar (1997) PCE, 20, 537-557.
    * Dai et al. (2004) Journal of Climate, 17, 2281-2299.
    """

    qcan = np.zeros((2,3))
    apar = np.zeros(2)
    lai_leaf = np.zeros(2)
    cexpkdm = np.zeros(2)
    kdm = np.zeros(2)
    kbx = np.zeros(3)
    gauss_w = np.array([0.308, 0.514, 0.178]) # Gaussian integ. weights
    cos3 = np.zeros(3)
    c1 = np.zeros(3)
    rho_td = np.zeros(2)
    kbm = np.zeros(2)
    rhocbm = np.zeros(2)
    rho_tb = np.zeros(2)
    cexpkbm = np.zeros(2)
    rhocdf = np.zeros(3)
    albsoilsn = np.zeros(2)

    tk = tair + c.DEG_2_KELVIN

    # surface temperaute - just using air temp
    tsurf = tk

    # Estimate LWdown based on an emprical function of air temperature (K)
    # following Swinbank, W. C. (1963): Long-wave radiation from clear skies,
    # Q. J. R. Meteorol. Soc., 89, 339–348, doi:10.1002/qj.49708938105.
    lw_down = 0.0000094 * c.SIGMA * tk**6.0

    # black-body long-wave radiation
    flpwb = c.SIGMA * tk**4

    # soil emissivity
    emissivity_soil = 1.0

    # leaf emissivity # emissivity of leaf (-), Table 3, Wang and Leuning, 1998
    emissivity_leaf = 0.96

    # air emissivity
    emissivity_air = lw_down / flpwb

    # vegetation long-wave radiation (isothermal)
    flwv = emissivity_leaf * flpwb

    # soil long-wave radiation
    flws = c.SIGMA * emissivity_soil * tsurf**4

    # cos(15 45 75 degrees)
    cos3[0] = np.cos(np.deg2rad(15.0))
    cos3[1] = np.cos(np.deg2rad(45.0))
    cos3[2] = np.cos(np.deg2rad(75.0))

    # empirical param related to the leaf angle dist (= 0 for spherical LAD)
    chi = 9.99999978E-03

    # leaf angle parmameter 1
    xphi1 = 0.5 - chi * (0.633 + 0.33 * chi)

    # leaf angle parmameter 2
    xphi2 = 0.877 * (1.0 - 2.0 * xphi1)

    # Ross-Goudriaan function is the ratio of the projected area of leaves
    # in the direction perpendicular to the direction of incident solar
    # radiation and the actual leaf area. Approximated as eqn 28,
    # Kowalcyk et al. 2006)
    Gross = xphi1 + xphi2 * cos_zenith

    # extinction coefficient of direct beam radiation for a canopy with black
    # leaves, eq 26 Kowalcyk et al. 2006
    if lai > c.LAI_THRESH and direct_frac > c.RAD_THRESH:   # vegetated
        kb = Gross / cos_zenith
    else:   # i.e. bare soil
        kb = 0.5

    # extinction coefficient of diffuse radiation for a canopy with black
    # leaves, eq 27 Kowalcyk et al. 2006
    if lai > c.LAI_THRESH:  # vegetated

        # Approximate integration of kb
        kbx[0] = (xphi1 + xphi2 * cos3[0]) / cos3[0]
        kbx[1] = (xphi1 + xphi2 * cos3[1]) / cos3[1]
        kbx[2] = (xphi1 + xphi2 * cos3[2]) / cos3[2]

        kd = -np.log(np.sum(gauss_w * np.exp(-kbx * lai))) / lai
    else:   # i.e. bare soil
        kd = 0.7

    if np.abs(kb - kd) < c.RAD_THRESH:
        kb = kd + c.RAD_THRESH

    if direct_frac < c.RAD_THRESH:
        kb = 1.e5

    c1[0] = np.sqrt(1. - tau[0] - refl[0])
    c1[1] = np.sqrt(1. - tau[1] - refl[1])
    c1[2] = 1.0

    # Canopy reflection black horiz leaves
    # (eq. 6.19 in Goudriaan and van Laar, 1994):
    rhoch = (1.0 - c1) / (1.0 + c1)

    # extinction coefficient of nitrogen in the canopy, assumed to be 0.3 by
    # default which comes half Belinda's head and is supported by fig 10 in
    # Lloyd et al. Biogeosciences, 7, 1833–1859, 2010
    #kn = 0.3
    kn = 0.001

    # Relative leaf nitrogen concentration within canopy:
    cf2n = np.exp(-kn * lai)

    # fraction SW beam tranmitted through canopy
    transb = np.exp(-kb * lai)

    if lai > c.LAI_THRESH: # where vegetated
        # Diffuse SW transmission fraction ("black" leaves, extinction neglects
        # leaf SW transmittance and REFLectance);
        # from Monsi & Saeki 1953, quoted in eq. 18 of Sellers 1985:
        transd = np.exp(-kd * lai)
    else:
        transd = 1.0

    for b in range(3): #0 = visible; 1 = nir radiation; 2 = LW

        # Canopy reflection of diffuse radiation for black leaves:
        rhocdf[b] = rhoch[b] * 2. * \
                        (gauss_w[0] * kbx[0] / (kbx[0] + kd) + \
                         gauss_w[1] * kbx[1] / (kbx[1] + kd) + \
                         gauss_w[2] * kbx[2] / (kbx[2] + kd))

    # Calculate albedo
    soil_reflectance = 0.0665834472
    if soil_reflectance <= 0.14:
        sfact = 0.5
    elif soil_reflectance > 0.14 and soil_reflectance <= 0.20:
        sfact = 0.62
    else:
        sfact = 0.68

    # soil + snow reflectance (ignoring snow)
    albsoilsn[c.SHADED] = 2.0 * soil_reflectance / (1. + sfact)
    albsoilsn[c.SUNLIT] = sfact * albsoilsn[c.SHADED]

    # Update extinction coefficients and fractional transmittance for
    # leaf transmittance and reflection (ie. NOT black leaves):
    for b in range(2): #0 = visible; 1 = nir radiation

        # modified k diffuse(6.20)(for leaf scattering)
        kdm[b] = kd * c1[b]

        # Define canopy diffuse transmittance (fraction):
        cexpkdm[b] = np.exp(-kdm[b] * lai)

        # Calculate effective canopy-soiil diffuse reflectance (fraction)
        if lai > 0.001:
            rho_td[b] = rhocdf[b] + (albsoilsn[b] - rhocdf[b]) * cexpkdm[b]**2
        else:
            rho_td[b] = albsoilsn[b]

        # where vegetated and sunlit
        if lai > c.LAI_THRESH and np.sum(sw_rad) > c.RAD_THRESH:
            kbm[b] = kb * c1[b]
        else:
            kbm[b] = 1.e-9

        # Canopy reflection (6.21) beam:
        rhocbm[b] = 2. * kb / (kb + kd) * rhoch[b]

        # Canopy beam transmittance (fraction):
        cexpkbm[b] = np.exp(-min(kbm[b] * lai, 20.))

        # Calculate effective canopy-soil beam reflectance (fraction):
        rho_tb[b] = rhocbm[b] + (albsoilsn[b] - rhocbm[b]) * cexpkbm[b]**2
    else:
        cexpkbm[b] = 0.0
        rhocbm[b]  = 0.0
        rho_tb[b] = albsoilsn[b]

    # Longwave radiation absorbed by sunlit canopy fraction:
    qcan[c.SUNLIT,c.LW] = (flws - flwv) * kd * \
                            (transd - transb) / (kb - kd) + \
                            (emissivity_air - emissivity_leaf) * kd * \
                            flpwb * (1.0 - transd * transb) / (kb + kd)

    # Longwave radiation absorbed by shaded canopy fraction:
    qcan[c.SHADED,c.LW] = (1.0 - transd) * (flws + lw_down - 2.0 * flwv) - \
                            qcan[0,2]

    for b in range(2): #0 = visible; 1 = nir radiation

        if lai > c.LAI_THRESH and np.sum(sw_rad) > c.RAD_THRESH:

            cf1 = diffuse_frac * (1.0 - rho_td[b]) * kdm[b]
            cf2 = (1.0 - transb * cexpkdm[b]) / (kb + kdm[b])
            cf3 = (1.0 - transb * cexpkbm[b]) / (kb + kbm[b])
            cf4 = (1.0 - tau[b] - refl[b]) * kb
            cf5 = (1.0 - transb) / kb - (1.0 - transb**2) / (kb + kb)

            qcan[c.SUNLIT,b] = sw_rad[b] * (cf1 * cf2 + \
                                direct_frac * (1.0 - rho_tb[b]) * kbm[b] * \
                                cf3 + direct_frac * cf4 * cf5)

            qcan[c.SHADED,b] = sw_rad[b] * (cf1 * \
                                ((1.0 - cexpkdm[b]) / kdm[b] - cf2) + \
                                direct_frac * (1. - rho_tb[b]) * kbm[b] * \
                                ((1.0 - cexpkbm[b]) / kbm[b] - cf3) - \
                                direct_frac * cf4 * cf5)

            #print(qcan[c.SUNLIT,b])

            #qcan[c.SUNLIT,b] = sw_rad[b] * \
            #                    ( (diffuse_frac * (1.0 - rho_td[b])) *\
            #                     kdm[b] * cf2 + \
            #                     direct_frac * (1.0 - rho_tb[b]) * \
            #                     kbm[b] * cf3 +\
            #                     direct_frac * (1.0 - tau[b] - refl[b]) * kb *\
            #                     ( (1.0 - transb) / kb - (1.0 - transb**2) / (kb + kb)))

            #print(qcan[c.SUNLIT,b])

            #Ib = sw_rad[b] * direct_frac
            #Id = sw_rad[b] * diffuse_frac

            # B3b in Wang and Leuning 1998
            #a1 = Id * (1.0 - rho_td[b]) * kdm[b]
            #a2 = psi_func(kdm[b] + kb, lai)
            #a3 = Ib * (1.0 - rho_tb[b]) * kbm[b]
            #a4 = psi_func(kbm[b] + kb, lai)
            #a5 = Ib * (1.0 - tau[b] - refl[b]) * kb
            #a6 = psi_func(kb, lai) - psi_func(2.0 * kb, lai)
            #qcan[c.SUNLIT,b] = a1 * a2 + a3 * a4 + a5 * a6

            # B4 in Wang and Leuning 1998
            #a2 = psi_func(kdm[b], lai) - psi_func(kdm[b] + kb, lai)
            #a4 = psi_func(kbm[b], lai) - psi_func(kbm[b] + kb, lai)
            #qcan[c.SHADED,b] = a1 * a2 + a3 * a4 - a5 * a6

            #print(qcan[c.SUNLIT,b])
            #print("\n")


    apar[c.SUNLIT] = qcan[c.SUNLIT,c.VIS] * c.J_TO_UMOL
    apar[c.SHADED] = qcan[c.SHADED,c.VIS] * c.J_TO_UMOL

    # Total energy absorbed by canopy, summing VIS, NIR and LW components, to
    # leave us with the indivual leaf components.
    qcan = qcan.sum(axis=1)

    # where vegetated and sunlit
    if lai > c.LAI_THRESH and np.sum(sw_rad) > c.RAD_THRESH:
        lai_leaf[c.SUNLIT] = (1.0 - transb) / kb
    else:
        lai_leaf[c.SUNLIT] = 0.0

    lai_leaf[c.SHADED] = lai - lai_leaf[c.SUNLIT]

    return (qcan, apar, lai_leaf, kb, kn, cf2n, transb)

def psi_func(z, lai):
    # B5 function from Wang and Leuning which integrates property passed via
    # arg list over the canopy space
    #
    # References:
    # -----------
    # * Wang and Leuning (1998) AFm, 91, 89-111. Page 106

    return ( (1.0 - np.exp(-z * lai)) / z )


def calculate_cos_zenith(doy, xslat, hod):

    # calculate sin(bet), bet = elevation angle of sun
    # calculations according to goudriaan & van laar 1994 p30

    # sine of maximum declination
    sindec = -np.sin(23.45 * np.pi / 180.) * \
                np.cos(2. * np.pi * (doy + 10.0) / 365.0)

    z = max(np.sin(np.pi / 180. * xslat) * sindec + \
            np.cos(np.pi / 180. * xslat) * np.sqrt(1. - sindec * sindec) * \
            np.cos(np.pi * (hod - 12.0) / 12.0), 1e-8)

    return z

def calc_leaf_to_canopy_scalar(lai, kb, kn, cf2n, transb):
    """
    Calculate scalar to transform beam/diffuse leaf Vcmax, Jmax and Rd values
    to big leaf values.

    - Insert eqn C6 & C7 into B5

    Parameters:
    ----------
    lai : float
        leaf area index
    kb : float
        beam extinction coefficient for black leaves

    References:
    ----------
    * Wang and Leuning (1998) AFm, 91, 89-111; particularly the Appendix.
    """
    scalex = np.zeros(2)

    # Taken from CABLE, will need to pass tings to use
    scalex[c.SUNLIT] = (1.0 - transb * cf2n) / (kb + kn)
    scalex[c.SHADED] = (1.0 - cf2n) / kn - scalex[c.SUNLIT]

    return scalex
