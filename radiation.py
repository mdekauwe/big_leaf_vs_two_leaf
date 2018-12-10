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

    solar_constant = 1370.0 # W m–2

    direct_frac = 0.0
    tmpr = 0.847 + cos_zenith * (1.04 * cos_zenith - 1.61)
    tmpk = (1.47 - tmpr) / 1.66

    if cos_zenith > 1.0e-10 and sw_rad > 10.0:
        tmprat = sw_rad / (solar_constant * (1.0 + 0.033 * \
                    np.cos(2. * np.pi * (doy-10.0) / 365.0)) * cos_zenith)
    else:
        tmprat = 0.0

    if tmprat > 0.22:
        direct_frac = 6.4 * ( tmprat - 0.22 )**2

    if tmprat > 0.35:
        direct_frac = min( 1.66 * tmprat - 0.4728, 1.0 )

    if tmprat > tmpk:
        direct_frac = max( 1.0 - tmpr, 0.0 )

    if cos_zenith < 1.0e-2:
        direct_frac = 0.0

    diffuse_frac = 1.0 - direct_frac

    return (diffuse_frac, direct_frac)

def calculate_absorbed_radiation(p, par, cos_zenith, lai, direct_frac,
                                 diffuse_frac, doy, sw_rad, tair):
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
    cexpk_dash_d = np.zeros(2)
    k_dash_d = np.zeros(2)
    kbx = np.zeros(3)
    gauss_w = np.array([0.308, 0.514, 0.178]) # Gaussian integ. weights
    cos3 = np.zeros(3)
    c1 = np.zeros(3)
    rho_td = np.zeros(2)
    k_dash_b = np.zeros(2)
    rhocbm = np.zeros(2)
    rho_tb = np.zeros(2)
    cexpk_dash_b = np.zeros(2)
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

    # air emissivity
    emissivity_air = lw_down / flpwb

    # vegetation long-wave radiation (isothermal)
    flwv = p.emissivity_leaf * flpwb

    # soil long-wave radiation
    flws = c.SIGMA * p.emissivity_soil * tsurf**4

    # cos(15 45 75 degrees)
    cos3[0] = np.cos(np.deg2rad(15.0))
    cos3[1] = np.cos(np.deg2rad(45.0))
    cos3[2] = np.cos(np.deg2rad(75.0))

    # leaf angle parmameter 1
    xphi1 = 0.5 - p.chi * (0.633 + 0.33 * p.chi)

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

    c1[0] = np.sqrt(1. - p.tau[0] - p.refl[0])
    c1[1] = np.sqrt(1. - p.tau[1] - p.refl[1])
    c1[2] = 1.0

    # Canopy reflection black horiz leaves
    # (eq. 6.19 in Goudriaan and van Laar, 1994):
    rhoch = (1.0 - c1) / (1.0 + c1)

    for b in range(3): #0 = visible; 1 = nir radiation; 2 = LW

        # Canopy reflection of diffuse radiation for black leaves:
        rhocdf[b] = rhoch[b] * 2. * \
                        (gauss_w[0] * kbx[0] / (kbx[0] + kd) + \
                         gauss_w[1] * kbx[1] / (kbx[1] + kd) + \
                         gauss_w[2] * kbx[2] / (kbx[2] + kd))

    # Calculate albedo
    if p.soil_reflectance <= 0.14:
        sfact = 0.5
    elif p.soil_reflectance > 0.14 and p.soil_reflectance <= 0.20:
        sfact = 0.62
    else:
        sfact = 0.68

    # soil + snow reflectance (ignoring snow)
    albsoilsn[c.SHADED] = 2.0 * p.soil_reflectance / (1. + sfact)
    albsoilsn[c.SUNLIT] = sfact * albsoilsn[c.SHADED]

    # Update extinction coefficients and fractional transmittance for
    # leaf transmittance and reflection (ie. NOT black leaves):
    for b in range(2): #0 = visible; 1 = nir radiation

        # modified k diffuse(6.20)(for leaf scattering)
        k_dash_d[b] = kd * c1[b]

        # Define canopy diffuse transmittance (fraction):
        cexpk_dash_d[b] = np.exp(-k_dash_d[b] * lai)

        # Calculate effective canopy-soiil diffuse reflectance (fraction)
        if lai > 0.001:
            rho_td[b] = rhocdf[b] + (albsoilsn[b] - rhocdf[b]) * \
                            cexpk_dash_d[b]**2
        else:
            rho_td[b] = albsoilsn[b]

        # where vegetated and sunlit
        if lai > c.LAI_THRESH and np.sum(sw_rad) > c.RAD_THRESH:
            k_dash_b[b] = kb * c1[b]
        else:
            k_dash_b[b] = 1.e-9

        # Canopy reflection (6.21) beam:
        rhocbm[b] = 2. * kb / (kb + kd) * rhoch[b]

        # Canopy beam transmittance (fraction):
        cexpk_dash_b[b] = np.exp(-min(k_dash_b[b] * lai, 30.))

        # Calculate effective canopy-soil beam reflectance (fraction):
        rho_tb[b] = rhocbm[b] + (albsoilsn[b] - rhocbm[b]) * cexpk_dash_b[b]**2

        if lai > c.LAI_THRESH and np.sum(sw_rad) > c.RAD_THRESH:

            Ib = sw_rad[b] * direct_frac
            Id = sw_rad[b] * diffuse_frac

            # Radiation absorbed by the sunlit leaf, B3b Wang and Leuning 1998
            a1 = Id * (1.0 - rho_td[b]) * k_dash_d[b]
            a2 = psi_func(k_dash_d[b] + kb, lai)
            a3 = Ib * (1.0 - rho_tb[b]) * k_dash_b[b]
            a4 = psi_func(k_dash_b[b] + kb, lai)
            a5 = Ib * (1.0 - p.tau[b] - p.refl[b]) * kb
            a6 = psi_func(kb, lai) - psi_func(2.0 * kb, lai)
            qcan[c.SUNLIT,b] = a1 * a2 + a3 * a4 + a5 * a6

            # Radiation absorbed by the shaded leaf, B4  Wang and Leuning 1998
            a2 = psi_func(k_dash_d[b], lai) - psi_func(k_dash_d[b] + kb, lai)
            a4 = psi_func(k_dash_b[b], lai) - psi_func(k_dash_b[b] + kb, lai)
            qcan[c.SHADED,b] = a1 * a2 + a3 * a4 - a5 * a6

    # Longwave radiation absorbed by sunlit leaves under isothermal conditions
    # B18 Wang and Leuning 1998
    a1 = -kd * c.SIGMA * tk**4
    a2 = p.emissivity_leaf * (1.0 - emissivity_air)
    a3 = psi_func(kb + kd, lai)
    a4 = 1.0 - p.emissivity_soil
    a5 = (p.emissivity_leaf - emissivity_air)
    a6 = psi_func(2.0 * kd, lai) * psi_func(kb - kd, lai)
    qcan[c.SUNLIT,c.LW] = a1 * (a2 * a3 + a4 * a5 * a6)

    # Longwave radiation absorbed by shaded leaves under isothermal conditions
    # B19 Wang and Leuning 1998
    a3 = psi_func(kd, lai)
    a6 = np.exp(-kd * lai) * a3
    qcan[c.SHADED,c.LW] = a1 * (a2 * a3 - a4 * a5 * a6) - qcan[c.SUNLIT,c.LW]

    apar[c.SUNLIT] = qcan[c.SUNLIT,c.VIS] * c.J_TO_UMOL
    apar[c.SHADED] = qcan[c.SHADED,c.VIS] * c.J_TO_UMOL

    # Total energy absorbed by canopy, summing VIS, NIR and LW components, to
    # leave us with the indivual leaf components.
    qcan = qcan.sum(axis=1)

    # where vegetated and sunlit
    if lai > c.LAI_THRESH and np.sum(sw_rad) > c.RAD_THRESH:
        lai_leaf[c.SUNLIT] = (1.0 - np.exp(-kb * lai)) / kb
    else:
        lai_leaf[c.SUNLIT] = 0.0

    lai_leaf[c.SHADED] = lai - lai_leaf[c.SUNLIT]

    return (qcan, apar, lai_leaf, kb)

def psi_func(z, lai):
    # B5 function from Wang and Leuning which integrates property passed via
    # arg list over the canopy space
    #
    # References:
    # -----------
    # * Wang and Leuning (1998) AFm, 91, 89-111. Page 106
    # min check avoids floatin underflow issues
    return ( (1.0 - np.exp(-min(z * lai, 30.0))) / z )

def calculate_cos_zenith(doy, xslat, hod):

    # calculate sin(bet), bet = elevation angle of sun
    # calculations according to goudriaan & van laar 1994 p30

    # sine of maximum declination
    sindec = -np.sin(23.45 * np.pi / 180.) * \
                np.cos(2. * np.pi * (doy + 10.0) / 365.0)

    cos_zenith = max(np.sin(np.pi / 180. * xslat) * sindec + \
                     np.cos(np.pi / 180. * xslat) * \
                     np.sqrt(1. - sindec * sindec) * \
                     np.cos(np.pi * (hod - 12.0) / 12.0), 1e-8)

    zenith_angle = np.rad2deg(np.arccos(cos_zenith))
    elevation = 90.0 - zenith_angle

    return (cos_zenith, elevation)

def calc_leaf_to_canopy_scalar(lai, k=None, kn=None, kb=None, big_leaf=False):
    """
    Calculate scalar to transform from big leaf to canopy.

    In the big-leaf model this scalar is applied to An, gsc and so E. This
    assumes that the photosynthetic capacity is assumed to decline exponentially
    through the canopy in proportion to the incident radiation estimated by
    Beer’s Law

    In the 2-leaf model this scalar is applied to the leaf Vcmax, Jmax and Rd
    values.

    Parameters:
    ----------
    lai : float
        leaf area index
    kb : float
        beam extinction coefficient for black leaves

    References:
    ----------
    * Wang and Leuning (1998) AFm, 91, 89-111; particularly the Appendix.
      Insert eqn C6 & C7 into B5
    """
    if big_leaf:
        scalex = (1.0 - np.exp(-k * lai)) / k
    else:
        scalex = np.zeros(2)
        scalex[c.SUNLIT] = (1.0 - np.exp(-kb * lai) * \
                                np.exp(-kn * lai)) / (kb + kn)
        scalex[c.SHADED] = (1.0 - np.exp(-kn * lai)) / kn - scalex[c.SUNLIT]

    return scalex
