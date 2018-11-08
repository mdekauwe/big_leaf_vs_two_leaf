#!/usr/bin/env python
"""
Various radiation funcs needed for the two-leaf approximation
"""

import sys
import numpy as np
import math

__author__  = "Martin De Kauwe"
__version__ = "1.0 (09.11.2018)"
__email__   = "mdekauwe@gmail.com"


def calculate_solar_geometry(doy, hod, latitude, longitude):
    """
    The solar zenith angle is the angle between the zenith and the centre
    of the sun's disc. The solar elevation angle is the altitude of the
    sun, the angle between the horizon and the centre of the sun's disc.
    Since these two angles are complementary, the cosine of either one of
    them equals the sine of the other, i.e. cos theta = sin beta. I will
    use cos_zen throughout code for simplicity.
    Arguments:
    ----------
    doy : double
        day of year
    hod : double:
        hour of the day [0.5 to 24]
    Returns:
    --------
    cos_zen : double
        cosine of the zenith angle of the sun in degrees (returned)
    elevation : double
        solar elevation (degrees) (returned)
    References:
    -----------
    * De Pury & Farquhar (1997) PCE, 20, 537-557.
    """

    # need to convert 30 min data, 0-47 to 0-23.5
    hod /= 2.0

    gamma = day_angle(doy)
    rdec = calculate_solar_declination(doy, gamma)
    et = calculate_eqn_of_time(gamma)
    t0 = calculate_solar_noon(et, longitude)
    h = calculate_hour_angle(hod, t0)
    rlat = latitude * np.pi / 180.0 # radians

    # A13 - De Pury & Farquhar
    sin_beta = np.sin(rlat) * np.sin(rdec) + np.cos(rlat) * \
                np.cos(rdec) * np.cos(h)
    cos_zenith = sin_beta # The same thing, going to use throughout
    if cos_zenith > 1.0:
        cos_zenith = 1.0
    elif cos_zenith < 0.0:
        cos_zenith = 0.0

    return cos_zenith

def day_angle(doy):
    """
    Calculation of day angle - De Pury & Farquhar, '97: eqn A18
    Reference:
    ----------
    * De Pury & Farquhar (1997) PCE, 20, 537-557.
    * J. W. Spencer (1971). Fourier series representation of the position of
      the sun.
    Returns:
    ---------
    gamma - day angle in radians.
    """
    return (2.0 * np.pi * (float(doy) - 1.0) / 365.0)

def calculate_solar_declination(doy, gamma):
    """
    Solar Declination Angle is a function of day of year and is indepenent
    of location, varying between 23deg45' to -23deg45'
    Arguments:
    ----------
    doy : int
        day of year, 1=jan 1
    gamma : double
        fractional year (radians)
    Returns:
    --------
    dec: float
        Solar Declination Angle [radians]
    Reference:
    ----------
    * De Pury & Farquhar (1997) PCE, 20, 537-557.
    * Leuning et al (1995) Plant, Cell and Environment, 18, 1183-1200.
    * J. W. Spencer (1971). Fourier series representation of the position of
      the sun.
    """

    # Solar Declination Angle (radians) A14 - De Pury & Farquhar
    decl = -23.4 * (np.pi / 180.) * np.cos(2.0 * np.pi *\
            (float(doy) + 10.) / 365.);

    return decl

def calculate_eqn_of_time(gamma):
    """
    Equation of time - correction for the difference btw solar time
    and the clock time.
    Arguments:
    ----------
    doy : int
        day of year
    gamma : double
        fractional year (radians)
    References:
    -----------
    * De Pury & Farquhar (1997) PCE, 20, 537-557.
    * Campbell, G. S. and Norman, J. M. (1998) Introduction to environmental
      biophysics. Pg 169.
    * J. W. Spencer (1971). Fourier series representation of the position of
      the sun.
    * Hughes, David W.; Yallop, B. D.; Hohenkerk, C. Y. (1989),
      "The Equation of Time", Monthly Notices of the Royal Astronomical
      Society 238: 1529–1535
    """


    #
    # from Spencer '71. This better matches the de Pury worked example (pg 554)
    # The de Pury version is this essentially with the 229.18 already applied
    # It probably doesn't matter which is used, but there is some rounding
    # error below (radians)
    #
    et = 0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma) -\
         0.014615 * np.cos(2.0 * gamma) - 0.04089 * np.sin(2.0 * gamma)

    # radians to minutes
    et *= 229.18;

    return et

def calculate_solar_noon(et, longitude):
    """
    Calculation solar noon - De Pury & Farquhar, '97: eqn A16
    Reference:
    ----------
    * De Pury & Farquhar (1997) PCE, 20, 537-557.
    Returns:
    ---------
    t0 - solar noon (hours).
    """

    # all international standard meridians are multiples of 15deg east/west of
    # greenwich
    Ls = round_to_value(longitude, 15.)
    t0 = 12.0 + (4.0 * (Ls - longitude) - et) / 60.0

    return t0

def round_to_value(number, roundto):
    return (round(number / roundto) * roundto)

def calculate_hour_angle(t, t0):
    """
    Calculation solar noon - De Pury & Farquhar, '97: eqn A15
    Reference:
    ----------
    * De Pury & Farquhar (1997) PCE, 20, 537-557.
    Returns:
    ---------
    h - hour angle (radians).
    """
    return (np.pi * (t - t0) / 12.0)

def calc_extra_terrestrial_rad(doy, cos_zenith):
    """
    Solar radiation incident outside the earth's atmosphere, e.g.
    extra-terrestrial radiation. The value varies a little with the earths
    orbit.
    Using formula from Spitters not Leuning!
    Arguments:
    ----------
    doy : double
        day of year
    cos_zenith : double
        cosine of zenith angle (radians)
    Returns:
    --------
    So : float
        solar radiation normal to the sun's bean outside the Earth's atmosphere
        (J m-2 s-1)
    Reference:
    ----------
    * Spitters et al. (1986) AFM, 38, 217-229, equation 1.
    """

    # Solar constant (J m-2 s-1)
    Sc = 1370.0

    if cos_zenith > 0.0:
        #
        # remember sin_beta = cos_zenith; trig funcs are cofuncs of each other
        # sin(x) = cos(90-x) and cos(x) = sin(90-x).
        #
        So = Sc * (1.0 + 0.033 * np.cos(doy / 365.0 * 2.0 * np.pi)) * cos_zenith
    else:
        So = 0.0

    return So
