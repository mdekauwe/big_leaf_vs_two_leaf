#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Isothermal Penman-Monteith

Reference:
==========
* Leuning et al. (1995) Leaf nitrogen, photosynthesis, conductance and
  transpiration: scaling from leaves to canopies
* Wang and Leuning (1998) A two-leaf model for canopy conductance,
  photosynthesis and partitioning of available energy I: Model description and
  comparison with a multi-layered model
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (23.07.2015)"
__email__ = "mdekauwe@gmail.com"


import math
import numpy as np
import sys

from utils import calc_esat
import constants as c
import parameters as p

class PenmanMonteith(object):

    def __init__(self, angle=35.0):
        # not used
        self.angle = angle              # angle from horizontal (deg) 0-90

    def main(self, p, tleaf, tair, gsc, vpd, pressure, wind, par):

        tleaf_k = tleaf + c.DEG_2_KELVIN
        tair_k = tair + c.DEG_2_KELVIN

        air_density = pressure  / (c.RSPECIFC_DRY_AIR * tair_k)
        cmolar = pressure  / (c.RGAS * tair_k)
        rnet = self.calc_rnet(par, tair, tair_k, tleaf_k, vpd, pressure)

        (grn, gh, gbH, gw) = self.calc_conductances(p, tair_k, tleaf, tair,
                                                    wind, gsc, cmolar)
        (et, lambda_et) = self.calc_et(tleaf, tair, vpd, pressure, wind, par,
                                       gh, gw, rnet)
        return (et, lambda_et)

    def calc_et(self, tleaf, tair, vpd, pressure, wind, par, gh, gw,
                rnet):
        """
        Calculate transpiration following Penman-Monteith at the leaf level
        accounting for effects of leaf temperature and feedback on evaporation.
        For example, if leaf temperature is above the leaf temp, it can increase
        vpd, but it also reduves the lw and thus the net rad availble for
        evaporaiton.

        Parameters:
        ----------
        tair : float
            air temperature (deg C)
        tleaf : float
            leaf temperature (deg C)
        vpd : float
            Vapour pressure deficit (kPa, needs to be in Pa, see conversion
            below)
        pressure : float
            air pressure (using constant) (Pa)
        wind : float
            wind speed (m s-1)
        par : float
            Photosynthetically active radiation (umol m-2 s-1)
        gh : float
            boundary layer conductance to heat - free & forced & radiative
            components (mol m-2 s-1)
        gw :float
            conductance to water vapour - stomatal & bdry layer components
            (mol m-2 s-1)
        rnet : float
            Net radiation (J m-2 s-1 = W m-2)

        Returns:
        --------
        et : float
            transpiration (mol H2O m-2 s-1)
        lambda_et : float
            latent heat flux (W m-2)
        """
        # latent heat of water vapour at air temperature (J mol-1)
        lambda_et = (c.H2OLV0 - 2.365E3 * tair) * c.H2OMW

        # slope of sat. water vapour pressure (e_sat) to temperature curve
        # (Pa K-1), note kelvin conversion in func
        slope = (calc_esat(tair + 0.1) - calc_esat(tair)) / 0.1

        # psychrometric constant (Pa K-1)
        gamma = c.CP * c.AIR_MASS * pressure / lambda_et

        # Y cancels in eqn 10
        arg1 = slope * rnet + (vpd * c.KPA_2_PA) * gh * c.CP * c.AIR_MASS
        arg2 = slope + gamma * gh / gw

        # W m-2
        LE = max(0.0, arg1 / arg2)

        # transpiration, mol H20 m-2 s-1
        et = max(0.0, LE / lambda_et)

        return (et, LE)

    def calc_conductances(self, p, tair_k, tleaf, tair, wind, gsc, cmolar,
                          lai=None):
        """
        Both forced and free convection contribute to exchange of heat and mass
        through leaf boundary layers at the wind speeds typically encountered
        within plant canopies (<0-5ms~'). It is particularly imponant to
        includethe contribution of buoyancy forces to the boundary
        conductance for sunlit leaves deep within the canopy where wind speeds
        are low, for without this mechanism computed leaf temperatures become
        excessively high.

        Parameters:
        ----------
        tair_k : float
            air temperature (K)
        tleaf : float
            leaf temperature (deg C)
        wind : float
            wind speed (m s-1)
        gsc : float
            stomatal conductance to CO2 (mol m-2 s-1)
        cmolar : float
            Conversion from m s-1 to mol m-2 s-1

        Returns:
        --------
        grn : float
            radiation conductance (mol m-2 s-1)
        gh : float
            total leaf conductance to heat (mol m-2 s-1), *note* two sided.
        gbH : float
            total boundary layer conductance to heat for one side of the leaf
        gw : float
            total leaf conductance to water vapour (mol m-2 s-1)

        References
        ----------
        * Leuning 1995, appendix E
        * Medlyn et al. 2007 appendix, for need for cmolar
        """

        # radiation conductance for big leaf from Wang and Leuning (1998)
        # just below eqn 9. This differs from Leuning (1995) where it is
        # expressed as a function of the diffuse extinction coefficent and
        # cumulative LAI (NB. units already in mol m-2 s-1).
        grn = (4.0 * c.SIGMA * tair_k**3 * \
               p.emissivity_leaf) / (c.CP * c.AIR_MASS)

        # boundary layer conductance for heat: single sided, forced convection
        # (mol m-2 s-1)
        gbHw = 0.003 * math.sqrt(wind / p.leaf_width) * cmolar

        if np.isclose(tleaf - tair, 0.0):
            gbHf = 0.0
        else:
            # grashof number
            grashof_num = max(1.e-06, 1.6E8 * math.fabs(tleaf - tair) * \
                                      p.leaf_width**3)

            # boundary layer conductance for heat: single sided, free convection
            # (mol m-2 s-1)
            gbHf = 0.5 * c.DHEAT * grashof_num**0.25 / p.leaf_width * cmolar

            if lai is None:
                gbHf = 0.5 * c.DHEAT * \
                        grashof_num**0.25 / p.leaf_width * cmolar
            else:
                # Cable uses the leaf LAI to adjust this for the 2-leaf
                gbHf = lai * 0.5 * c.DHEAT * \
                        grashof_num**0.25 / p.leaf_width * cmolar
            gbHf = max(1.e-06, gbHf)

        # total boundary layer conductance for heat
        gbH = gbHw + gbHf

        # total boundary leaf conductance for heat (mol m-2 s-1) - two sided
        gh = 2.0 * (gbH + grn)

        # total leaf conductance for water vapour (mol m-2 s-1)
        gbw = gbH * c.GBH_2_GBW
        gsw = gsc * c.GSC_2_GSW
        gw = (gbw * gsw) / (gbw + gsw)

        return (grn, gh, gbH, gw)

    def calc_rnet(self, par, tair, tair_k, tleaf_k, vpd, pressure):
        """
        Net isothermal radaiation (Rnet, W m-2), i.e. the net radiation that
        would be recieved if leaf and air temperature were the same.

        References:
        ----------
        Jarvis and McNaughton (1986)

        Parameters:
        ----------
        par : float
            Photosynthetically active radiation (umol m-2 s-1)
        tair : float
            air temperature (deg C)
        tair_k : float
            air temperature (K)
        tleaf_k : float
            leaf temperature (K)
        vpd : float
            Vapour pressure deficit (kPa, needs to be in Pa, see conversion
            below)
        pressure : float
            air pressure (using constant) (Pa)

        Returns:
        --------
        rnet : float
            Net radiation (J m-2 s-1 = W m-2)

        """

        # Short wave radiation (W m-2)
        #sw_rad = par * c.PAR_2_SW

        # this matches CABLE's logic ...  which means they are halving the
        # SW_down that is used to compute the direct/diffuse terms and presumbly
        # all other calcs
        sw_rad = par / 4.6 # W m-2

        # absorbed short-wave radiation
        #SW_abs = self.SW_abs * math.cos(math.radians(self.angle)) * SW_rad

        # atmospheric water vapour pressure (Pa)
        ea = max(0.0, calc_esat(tair) - (vpd * c.KPA_2_PA))

        # apparent emissivity for a hemisphere radiating at air temperature
        # eqn D4
        emissivity_atm = 0.642 * (ea / tair_k)**(1.0 / 7.0)

        # isothermal net LW radiaiton at top of canopy, assuming emissivity of
        # the canopy is 1
        net_lw_rad = (1.0 - emissivity_atm) * c.SIGMA * tair_k**4

        # black leaves, table 1, Leuning 1995
        #kd = 0.8

        # isothermal net radiation (W m-2)
        rnet = p.SW_abs * sw_rad - net_lw_rad #* kd * exp(-kd * lai)

        return rnet

    def calc_slope_of_saturation_vapour_pressure_curve(self, tair):
        """ Eqn 13 from FAO paper, Allen et al. 1998.

        Parameters:
        ----------
        tair : float
            air temperature (deg C)

        Returns:
        --------
        slope : float
            slope of saturation vapour pressure curve [Pa degC-1]

        """
        t = tair + 237.3
        arg1 = 4098.0 * (0.6108 * math.exp((17.27 * tair) / t))
        arg2 = t**2

        return (arg1 / arg2) * c.KPA_2_PA


def calc_net_radiation(doy, hod, latitude, longitude, sw_rad, tair,
                       ea, albedo=0.23, elevation=0.0):

    cos_zenith = calculate_solar_geometry(doy, hod, latitude, longitude)

    # J m-2 s-1
    Rext = calc_extra_terrestrial_rad(doy, cos_zenith)

    # Clear-sky solar radiation, J m-2 s-1
    Rs0 = (0.75 + 2E-5 * elevation) * Rext

    # net longwave radiation, rnl
    arg1 = c.SIGMA * (tair + c.DEG_2_KELVIN)**4
    arg2 = 0.34 - 0.14 * math.sqrt(ea * c.PA_2_KPA)
    if Rs0 > 0.000001:  #divide by zero
        arg3 = 1.35 * sw_rad / Rs0 - 0.35
    else:
        arg3 = 0.0
    Rnl = arg1 * arg2 * arg3

    # net shortwave radiation, J m-2 s-1
    Rns = (1.0 - albedo) * sw_rad

    # net radiation, J m-2 s-1 or W m-2
    Rn = Rns - Rnl

    return Rn

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
    rlat = latitude * math.pi / 180.0 # radians

    # A13 - De Pury & Farquhar
    sin_beta = math.sin(rlat) * math.sin(rdec) + math.cos(rlat) * \
                math.cos(rdec) * math.cos(h)
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
    return (2.0 * math.pi * (float(doy) - 1.0) / 365.0)

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
    decl = -23.4 * (math.pi / 180.) * math.cos(2.0 * math.pi *\
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
    et = 0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma) -\
         0.014615 * math.cos(2.0 * gamma) - 0.04089 * math.sin(2.0 * gamma)

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
    return (math.pi * (t - t0) / 12.0)

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
        So = Sc * \
            (1.0 + 0.033 * math.cos(doy / 365.0 * 2.0 * math.pi)) * cos_zenith
    else:
        So = 0.0

    return So


def round_to_value(number, roundto):
    return (round(number / roundto) * roundto)


if __name__ == '__main__':

    par = 1000.0
    tleaf = 21.5
    tair = 20.0
    gs = 0.15
    vpd = 2.0
    pressure = 101325.0 # Pa
    wind = 2.0

    PM = PenmanMonteith()
    (et, lambda_et) = PM.main(p, tleaf, tair, gs, vpd, pressure, wind, par)

    print(et, lambda_et)
