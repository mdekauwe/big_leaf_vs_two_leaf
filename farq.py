#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model of C3 photosynthesis, this is passed to fitting function and we are
optimising Jmax25, Vcmax25, Rd, Eaj, Eav, deltaS

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (13.08.2012)"
__email__ = "mdekauwe@gmail.com"

import sys
import numpy as np
import os
import math
import constants as c

class FarquharC3(object):
    """
    Rate of photosynthesis in a leaf depends on the the rates of
    carboxylation (Ac) and the regeneration of ribulose-1,5-bisphosphate (RuBP)
    catalysed by the enzyme RUBISCO (Aj). This class returns the net leaf
    photosynthesis (An) which is the minimum of this two limiting processes
    less the rate of mitochondrial respiration in the light (Rd). We are
    ignoring the the "export" limitation (Ap) which could occur under high
    levels of irradiance.

    Model assumes conductance between intercellular space and the site of
    carboxylation is zero. The models parameters Vcmax, Jmax, Rd along with
    the calculated values for Kc, Ko and gamma star all vary with temperature.
    The parameters Jmax and Vcmax are typically fitted with a temperature
    dependancy function, either an exponential Arrheniuous or a peaked
    function, i.e. the Arrhenious function with a switch off point.

    All calculations in Kelvins...

    References:
    -----------
    * De Pury and Farquhar, G. D. (1997) Simple scaling of photosynthesis from
      leaves to canopies without the errors of big-leaf models. Plant Cell and
      Environment, 20, 537-557.
    * Farquhar, G.D., Caemmerer, S. V. and Berry, J. A. (1980) A biochemical
      model of photosynthetic CO2 assimilation in leaves of C3 species. Planta,
      149, 78-90.
    * Medlyn, B. E., Dreyer, E., Ellsworth, D., Forstreuter, M., Harley, P.C.,
      Kirschbaum, M.U.F., Leroux, X., Montpied, P., Strassemeyer, J.,
      Walcroft, A., Wang, K. and Loustau, D. (2002) Temperature response of
      parameters of a biochemically based model of photosynthesis. II.
      A review of experimental data. Plant, Cell and Enviroment 25, 1167-1179.
    """

    def __init__(self, peaked_Jmax=False, peaked_Vcmax=False,
                 model_Q10=False, gs_model=None,
                 adjust_for_low_temp=False):
        """
        Parameters
        ----------
        peaked_Jmax : logical
            use peaked or non-peaked arrhenius func
        peaked_Vcmax : logical
            use peaked or non-peaked arrhenius func
        model_Q10 : logical
            use Q10 to calculate Rd
        gs_model : string
            medlyn or leuning model
        adjust_for_low_temp : logical
            adjust Vcmax/Jmax at the low temperature end
        """
        self.peaked_Jmax = peaked_Jmax
        self.peaked_Vcmax = peaked_Vcmax
        self.model_Q10 = model_Q10
        self.gs_model = gs_model
        self.adjust_for_low_temp = adjust_for_low_temp

    def photosynthesis(self, p, Cs=None, Tleaf=None, Par=None, vpd=None,
                       mult=None, scalex=None):
        """
        Parameters
        ----------
        p : struct
            contains all the model params
        Cs : float
            leaf surface CO2 concentration [umol mol-1]
        Tleaf : float
            leaf temp [deg K]
        Par : float
            photosynthetically active radiation [umol m-2 s-1].
        vpd : float
            vapour pressure deficit [kPa].
        mult : float
            if passing a user defined gs model, i.e. not Medlyn or Leuning.
        scalex : float
            scaler to transform leaf to big leaf

        Returns:
        --------
        An : float
            Net leaf assimilation rate [umol m-2 s-1]
        gsc : float
            stomatal conductance to CO2 [mol m-2 s-1]
        """

        # calculate temp dependancies of MichaelisMenten constants for CO2, O2
        Km = self.calc_michaelis_menten_constants(p, Tleaf)

        # Effect of temp on CO2 compensation point
        gamma_star = self.arrh(p.gamstar25, p.Eag, Tleaf)

        # Calculate temperature dependancies on Vcmax and Jmax
        if self.peaked_Vcmax:
            Vcmax = self.peaked_arrh(p.Vcmax25, p.Eav, Tleaf, p.deltaSv, p.Hdv)
        else:
            Vcmax = self.arrh(Vcmax25, Eav, Tleaf)

        if self.peaked_Jmax:
            Jmax = self.peaked_arrh(p.Jmax25, p.Eaj, Tleaf, p.deltaSj, p.Hdj)
        else:
            Jmax = self.arrh(p.Jmax25, p.Eaj, Tleaf)

        # Calculations at 25 degrees C or the measurement temperature
        if p.Rd25 is not None:
            Rd = self.calc_resp(Tleaf, p.Q10, p.Rd25, p.Ear)
        else:
            Rd = 0.015 * Vcmax

        # Scaling from single leaf to canopy, see Wang & Leuning 1998 appendix C
        if scalex is not None:
            Rd *= scalex
            Vcmax *= scalex
            Jmax *= Jmax

        # Rate of electron transport, which is a function of absorbed PAR
        if Par is not None:
            J = self.calc_electron_transport_rate(p, Par, Jmax)
        # all measurements are calculated under saturated light!!
        else:
            J = Jmax
        Vj = J / 4.0

        if self.adjust_for_low_temp:
            Jmax = self.adj_for_low_temp(Jmax, Tleaf)
            Vcmax = self.adj_for_low_temp(Vcmax, Tleaf)

        if self.gs_model == "leuning":
            g0 = p.g0 * c.GSW_2_GSC
            gs_over_a = p.g1 / (Cs - gamma_star) / (1.0 + vpd / p.D0)

            # conductance to CO2
            gs_over_a *= c.GSW_2_GSC
            ci_over_ca = 1.0 - 1.6 * (1.0 + vpd / p.D0) / p.g1

        elif self.gs_model == "medlyn":
            if np.isclose(p.g0, 0.0):
                # I want a zero g0, but zero messes up the convergence,
                # numerical fix
                g0 = 1E-09
            else:
                g0 = p.g0 * c.GSW_2_GSC
            if vpd < 0.05:
                vpd = 0.05

            # 1.6 (from corrigendum to Medlyn et al 2011) is missing here,
            # because we are calculating conductance to CO2!
            if np.isclose(Cs, 0.0):
                gs_over_a = 0.0
            else:
                gs_over_a = (1.0 + p.g1 / math.sqrt(vpd)) / Cs
            ci_over_ca = p.g1 / (p.g1 + math.sqrt(vpd))

        elif self.gs_model == "user_defined":
            # Multiplier is user-defined.
            g0 = p.g0 / c.GSC_2_GSW
            gs_over_a = mult / c.GSC_2_GSW

        if ( np.isclose(Par, 0.0) | np.isclose(Vj, 0.0) ):
            Cic = Cs
            Cij = Cs
        else:
            # Solution when Rubisco activity is limiting
            (Cic) = self.solve_ci(g0, gs_over_a, Rd, Cs, gamma_star, Vcmax, Km)

            # Solution when electron transport rate is limiting
            (Cij) = self.solve_ci(g0, gs_over_a, Rd, Cs, gamma_star, Vj,
                                  2.0*gamma_star)

        if Cic <= 0.0 or Cic > Cs:
            Ac = 0.0
        else:
            Ac = self.assim(Cic, gamma_star, a1=Vcmax, a2=Km)
        Aj = self.assim(Cij, gamma_star, a1=Vj, a2=2.0*gamma_star)

        # When below light-compensation points, assume Ci=Ca.
        if Aj <= Rd + 1E-09:
            Cij = Cs
            Aj = self.assim(Cij, gamma_star, a1=Vj, a2=2.0*gamma_star)

        # Hyperbolic minimum.
        A = -self.quadratic(a=1.0-1E-04, b=Ac+Aj, c=Ac*Aj, large=True)

        # Net photosynthesis
        An = A - Rd

        # Calculate conductance to CO2
        gsc = max(g0, g0 + gs_over_a * An)

        # Extra step here; GS can be negative
        if gsc < g0:
            gsc = g0

        # Calculate conductance to water
        gsw = gsc * c.GSC_2_GSW

        # calculate the real Ci
        if gsc > 0.0 and An > 0.0:
            Ci = Cs - An / gsc
        else:
            Ci = Cs

        if np.isclose(Cs, 0.0):
            An = 0.0 - Rd
            gsc = 0.0
            Ci = Cs

        return (An, gsc)

    def calc_electron_transport_rate(self, p, Par, Jmax):
        """
        Electron transport rate for a given absorbed irradiance

        Parameters
        ----------
        p : struct
            contains all the model params
        Par : float
            photosynthetically active radiation [umol m-2 s-1].
        Jmax : float
            potential rate of electron transport

        theta_J : float
            Curvature of the light response (-)
        alpha : float
            Leaf quantum yield (initial slope of the A-light response curve)
            [mol mol-1]

        Reference:
        ----------
        * Farquhar G.D. & Wong S.C. (1984) An empirical model of stomatal
          conductance. Australian Journal of Plant Physiology 11, 191-210,
          eqn A but probably clearer in:
        * Leuning, R. et a., Leaf nitrogen, photosynthesis, conductance and
          transpiration: scaling from leaves to canopies, Plant Cell Environ.,
          18, 1183– 1200, 1995. Leuning 1995, eqn C3.
        """
        A = p.theta_J
        B = -(p.alpha * Par + Jmax);
        C = p.alpha * Par * Jmax;

        J = self.quadratic(a=A, b=B, c=C, large=False)

        return J

    def solve_ci(self, g0, gs_over_a, rd, Cs, gamma_star, gamma, beta):
        """
        Solve intercellular CO2 concentration using quadric equation, following
        Leuning 1990, see eqn 15a-c, solving simultaneous solution for Eqs 2, 12
        and 13

        Parameters
        ----------
        g0 : float
            residual stomatal conductance as net assimilation rate reaches zero
            (mol m-2 s-1)
        g1 : float
            slope of the sensitivity of stomatal conductance to assimilation
            (mol m-2 s-1)
        gs_over_a : float
            gs / A
        rd : float
            Rspiration rate [umol m-2 s-1]
        Cs : float
            leaf surface CO2 concentration [umol mol-1]
        gamma_star : float
            CO2 compensation point - base rate at 25 deg C / 298 K [umol mol-1]
        gamma : float
            if calculating Cic, this will be Vcmax
            if calculating Cij, this will be Vj
        beta : float
            if calculating Cic, this will be Km
            if calculating Cij, this will be 2.0*gamma_star

        Reference:
        ----------
        * Leuning (1990) Modelling Stomatal Behaviour and Photosynthesis of
          Eucalyptus grandis. Aust. J. Plant Physiol., 17, 159-75.
        """

        A = g0 + gs_over_a * (gamma - rd)

        arg1 = (1. - Cs * gs_over_a) * (gamma - rd)
        arg2 = g0 * (beta - Cs)
        arg3 = gs_over_a * (gamma * gamma_star + beta * rd)
        B = arg1 + arg2 - arg3

        arg1 = -(1.0 - Cs * gs_over_a)
        arg2 = (gamma * gamma_star + beta * rd)
        arg3 =  g0 * beta * Cs
        C = arg1 * arg2 - arg3

        Ci = self.quadratic(a=A, b=B, c=C, large=True)

        return Ci

    def calc_michaelis_menten_constants(self, p, Tleaf):
        """ Michaelis-Menten constant for O2/CO2, Arrhenius temp dependancy

        Parameters:
        ----------
        Tleaf : float
            leaf temperature [deg K]

        Kc25 : float
            Michaelis-Menten coefficents for carboxylation by Rubisco at
            25degC [umol mol-1] or 298 K
        Kc25 : float
            Michaelis-Menten coefficents for oxygenation by Rubisco at
            25degC [mmol mol-1]. Note value in Bernacchie 2001 is in mmol!!
            or 298 K
        Ec : float
            Activation energy for carboxylation [J mol-1]
        Eo : float
            Activation energy for oxygenation [J mol-1]
        Oi : float
            intercellular concentration of O2 [mmol mol-1]

        Returns:
        --------
        Km : float
        """
        Kc = self.arrh(p.Kc25, p.Ec, Tleaf)
        Ko = self.arrh(p.Ko25, p.Eo, Tleaf)

        Km = Kc * (1.0 + p.Oi / Ko)

        return Km

    def arrh(self, k25, Ea, Tk):
        """ Temperature dependence of kinetic parameters is described by an
        Arrhenius function.

        Parameters:
        ----------
        k25 : float
            rate parameter value at 25 degC or 298 K
        Ea : float
            activation energy for the parameter [J mol-1]
        Tk : float
            leaf temperature [deg K]

        Returns:
        -------
        kt : float
            temperature dependence on parameter

        References:
        -----------
        * Medlyn et al. 2002, PCE, 25, 1167-1179.
        """
        return k25 * np.exp((Ea * (Tk - 298.15)) / (298.15 * c.RGAS * Tk))

    def peaked_arrh(self, k25, Ea, Tk, deltaS, Hd):
        """ Temperature dependancy approximated by peaked Arrhenius eqn,
        accounting for the rate of inhibition at higher temperatures.

        Parameters:
        ----------
        k25 : float
            rate parameter value at 25 degC or 298 K
        Ea : float
            activation energy for the parameter [J mol-1]
        Tk : float
            leaf temperature [deg K]
        deltaS : float
            entropy factor [J mol-1 K-1)
        Hd : float
            describes rate of decrease about the optimum temp [J mol-1]

        Returns:
        -------
        kt : float
            temperature dependence on parameter

        References:
        -----------
        * Medlyn et al. 2002, PCE, 25, 1167-1179.

        """
        arg1 = self.arrh(k25, Ea, Tk)
        arg2 = 1.0 + np.exp((298.15 * deltaS - Hd) / (298.15 * c.RGAS))
        arg3 = 1.0 + np.exp((Tk * deltaS - Hd) / (Tk * c.RGAS))

        return arg1 * arg2 / arg3

    def assim(self, Ci, gamma_star, a1, a2):
        """calculation of photosynthesis with the limitation defined by the
        variables passed as a1 and a2, i.e. if we are calculating vcmax or
        jmax limited assimilation rates.

        Parameters:
        ----------
        Ci : float
            intercellular CO2 concentration.
        gamma_star : float
            CO2 compensation point in the abscence of mitochondrial respiration
            [umol m-2 s-1]
        a1 : float
            variable depends on whether the calculation is light or rubisco
            limited.
        a2 : float
            variable depends on whether the calculation is light or rubisco
            limited.

        Returns:
        -------
        assimilation_rate : float
            assimilation rate assuming either light or rubisco limitation.
            [umol m-2 s-1]
        """
        return a1 * (Ci - gamma_star) / (a2 + Ci)

    def calc_resp(self, Tleaf=None, Q10=None, Rd25=None, Ear=None, Tref=25.0):
        """ Calculate leaf respiration accounting for temperature dependence.

        Parameters:
        ----------
        Tleaf : float
            leaf temp [deg K]
        Q10 : float
            ratio of respiration at a given temperature divided by respiration
            at a temperature 10 degrees lower
        Rd25 : float
            Estimate of respiration rate at the reference temperature 25 deg C
            or or 298 K
        Ear : float
            activation energy for the parameter [J mol-1]
        Tref : float
            reference temperature

        Returns:
        -------
        Rd : float
            leaf respiration

        References:
        -----------
        Tjoelker et al (2001) GCB, 7, 223-230.
        """
        if self.model_Q10:
            Rd = Rd25 * Q10**(((Tleaf - c.DEG_2_KELVIN) - Tref) / 10.0)
        else:
            Rd = self.arrh(Rd25, Ear, Tleaf)

        return Rd

    def quadratic(self, a=None, b=None, c=None, large=False):
        """ minimilist quadratic solution as root for J solution should always
        be positive, so I have excluded other quadratic solution steps. I am
        only returning the smallest of the two roots

        Parameters:
        ----------
        a : float
            co-efficient
        b : float
            co-efficient
        c : float
            co-efficient

        Returns:
        -------
        val : float
            positive root
        """
        d = b**2.0 - 4.0 * a * c # discriminant
        if d < 0.0:
            raise ValueError('imaginary root found')
        #root1 = np.where(d>0.0, (-b - np.sqrt(d)) / (2.0 * a), d)
        #root2 = np.where(d>0.0, (-b + np.sqrt(d)) / (2.0 * a), d)

        if large:
            if np.isclose(a, 0.0) and b > 0.0:
                root = -c / b
            elif np.isclose(a, 0.0) and np.isclose(b, 0.0):
                root = 0.0
                if c != 0.0:
                    raise ValueError('Cant solve quadratic')
            else:
                root = (-b + np.sqrt(d)) / (2.0 * a)
        else:
            if np.isclose(a, 0.0) and b > 0.0:
                root = -c / b
            elif np.isclose(a, 0.0) and np.isclose(b, 0.0):
                root == 0.0
                if c != 0.0:
                    raise ValueError('Cant solve quadratic')
            else:
                root = (-b - np.sqrt(d)) / (2.0 * a)

        return root

    def adj_for_low_temp(self, param, Tk, lower_bound=0.0, upper_bound=10.0):
        """
        Function allowing Jmax/Vcmax to be forced linearly to zero at low T

        Parameters:
        ----------
        param : float
            value to adjust
        Tk : float
            air temperature (Kelvin)
        """
        Tc = Tk - c.DEG_2_KELVIN

        if Tc < lower_bound:
            param = 0.0
        elif Tc < upper_bound:
            param *= (Tc - lower_bound) / (upper_bound - lower_bound)

        return param
