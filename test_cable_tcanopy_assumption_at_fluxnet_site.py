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
import matplotlib.pyplot as plt
import random
import math
import pandas as pd
import xarray as xr

import parameters as p
import constants as c
from farq import FarquharC3
from penman_monteith_leaf import PenmanMonteith
from two_leaf_cable_assumption import Canopy as TwoLeafCable
from two_leaf import Canopy as TwoLeaf

__author__  = "Martin De Kauwe"
__version__ = "1.0 (09.11.2018)"
__email__   = "mdekauwe@gmail.com"


def main(met_fn, flx_fn, cab_fn, year_to_run, site):

    fpath = "/Users/mdekauwe/Downloads/"
    fname = "%s_met_and_plant_data_drought_2003.csv" % (site)
    fn = os.path.join(fpath, fname)
    df = pd.read_csv(fn, skiprows=range(1,2))

    (df_flx) = read_obs_file(flx_fn)
    df_flx = df_flx[df_flx.index.year == year_to_run]

    (df_met, lat, lon) = read_met_file(met_fn)
    df_met = df_met[df_met.index.year == year_to_run]

    par = df_met.PAR
    tair = df_met.Tair
    vpd = df_met.vpd
    wind = df_met.Wind
    pressure = df_met.PSurf
    Ca = 400.0
    #LAI = 3.0
    LAI = df.LAI


    #
    ## Parameters
    #
    p.g0 = 1E-09
    p.g1 = df.g1[0] #4.12
    p.D0 = 1.5 # kpa
    p.Vcmax25 = df.Vmax25[0]
    p.Jmax25 = p.Vcmax25 * 1.67

    p.Rd25 = None # Vcmax * 0.015
    p.Eaj = 30000.0
    p.Eav = 60000.0
    p.deltaSj = 650.0
    p.deltaSv = 650.0
    p.Hdv = 200000.0
    p.Hdj = 200000.0


    ##
    ### Run 2-leaf
    ##
    T = TwoLeaf(p, gs_model="medlyn")
    C = TwoLeafCable(p, gs_model="medlyn")

    ndays = int(len(df_met)/48)

    Anc_store = np.zeros(ndays)
    Ec_store = np.zeros(ndays)


    An_store = np.zeros(ndays)
    An_sun_store = np.zeros(ndays)
    An_sha_store = np.zeros(ndays)
    par_sun_store = np.zeros(ndays)
    par_sha_store = np.zeros(ndays)
    lai_sun_store = np.zeros(ndays)
    lai_sha_store = np.zeros(ndays)
    E_store = np.zeros(ndays)
    Tcan_store = np.zeros(ndays)

    Anc_store = np.zeros(ndays)
    Anc_sun_store = np.zeros(ndays)
    Anc_sha_store = np.zeros(ndays)
    parc_sun_store = np.zeros(ndays)
    parc_sha_store = np.zeros(ndays)
    laic_sun_store = np.zeros(ndays)
    laic_sha_store = np.zeros(ndays)
    Ec_store = np.zeros(ndays)
    Tcanc_store = np.zeros(ndays)
    Sunc_store = np.zeros(ndays)
    Shac_store = np.zeros(ndays)
    gpp_obs = np.zeros(ndays)
    e_obs = np.zeros(ndays)
    lai_obs = np.zeros(ndays)

    et_conv = c.MOL_WATER_2_G_WATER * c.G_TO_KG * 1800.
    an_conv = c.UMOL_TO_MOL * c.MOL_C_TO_GRAMS_C * 1800.

    cnt = 0
    for doy in range(ndays-1):


        Anx = 0.0
        Ex = 0.0
        anxsun = 0.0
        anxsha = 0.0
        parxsun = 0.0
        parxsha = 0.0
        laixsun = 0.0
        laixsha = 0.0
        Tcanx = 0.0

        Anc = 0.0
        Ec = 0.0
        Acabx = 0.0
        Ecabx = 0.0
        Lcabx = 0.0
        ancsun = 0.0
        ancsha = 0.0
        parcsun = 0.0
        parcsha = 0.0
        laicsun = 0.0
        laicsha = 0.0
        Tcancx = 0.0

        Aobsx = 0.0
        Eobsx = 0.0
        Lobsx = 0.0

        Sun_fracC = 0.0
        Sha_fracC = 0.0

        if doy == 0:
            tsoil = np.mean(tair[cnt:cnt+48])
        else:
            tsoil = np.mean(tair[cnt-48:cnt])

        hr_cnt = 0
        for i in range(48):

            if doy < 364:
                laix = LAI[cnt]
            else:
                #print(cnt, len(LAI), LAI[17518])
                laix = LAI[17518]

            hod = float(i)/2. + 1800. / 3600. / 2.

            (An, et, Tcan,
             apar, lai_leaf) = T.main(tair[cnt], par[cnt], vpd[cnt],
                                      wind[cnt], pressure[cnt], Ca, doy+1,
                                      hod, laix, tsoil)

            lambda_et = (c.H2OLV0 - 2.365E3 * tair[cnt]) * c.H2OMW

            Anx += np.sum(An) * an_conv
            anxsun += An[c.SUNLIT] * an_conv
            anxsha += An[c.SHADED] * an_conv
            parxsun += apar[c.SUNLIT] / c.UMOLPERJ / c.MJ_TO_J * 1800.0
            parxsha += apar[c.SHADED] / c.UMOLPERJ / c.MJ_TO_J * 1800.0 # MJ m-2 d-1
            laixsun += lai_leaf[c.SUNLIT]
            laixsha += lai_leaf[c.SHADED]
            sun_frac = lai_leaf[c.SUNLIT] / np.sum(lai_leaf)
            sha_frac = lai_leaf[c.SHADED] / np.sum(lai_leaf)

            if par[cnt] > 0.0:
                Tcanx += (Tcan[c.SUNLIT] * sun_frac) + (Tcan[c.SHADED] * sha_frac)


            Lobsx += laix
            Ex += np.sum(et) * et_conv


            (An, et, Tcan,
             apar, lai_leaf) = C.main(tair[cnt], par[cnt], vpd[cnt],
                                      wind[cnt], pressure[cnt], Ca, doy+1,
                                      hod, laix, tsoil)

            Anc += np.sum(An) * an_conv
            ancsun += An[c.SUNLIT] * an_conv
            ancsha += An[c.SHADED] * an_conv
            parcsun += apar[c.SUNLIT] / c.UMOLPERJ / c.MJ_TO_J * 1800.0
            parcsha += apar[c.SHADED] / c.UMOLPERJ / c.MJ_TO_J * 1800.0 # MJ m-2 d-1
            laicsun += lai_leaf[c.SUNLIT]
            laicsha += lai_leaf[c.SHADED]

            if par[cnt] > 0.0:
                Tcancx += Tcan
            Ec += et * et_conv
            Sun_fracC += sun_frac
            Sha_fracC += sha_frac

            Aobsx += df_flx.GPP[cnt] * an_conv
            Eobsx += df_flx.Qle[cnt] / lambda_et * et_conv


            cnt += 1
            if par[cnt] > 0.0:
                hr_cnt += 1

        
        An_store[doy] = Anx
        An_sun_store[doy] = anxsun
        An_sha_store[doy] = anxsha
        par_sun_store[doy] = parxsun
        par_sha_store[doy] = parxsha
        lai_sun_store[doy] = laixsun / 48.
        lai_sha_store[doy] = laixsha / 48.
        E_store[doy] = Ex

        Tcan_store[doy] = (Tcanx / float(hr_cnt))

        Anc_store[doy] = Anc
        Anc_sun_store[doy] = ancsun
        Anc_sha_store[doy] = ancsha
        parc_sun_store[doy] = parcsun
        parc_sha_store[doy] = parcsha
        laic_sun_store[doy] = laicsun / 48.
        laic_sha_store[doy] = laicsha / 48.
        Ec_store[doy] = Ec
        Tcanc_store[doy] = (Tcancx / float(hr_cnt))
        Sunc_store[doy] = Sun_fracC / 48.
        Shac_store[doy] = Sha_fracC / 48.

        gpp_obs[doy] = Aobsx
        e_obs[doy] = Eobsx
        lai_obs[doy] = Lobsx / 48

    window = 3
    An_store = moving_average(An_store, n=window)
    An_sun_store = moving_average(An_sun_store, n=window)
    An_sha_store = moving_average(An_sha_store, n=window)
    par_sun_store = moving_average(par_sun_store, n=window)
    par_sha_store = moving_average(par_sha_store, n=window)
    lai_sun_store = moving_average(lai_sun_store, n=window)
    lai_sha_store = moving_average(lai_sha_store, n=window)
    E_store = moving_average(E_store, n=window)
    Tcan_store = moving_average(Tcan_store, n=window)

    Anc_store = moving_average(Anc_store, n=window)
    Anc_sun_store = moving_average(Anc_sun_store, n=window)
    Anc_sha_store = moving_average(Anc_sha_store, n=window)
    parc_sun_store = moving_average(parc_sun_store, n=window)
    parc_sha_store = moving_average(parc_sha_store, n=window)
    laic_sun_store = moving_average(laic_sun_store, n=window)
    laic_sha_store = moving_average(laic_sha_store, n=window)
    Ec_store = moving_average(Ec_store, n=window)
    Sunc_store = moving_average(Sunc_store, n=window)
    Shac_store = moving_average(Shac_store, n=window)
    Tcanc_store = moving_average(Tcanc_store, n=window)

    #gpp_obs = moving_average(gpp_obs, n=window)
    #e_obs = moving_average(e_obs, n=window)
    #lai_obs = moving_average(lai_obs, n=window)

    fig = plt.figure(figsize=(9,10))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.3)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)


    ax1.plot(An_store, label="2-leaf - Me")
    ax1.plot(Anc_store, label="2-leaf - CABLE")
    ax1.set_ylabel("GPP (g C m$^{-2}$ d$^{-1}$)")
    ax1.legend(numpoints=1, loc="best")

    ax2.plot(E_store, label="2-leaf - Me")
    ax2.plot(Ec_store, label="2-leaf - CABLE")
    ax2.set_ylabel("E (mm d$^{-1}$)")

    ax3.plot(Tcan_store, label="2-leaf - Me")
    ax3.plot(Tcanc_store, label="2-leaf - CABLE")
    ax3.set_ylabel("T$_{canopy}$ (degrees C)")


    print(np.mean(np.abs(Tcan_store - Tcanc_store)))

    ax4.plot(Sunc_store, label="Sun")
    ax4.plot(Shac_store, label="Shade")
    ax4.set_ylabel("Frac (-)")
    ax4.legend(numpoints=1, loc="best")

    ax1.locator_params(nbins=6, axis="y")
    ax2.locator_params(nbins=6, axis="y")
    ax3.locator_params(nbins=6, axis="y")
    ax4.locator_params(nbins=6, axis="y")


    #plt.setp(ax1.get_xticklabels(), visible=False)
    #plt.setp(ax2.get_xticklabels(), visible=False)
    #plt.setp(ax3.get_xticklabels(), visible=False)
    #plt.setp(ax4.get_xticklabels(), visible=False)

    plt.show()



def read_met_file(fname):
    """ Build a dataframe from the netcdf outputs """

    ds = xr.open_dataset(fname)
    lat = ds.latitude.values[0][0]
    lon = ds.longitude.values[0][0]

    rh_there = False
    # W/m2, deg K, mm/s, kg/kg, m/s, Pa, ppmw
    try:
        vars_to_keep = ['SWdown','Tair','Qair','Wind','PSurf']
        df = ds[vars_to_keep].squeeze(dim=["x","y","z"],
                                      drop=True).to_dataframe()
    except KeyError:
        vars_to_keep = ['SWdown','Tair','Wind','PSurf','RH']
        df = ds[vars_to_keep].squeeze(dim=["x","y","z"],
                                      drop=True).to_dataframe()
        rh_there = True

    time_idx = df.index

    df = df.reindex(time_idx)
    df['year'] = df.index.year
    df['doy'] = df.index.dayofyear

    df["PAR"] = df.SWdown * c.SW_2_PAR
    df["Tair"] -= c.DEG_2_KELVIN

    if rh_there:
        esat = calc_esat(df.Tair)
        e = (df.RH / 100.) * esat
        df["vpd"] = (esat - e) / 1000.

    else:
        df["vpd"] = qair_to_vpd(df.Qair, df.Tair, df.PSurf)

    return df, lat, lon

def read_obs_file(fname):
    """ Build a dataframe from the netcdf outputs """

    ds = xr.open_dataset(fname)

    vars_to_keep = ['GPP','Qle']
    df = ds[vars_to_keep].squeeze(dim=["x","y"],
                                  drop=True).to_dataframe()

    # PALS-style netcdf is missing the first (half)hour timestamp and has
    # one extra from the next year, i.e. everything is shifted, so we need
    # to fix this. We will duplicate the
    # first hour interval and remove the last
    time_idx = df.index
    #diff = df.index.minute[1] - df.index.minute[0]
    #if diff == 0:
    #    time_idx = time_idx.shift(-1, freq='H')
    #    df = df.shift(-1, freq='H')
    #else:
    #    time_idx = time_idx.shift(-1, freq='30min')
    #    df = df.shift(-1, freq='30min')

    df = df.reindex(time_idx)
    df['year'] = df.index.year
    df['doy'] = df.index.dayofyear

    return df

def read_cable_file(fname):
    """ Build a dataframe from the netcdf outputs """
    ds = xr.open_dataset(fname)
    lat = ds.latitude.values[0][0]
    lon = ds.longitude.values[0][0]

    vars_to_keep = ['GPP','Qle','TVeg','Evap','LAI','GPP_shaded',\
                    'GPP_sunlit','PAR_sunlit','PAR_shaded',\
                    'LAI_sunlit','LAI_shaded']
    df = ds[vars_to_keep].squeeze(dim=["x","y"],
                                  drop=True).to_dataframe()

    # PALS-style netcdf is missing the first (half)hour timestamp and has
    # one extra from the next year, i.e. everything is shifted, so we need
    # to fix this. We will duplicate the
    # first hour interval and remove the last
    time_idx = df.index
    #diff = df.index.minute[1] - df.index.minute[0]
    #if diff == 0:
    #    time_idx = time_idx.shift(-1, freq='H')
    #    df = df.shift(-1, freq='H')
    #else:
    #    time_idx = time_idx.shift(-1, freq='30min')
    #    df = df.shift(-1, freq='30min')

    df = df.reindex(time_idx)
    df['year'] = df.index.year
    df['doy'] = df.index.dayofyear

    return df, lat, lon

def qair_to_vpd(qair, tair, press):

    # saturation vapor pressure
    es = 100.0 * 6.112 * np.exp((17.67 * tair) / (243.5 + tair))

    # vapor pressure
    ea = (qair * press) / (0.622 + (1.0 - 0.622) * qair)

    vpd = (es - ea) * c.PA_TO_KPA

    vpd = np.where(vpd < 0.05, 0.05, vpd)

    return vpd

def calc_esat(tair):
    """
    Calculates saturation vapour pressure

    Params:
    -------
    tair : float
        deg C

    Reference:
    ----------
    * Jones (1992) Plants and microclimate: A quantitative approach to
    environmental plant physiology, p110
    """

    esat = 613.75 * np.exp(17.502 * tair / (240.97 + tair))

    return esat

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == '__main__':

    fpath = "/Users/mdekauwe/research/CABLE_runs/met_data/fluxnet2015/"
    #fname = "FI-Hyy_1996-2014_FLUXNET2015_Met.nc"
    fname = "FR-Pue_2000-2014_FLUXNET2015_Met.nc"
    #fname = "ES-ES1_1999-2006_LaThuile_Met.nc"
    met_fn = os.path.join(fpath, fname)

    fpath = "/Users/mdekauwe/research/CABLE_runs/flux_files/fluxnet2015"
    #fname = "FI-Hyy_1996-2014_FLUXNET2015_Flux.nc"
    fname = "FR-Pue_2000-2014_FLUXNET2015_Flux.nc"
    #fname = "ES-ES1_1999-2006_LaThuile_Flux.nc"
    flx_fn = os.path.join(fpath, fname)

    site = os.path.basename(met_fn).split(".")[0][0:6]

    fpath = "/Users/mdekauwe/research/CABLE_runs/runs/FI-Hyy_CMIP6-MOSRS/outputs/"
    #fpath = "/Users/mdekauwe/Desktop/"
    fname = "%s_out.nc" %  (site)
    cab_fn = os.path.join(fpath, fname)

    #year_to_run = 2003
    year_to_run = 2003 # debugging to get first year

    main(met_fn, flx_fn, cab_fn, year_to_run, site)
