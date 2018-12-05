#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys

site = "FI-Hyy" #"FR-Pue" #"FI-Hyy"
fpath = "/Users/mdekauwe/Downloads/"
fname = "%s_met_and_plant_data_drought_2003.csv" % (site)
fn = os.path.join(fpath, fname)
df = pd.read_csv(fn, skiprows=range(1,2))

ndays = int(len(df)/48)

df_out = pd.DataFrame(columns=['LAI'], index=range(365))

cnt = 0
k = 0
for doy in range(365):
    laix = 0.0
    for i in range(48):
        if doy < 364:
            laix += df.LAI[k]
        k += 1
    df_out.loc[cnt].LAI = laix / 48.
    cnt += 1
df_out.loc[364] = df_out.loc[363]

opath = "/Users/mdekauwe/Desktop/"
ofname = "%s_lai.csv" % (site)
fn = os.path.join(opath, ofname)
df_out.to_csv(fn, index=False)
