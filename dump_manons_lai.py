#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys

site = "FI-Hyy"
fpath = "/Users/mdekauwe/Downloads/"
fname = "%s_met_and_plant_data_drought_2003.csv" % (site)
fn = os.path.join(fpath, fname)
df = pd.read_csv(fn, skiprows=range(1,2))

ndays = int(len(df)/48)

df_out = pd.DataFrame(columns=['LAI'], index=range(ndays+1))

cnt = 0
for doy in range(ndays):

    hod = 0

    for i in range(48):

        if doy < 364:
            df_out.loc[cnt].LAI = df.LAI[cnt]
        else:
            df_out.loc[cnt].LAI = df.LAI[cnt-1]

    cnt += 1

opath = "/Users/mdekauwe/Desktop/"
fname = "%s_lai.csv" % (site)
fn = os.path.join(fpath, fname)
df_out.to_csv(fn, index=False)
