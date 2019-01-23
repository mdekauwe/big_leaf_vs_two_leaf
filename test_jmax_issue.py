import parameters as p
import constants as c
from farq import FarquharC3
from utils import calc_esat
from get_days_met_forcing import get_met_data
import numpy as np
import matplotlib.pyplot as plt

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

#
##  Fixed met stuff
#
wind = 2.5
pressure = 101325.0
Ca = 400.0
Cs = Ca
lai = p.LAI

F = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True, model_Q10=True,
               gs_model="medlyn")

An = np.zeros(48)
Anc = np.zeros(48)
Anj = np.zeros(48)
gsc = np.zeros(48)
for i in range(48):
    Tleaf_K = tair[i] + c.DEG_2_KELVIN

    (An[i], Anc[i], Anj[i], gsc[i]) = F.photosynthesis(p, Cs=Cs, Tleaf=Tleaf_K,
                                                       Par=par[i], vpd=vpd[i])

plt.plot(An, label="An")
plt.plot(Anc, label="Ac")
plt.plot(Anj, label="Aj")
plt.legend(numpoints=1, loc="best")
plt.show()
