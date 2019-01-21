import parameters as p
from farq import FarquharC3

p.Vcmax25 = 10.
Jmax25 = p.Vcmax25 * 1.67

Cs = 400.
Tleaf_K = 25.0 + 273.15
par = 1800.
dleaf = 1.5
F = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True, model_Q10=True,
               gs_model="medlyn")
(An, gsc) = F.photosynthesis(p, Cs=Cs, Tleaf=Tleaf_K,
                             Par=par, vpd=dleaf)

print(An, gsc)
