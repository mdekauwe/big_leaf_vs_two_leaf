import numpy as np

#
## Parameters - Dushan to set these ...
#
lat = -23.575001
lon = 152.524994

# leaf transmissivity [-] (VIS: 0.07 - 0.15)
# ENF: 0.05; EBF: 0.05; DBF: 0.05; C3G: 0.070
tau = np.array([0.09, 0.3])

# leaf reflectance [-] (VIS:0.07 - 0.15)
# ENF: 0.062;EBF: 0.076;DBF: 0.092; C3G: 0.11
refl = np.array([0.09, 0.3])

Oi = 210.0
gamstar25 = 42.75
Kc25 = 404.9
Ko25 = 278.4
Ec = 79430.0
Eo = 36380.0
Eag = 37830.0
Ear = None # 34000.0
theta_hyperbol = 0.9995
theta_J = 0.7
quantum_yield = 0.3
absorptance = 0.8
alpha = quantum_yield * absorptance # (Medlyn et al 2002)
g0 = 0.001
g1 = 4.0
D0 = 1.5 # kpa
Hdv=200000.0
Hdj=200000.0
Vcmax25 = 60.0
Jmax25 = Vcmax25 * 1.67
Rd25 = 2.0
Eaj = 30000.0
Eav = 60000.0
deltaSj = 650.0
deltaSv = 650.0
Hdv = 200000.0
Hdj = 200000.0
Q10 = 2.0
gamma = 0.0
leaf_width = 0.02
LAI = 1.5
# Cambell & Norman, 11.5, pg 178
# The solar absorptivities of leaves (-0.5) from Table 11.4 (Gates, 1980)
# with canopies (~0.8) from Table 11.2 reveals a surprising difference.
# The higher absorptivityof canopies arises because of multiple reflections
# among leaves in a canopy and depends on the architecture of the canopy.
SW_abs = 0.8 # use canopy absorptance of solar radiation
