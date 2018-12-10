import numpy as np

#
## Parameters - Dushan to set these ...
#
lat = -23.575001
lon = 152.524994

g0 = 0.001
g1 = 4.0
D0 = 1.5 # kpa
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

#
##  Fixed met stuff
#
wind = 2.5
pressure = 101325.0
Ca = 400.0
