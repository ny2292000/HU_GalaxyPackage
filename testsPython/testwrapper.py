# MODELING M33 GALAXY
####################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting.plotting import plotRotationCurve, move_rotation_curve, calculate_density_parameters
from hugalaxy import GalaxyWrapper
import time
time.sleep(30)  # Sleep for 30 seconds

# Rest of your script...


m33_rotational_curve = np.array( [
    [0.0, 0.0],
    [1508.7187, 38.674137],
    [2873.3889, 55.65067],
    [4116.755, 67.91063],
    [5451.099, 79.22689],
    [6846.0957, 85.01734],
    [8089.462, 88.38242],
    [9393.48, 92.42116],
    [10727.824, 95.11208],
    [11880.212, 98.342697],
    [13275.208, 99.82048],
    [14609.553, 102.10709],
    [18521.607, 104.25024],
    [22403.336, 107.60643],
    [26406.369, 115.40966],
    [30379.076, 116.87875],
    [34382.107, 116.05664],
    [38354.813, 117.93005],
    [42266.87, 121.42091],
    [46300.227, 128.55017],
    [50212.285, 132.84966]
])

M33_Distance = 3.2E6
Radius_Universe_4D = 14.03E9
redshift = M33_Distance / (Radius_Universe_4D - M33_Distance)
nr = 120
# NZ should always be ODD
nz = 101
ntheta = 180
R_max = 50000.0

rho_0, alpha_0, rho_1, alpha_1, h0 = calculate_density_parameters(redshift)
GalaxyMass = 5E10
# Create The Galaxy
M33 = GalaxyWrapper(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0, R_max, nr,
                    nz, ntheta, redshift,GPU_ID=0, cuda=True, taskflow=True)
# Load the new rotation curve
M33.read_galaxy_rotation_curve(m33_rotational_curve)
# Simulate the new curve
M33.simulate_rotation_curve()
# Plot it
plotRotationCurve(M33)
df = pd.DataFrame(columns=["rho_0","alpha_0", "rho_1", "alpha_1", "h0"])

# for redshift_birth in np.arange(0,35,1):
#     r4d = 14/(1+redshift_birth)
#     M33.redshift=redshift_birth
#     M33.rho_0, M33.alpha_0, M33.rho_1, M33.alpha_1, M33.h0 = calculate_density_parameters(redshift_birth)
#     new_rotation_curve = move_rotation_curve(m33_rotational_curve, redshift, redshift_birth )
#     M33.read_galaxy_rotation_curve(new_rotation_curve)
#     values = M33.move_galaxy(True)
#     df.loc[redshift_birth] = values
# r4d = 14/(1+df.index)
# df["redshift_birth"]=df.index
# df["r4d"]=r4d

a=1