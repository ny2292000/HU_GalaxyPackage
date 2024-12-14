import matplotlib
# matplotlib.use('TkAgg')  # Use the TkAgg backend (or another appropriate one)
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import pandas as pd
from hugalaxy.hugalaxy import GalaxyWrapper, calculate_density_parameters, move_rotation_curve
from hugalaxy.plotting import plotRotationCurve
from hugalaxy.calibration import calibrate_density_coefficients
from timeit import default_timer as timer

####################################################
# MODELING M33 GALAXY
####################################################

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

print(np.shape(m33_rotational_curve))

M33_Distance = 3.2E6
Radius_Universe_4D = 14.01E9
redshift = M33_Distance / (Radius_Universe_4D - M33_Distance)
nr = 320
# NZ should always be ODD
nz = 101
ntheta = 180
R_max = 50000.0

rho_0, alpha_0, rho_1, alpha_1, h0 = calculate_density_parameters(redshift)
GalaxyMass = 5E10
# Create The Galaxy
M33 = GalaxyWrapper(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0, R_max, nr,
                    nz, ntheta, redshift,GPU_ID=0, cuda=True, taskflow=True)
M33.density_wrapper_internal()
M33.read_galaxy_rotation_curve(m33_rotational_curve)
tic = time.perf_counter()
[fitting_coeffs, df] = calibrate_density_coefficients(m33_rotational_curve, redshift,M33, 20)
toc = time.perf_counter()
print(f"Calibrated Galaxy Evolution in {toc - tic:0.4f} seconds")
