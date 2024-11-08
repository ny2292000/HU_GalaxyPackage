import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend (or another appropriate one)
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from hugalaxy import GalaxyWrapper, calculate_density_parameters, move_rotation_curve
from hugalaxy.plotting import plotRotationCurve
from hugalaxy.calibration import calibrate_density_coefficients
from timeit import default_timer as timer
Radius_4D=14.01
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
data=calibrate_density_coefficients(m33_rotational_curve, redshift, M33)

df = pd.DataFrame(data, columns=["rho_0","alpha_0", "rho_1", "alpha_1", "h0"])
r4d = Radius_4D/(1+df.index)
df["redshift_birth"]=df.index
df["r4d"]=r4d
# Assuming you have the DataFrame df with the required columns

# Calculate log(r4d) column
df['log_r4d'] = np.log10(14 / (1 + df['redshift_birth']))

# Define the degree of the polynomial fit
degree = 1

# Create an empty dictionary to store the fitting coefficients
fitting_coeffs = {}

# Loop over the columns to calculate the fitting coefficients
for column in ['rho_0', 'alpha_0', 'rho_1', 'alpha_1', 'h0']:
    # Calculate log(column) column
    df['log_' + column] = np.log10(df[column])

    # Perform the polynomial fit using numpy.polyfit
    fit_coeffs = np.polyfit(df['log_r4d'], df['log_' + column], degree)

    # Store the fitting coefficients in the dictionary
    fitting_coeffs[column] = fit_coeffs

# Print the fitting coefficients
for column, coeffs in fitting_coeffs.items():
    print('pow(r4d, {}) * pow(10, {}}),'.format(coeffs[0],coeffs[1]))
