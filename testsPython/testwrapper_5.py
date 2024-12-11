import os
import matplotlib
# matplotlib.use('TkAgg')  # Use the TkAgg backend (or another appropriate one)
import matplotlib.pyplot as plt
import numpy as np
import math
import xarray as xr
import pandas as pd
from hugalaxy import GalaxyWrapper, plotRotationCurve, calculate_density_parameters, move_rotation_curve
from timeit import default_timer as timer
import jupyter_to_medium as medium
import time
import ipywidgets as widgets
from timeit import default_timer as timer
from IPython.display import display, HTML
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection
from matplotlib.animation import FuncAnimation
import astropy.constants as cc



# Set the directory where your files are located
directory = './data'

# Loop through each file in the directory
for filename in os.listdir(directory):
    if ".00000" in filename:
        # Create the new filename by replacing "00000" with nothing
        new_filename = filename.replace(".00000", "")
        # Get the full path of the current and new file names
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f'Renamed "{filename}" to "{new_filename}"')




M_sun = cc.M_sun
M_sun

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

M33_Distance = 3.2E6
Radius_Universe_4D = 14.03E9
redshift0 = M33_Distance / (Radius_Universe_4D - M33_Distance)

nr = 200
# NZ should always be ODD
nz = 101
ntheta = 180
R_max = 50000.0

rho_0, alpha_0, rho_1, alpha_1, h0 = calculate_density_parameters(redshift0)
GalaxyMass = 5E10
# Create The Galaxy
M33 = GalaxyWrapper(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0, R_max, nr,
                    nz, ntheta, redshift0,GPU_ID=0, cuda=True, taskflow=False)
eta = 1E-2
temperature = 7

time_step_years = 100E3
n_epochs = 20
filename_base = "./data_galaxy_formation/"
for z in np.arange(5,2,-1):
    new_m33_rotational_curve = move_rotation_curve(m33_rotational_curve, redshift0, z)
    M33.read_galaxy_rotation_curve(new_m33_rotational_curve)
    M33.redshift = z
    current_time = 14.04E9/(1+M33.redshift)
    final_time = current_time + n_epochs * time_step_years
    epochs = np.geomspace(current_time, final_time, n_epochs)
    redshifts = pd.DataFrame(14.04E9/epochs -1)
    start_time = time.time()
    M33.FreeFallGalaxyFormation(epochs,filename_base)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The function for z = {z} took {elapsed_time} seconds to execute.")