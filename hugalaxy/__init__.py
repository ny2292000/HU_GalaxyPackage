from hugalaxy.HU_Galaxy_GalaxyWrapper import *
# from HU_Galaxy_GalaxyWrapper import *

__version__ = "0.0.1"  # Define the version of your package
import matplotlib.pyplot as plt
import numpy as np
def plotRotationCurve(M33):
    v_sim = np.array(M33.print_simulated_curve())
    m33_rotational_curve = np.array(M33.print_rotation_curve())
    plt.plot(m33_rotational_curve[:,0], m33_rotational_curve[:,1], color="blue" )
    plt.plot(v_sim[:,0], v_sim[:,1], color="red" )
    plt.xlabel("Radial Distance (lyr)")
    plt.ylabel("Tangential Velocity (km/s)")
    myMass = np.round(calculate_mass(M33.rho_0, M33.alpha_0, M33.h0)/1E10,2)
    gasMass = np.round(calculate_mass(M33.rho_1, M33.alpha_1, M33.h0)/1E10,2)
    plt.title("M33 Galaxy Rotation Curve \n Luminous Mass {}E10 SunMass \n Gas Mass {}E10 SunMass".format(myMass, gasMass))
    plt.xlim(0,50000)
    plt.ylim(0,135)
    plt.show()