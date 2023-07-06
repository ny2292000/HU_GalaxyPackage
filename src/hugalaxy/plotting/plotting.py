from hugalaxy import calculate_mass

import matplotlib.pyplot as plt
import numpy as np
def plotRotationCurve(M33):
    v_sim = np.array(M33.print_simulated_curve())
    m33_rotational_curve = M33.rotation_curve.T
    plt.plot(m33_rotational_curve[:,0], m33_rotational_curve[:,1], color="blue" )
    plt.plot(v_sim[:,0], v_sim[:,1], color="red" )
    plt.xlabel("Radial Distance (lyr)")
    plt.ylabel("Tangential Velocity (km/s)")
    myMass = np.round(calculate_mass(M33.rho_0, M33.alpha_0, M33.h0)/1E10,2)
    gasMass = np.round(calculate_mass(M33.rho_1, M33.alpha_1, M33.h0)/1E10,2)
    plt.title("M33 Galaxy (z={}) Rotation Curve \n Luminous Mass {}E10 SunMass \n Gas Mass {}E10 SunMass".format(M33.redshift, myMass, gasMass))
    plt.xlim(0,np.max(m33_rotational_curve[:,0]))
    plt.ylim(0,np.max(m33_rotational_curve[:,1]))
    plt.show()


def move_rotation_curve(rotation_curve, z1=0.0, z2=20.0):
    rescaling_factor=(1+z2)/(1+z1)
    result  = np.zeros(np.shape(rotation_curve))
    result[:,0]=rotation_curve[:,0]/rescaling_factor
    result[:,1]=rotation_curve[:,1]*rescaling_factor
    return result

__all__=["plotRotationCurve","move_rotation_curve"]