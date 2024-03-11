import matplotlib.pyplot as plt
import numpy as np
def plotRotationCurve(M33, redshift_birth):
    plt.plot(M33.x_rotation_points, M33.v_rotation_points, color="blue" )
    plt.plot(M33.x_rotation_points, M33.v_simulated_points, color="red" )
    plt.xlabel("Radial Distance (lyr)")
    plt.ylabel("Tangential Velocity (km/s)")
    myMass = np.round(M33.calculate_mass(M33.rho_0, M33.alpha_0, M33.h0)/1E10,2)
    gasMass = np.round(M33.calculate_mass(M33.rho_1, M33.alpha_1, M33.h0)/1E10,2)
    plt.title("M33 Galaxy (z={}) Rotation Curve \n Luminous Mass {}E10 SunMass \n Gas Mass {}E10 SunMass".format(M33.redshift, myMass, gasMass))
    plt.xlim(0,np.max(M33.x_rotation_points))
    plt.ylim(0,np.max(M33.v_rotation_points))
    filename = f"./Figures/RotationCurve_z_{redshift_birth}.png"
    plt.savefig(filename)
    plt.show()
    # Close the plot to free up memory
    plt.close()


__all__=["plotRotationCurve"]