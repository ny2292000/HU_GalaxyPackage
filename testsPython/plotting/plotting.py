import matplotlib.pyplot as plt
import numpy as np

def plotRotationCurve(M33):
    plt.plot(M33.x_rotation_points, M33.v_rotation_points, color="blue" )
    plt.plot(M33.x_rotation_points, M33.v_simulated_points, color="red" )
    plt.xlabel("Radial Distance (lyr)")
    plt.ylabel("Tangential Velocity (km/s)")
    myMass = np.round(M33.calculate_mass(M33.rho_0, M33.alpha_0, M33.h0)/1E10,2)
    gasMass = np.round(M33.calculate_mass(M33.rho_1, M33.alpha_1, M33.h0)/1E10,2)
    plt.title("M33 Galaxy (z={}) Rotation Curve \n Luminous Mass {}E10 SunMass \n Gas Mass {}E10 SunMass".format(M33.redshift, myMass, gasMass))
    plt.xlim(0,np.max(M33.x_rotation_points))
    plt.ylim(0,np.max(M33.v_rotation_points))
    plt.show()


def move_rotation_curve(rotation_curve, z1=0.0, z2=20.0):
    rescaling_factor=(1+z2)/(1+z1)
    result  = np.zeros(np.shape(rotation_curve))
    result[:,0]=rotation_curve[:,0]/rescaling_factor
    result[:,1]=rotation_curve[:,1]*rescaling_factor
    return result



def calculate_density_parameters(redshift):
    # Fitting coefficients for log(rho_0) versus log(r4d):
    # Slope: -2.9791370770349763
    # Intercept: 4.663067724899548
    #
    # Fitting coefficients for log(alpha_0) versus log(r4d):
    # Slope: -0.9962401859242176
    # Intercept: -2.1843923428300345
    #
    # Fitting coefficients for log(rho_1) versus log(r4d):
    # Slope: -3.0038710671577102
    # Intercept: 2.6205959676388595
    #
    # Fitting coefficients for log(alpha_1) versus log(r4d):
    # Slope: -1.0037795630256436
    # Intercept: -3.509866645107434
    #
    # Fitting coefficients for log(h0) versus log(r4d):
    # Slope: 0.9868817849104266
    # Intercept: 4.015946542551611

    r4d = 14.01 / (1 + redshift)
    values = np.array([
        r4d ** (-2.9791370770349763) * 10 ** 4.663067724899548,
        r4d ** (-0.9962401859242176) * 10 ** (-2.1843923428300345),
        r4d ** (-3.0038710671577102) * 10 ** 2.6205959676388595,
        r4d ** (-1.0037795630256436) * 10 ** (-3.509866645107434),
        r4d ** (0.9868817849104266) * 10 ** 4.015946542551611
    ])
    return values


__all__=["plotRotationCurve","move_rotation_curve", "calculate_density_parameters"]