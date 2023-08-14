import pandas as pd
import numpy as np

def calibrate_density_coefficients(m33_rotational_curve, redshift, M33):
    Radius_4D = 14.01
    data = M33.calibrate_df(m33_rotational_curve, redshift)
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
        print(f'Fitting coefficients for log({column}) versus log(r4d):')
        print('pow(r4d, {}) * pow(10, {}}),'.format(coeffs[0],coeffs[1]))
        print()
    return fitting_coeffs