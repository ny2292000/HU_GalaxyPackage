from HU_Galaxy.galaxy_wrapper import GalaxyWrapper

# Create an instance of GalaxyWrapper with appropriate arguments
# Replace the arguments below with the values you want to use
galaxy = GalaxyWrapper(
    GalaxyMass=1.0,
    rho_0=2.0,
    alpha_0=3.0,
    rho_1=4.0,
    alpha_1=5.0,
    h0=6.0,
    R_max=7.0,
    nr=8,
    nz=9,
    nr_sampling=10,
    nz_sampling=11,
    ntheta=12,
    redshift=0.0,
)

# Call the get_f_z method with required arguments
# Replace the sample_vector with the actual values you want to use
sample_vector = [1.0, 2.0, 3.0]
result = galaxy.get_f_z(sample_vector, debug=False)

print("Result of get_f_z:", result)
