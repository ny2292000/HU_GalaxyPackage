import numpy as np
from hugalaxy import GalaxyWrapper, calculate_mass, plotRotationCurve

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
nr = 300
nz = 100
ntheta = 180
nr_sampling = 103
nz_sampling = 104
R_max = 50000.0
x0 =  np.array([17.77398054090567, 0.0004746365353380392, 0.1489212156273614, 2.2261975386169744e-05, 141703.88829098945])
rho_0, alpha_0, rho_1, alpha_1, h0 = x0
GalaxyMass = 5E10

M33 = GalaxyWrapper(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0, R_max, nr, nz, nr_sampling, nz_sampling, ntheta, redshift, cuda=True)

M33.read_galaxy_rotation_curve(m33_rotational_curve)
v_sim = M33.simulate_rotation_curve()

print(M33.print_density_parameters())
myMass = calculate_mass(rho_0, alpha_0, h0)
gasMass = calculate_mass(rho_1, alpha_1, h0)
print( myMass, gasMass)

plotRotationCurve((M33))
a=1