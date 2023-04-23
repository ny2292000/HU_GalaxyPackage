import pandas as pd
import numpy as np
from HU_Galaxy.Galaxy import Galaxy


# Observed M33 Data
# m33 = pd.read_csv("./Figures/m33digitized.csv")

def density(rho_0, alpha_0, rho_1, alpha_1, r):
    return rho_0*np.exp(-alpha_0*r) + rho_1*np.exp(-alpha_1*r)


m33 = pd.DataFrame([
    [0,0],
    [1508.7187263078158, 38.67413645475929],
    [2873.388931008347, 55.65067058479735],
    [4116.7551175132685, 67.91063228902073],
    [5451.099317664899, 79.22689123713342],
    [6846.095526914331, 85.01733936311302],
    [8089.461713419256, 88.38242236113695],
    [9393.479909021986, 92.42115545346981],
    [10727.824109173616, 95.1120774743588],
    [11880.21228203184, 98.3426961125904],
    [13275.208491281272, 99.82047577495817],
    [14609.552691432898, 102.10708512738353],
    [18521.607278241085, 104.25023858227974],
    [22403.335860500374, 107.60643221913195],
    [26406.36846095525, 115.40965650282314],
    [30379.075056861242, 116.8787511571496],
    [34382.10765731613, 116.05663851361706],
    [38354.814253222125, 117.93004583640715],
    [42266.86884003031, 121.42090818618205],
    [46300.22744503409, 128.5501758458687],
    [50212.28203184227, 132.84966353257082] ]
    , columns= ["d", "v"])
m33.plot(x="d", y="v", xlim=[0,55000], ylim=[0,140],
         title="M33 Observed Rotation Curve",
         xlabel="distance (lyr)",
         ylabel="tangential velocity (km/s)")

M33_Distance = 3.2E6
Radius_Universe_4D = 14.03E9
redshift = M33_Distance / (Radius_Universe_4D - M33_Distance)
nr = 300
nz = 100
ntheta = 180
nr_sampling = 103
nz_sampling = 104
R_max = 50000.0
alpha_0 = 0.00042423668409927005
rho_0 = 12.868348904393013
alpha_1 = 2.0523892233327836e-05
rho_1 = 0.13249804158174094
h0 = 156161.88949004377
GalaxyMass = 5E10
pi= 3.141592653589793238
r = np.linspace(1, R_max, nr)
# r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr)
z = np.linspace(-h0 / 2.0, h0 / 2.0, nz)
rho = density(rho_0, alpha_0, rho_1, alpha_1, r)
theta = np.linspace(0, 2 * pi, ntheta)
M33 = Galaxy(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0,
           R_max, nr, nz, nr_sampling, nz_sampling, ntheta, redshift)

x0 = np.array([rho_0, alpha_0, rho_1, alpha_1, h0])

a,b = M33.get_f_z(x0, debug=True)
print(a)
