import pybind11_builtins as __pybind11_builtins
import pybind11_numpy as __pybind11_numpy
import numpy as np
import pybind11 as py


def density_wrapper(rho_0, alpha_0, rho_1, alpha_1, r, z):
    rho_0 *= 1.4171253E27
    rho_1 *= 1.4171253E27
    density = rho_0 * np.exp(-alpha_0 * r) + rho_1 * np.exp(-alpha_1 * r)
    return density


class GalaxyWrapper(__pybind11_builtins.pybind11_object):
    def __init__(self, GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0, R_max, nr, nz, nr_sampling, nz_sampling, ntheta,
                 redshift=0.0, cuda=False):
        self.GalaxyMass = GalaxyMass
        self.rho_0 = rho_0
        self.alpha_0 = alpha_0
        self.rho_1 = rho_1
        self.alpha_1 = alpha_1
        self.h0 = h0
        self.R_max = R_max
        self.nr = nr
        self.nz = nz
        self.nr_sampling = nr_sampling
        self.nz_sampling = nz_sampling
        self.ntheta = ntheta
        self.redshift = redshift
        self.cuda = cuda
        self.galaxy = GalaxyWrapper.GalaxyWrapper(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0, R_max, nr, nz,
                                                  nr_sampling, nz_sampling, ntheta, redshift, cuda)


    def DrudePropagator(self, epoch, time_step_years, eta, temperature):
        if self.galaxy is None:
            raise ValueError("Galaxy not initialized")
        result = self.galaxy.DrudePropagator(epoch, time_step_years, eta, temperature)
        return np.array(result)

    def get_f_z(self, x, debug=False):
        if self.galaxy is None:
            raise ValueError("Galaxy not initialized")
        f_z_r, f_z_z = self.galaxy.get_f_z(x, debug)
        return np.array(f_z_r), np.array(f_z_z)

    def get_galaxy(self):
        return self.galaxy

    def print_density_parameters(self):
        if self.galaxy is None:
            raise ValueError("Galaxy not initialized")
        return [self.galaxy.rho_0, self.galaxy.alpha_0, self.galaxy.rho_1, self.galaxy.alpha_1, self.galaxy.h0]

    def print_rotation_curve(self):
        if self.galaxy is None:
            raise ValueError("Galaxy not initialized")
        rotation_curve = []
        for i in range(self.galaxy.n_rotation_points):
            point = [self.galaxy.x_rotation_points[i], self.galaxy.v_rotation_points[i]]
            rotation_curve.append(point)
        return rotation_curve

    def print_simulated_curve(self):
        if self.galaxy is None:
            raise ValueError("Galaxy not initialized")
        simulated_curve = []
        for i in range(self.galaxy.n_rotation_points):
            point = [self.galaxy.x_rotation_points[i], self.galaxy.v_simulated_points[i]]
            simulated_curve.append(point)
        return simulated_curve

    def read_galaxy_rotation_curve(self, arg0):
        if self.galaxy is None:
            raise ValueError("Galaxy not initialized")
        vec = []
        for i in range(arg0.shape[0]):
            vec.append([arg0[i, 0], arg0[i, 1]])
        self.galaxy.read_galaxy_rotation_curve(vec)

    def simulate_rotation_curve(self):
        if self.galaxy is None:
            raise ValueError("Galaxy not initialized")
        result = self.galaxy.simulate_rotation_curve()
        return np.array(result)

    def get_redshift(self):
        return self.redshift

    def get_R_max(self):
        return self.R_max

    def get_nz_sampling(self):
        return self.nz_sampling

    def get_nr_sampling(self):
        return self.nr_sampling

    def get_nz(self):
        return self.nz

    def get_nr(self):
        return self.nr

    def get_alpha_0(self):
        return self.alpha_0

    def get_alpha_1(self):
        return self.alpha_1

    def get_rho_0(self):
        return self.rho_0

    def get_rho_1(self):
        return self.rho_1

    def get_h0(self):
        return self.h0


def calculate_mass(rho, alpha, h0):
    return rho * alpha * h0


def makeNumpy(result):
    nrows = len(result)
    ncols = len(result[0])
    data = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            data[i, j] = result[i][j]
    return data


pybind11_module_def_hugalaxy = {
    'density_wrapper': density_wrapper,
    'GalaxyWrapper': GalaxyWrapper,
    'calculate_mass': calculate_mass,
    'makeNumpy': makeNumpy,
}

m = py.module('hugalaxy.HU_Galaxy_GalaxyWrapper', pybind11_module_def_hugalaxy)
