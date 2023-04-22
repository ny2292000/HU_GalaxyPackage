import numpy as np
from HU_Galaxy_PyBind11 import Galaxy


class Galaxy:
    def __init__(self, GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0,
                 R_max, nr, nz, nr_sampling, nz_sampling, ntheta, redshift=0.0):
        self._galaxy = Galaxy(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0,
                                          R_max, nr, nz, nr_sampling, nz_sampling, ntheta, redshift)
    # Rest of the code


    def get_f_z(self, x, debug=False):
        # Convert input x to NumPy array
        x_np = np.array(x)

        # Call the C++ function
        f_z_r, f_z_z = self._galaxy.get_f_z(x_np.tolist(), debug)

        # Convert the result to NumPy arrays
        f_z_r_np = np.array(f_z_r)
        f_z_z_np = np.array(f_z_z)

        # Return a tuple of the two NumPy arrays
        return f_z_r_np, f_z_z_np
