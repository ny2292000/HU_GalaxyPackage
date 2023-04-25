//
// Created by mp74207 on 4/22/23.
//

#include <Python.h>
#include <nlopt.hpp>
#include <vector>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Galaxy.h"

namespace py = pybind11;


class GalaxyWrapper {
public:
    GalaxyWrapper(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
                  double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift = 0.0)
            : galaxy(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0, R_max, nr, nz, nr_sampling, nz_sampling, ntheta, redshift) {}

    std::pair<py::array_t<double>, py::array_t<double>> get_f_z(const std::vector<double> &x, bool debug = false) {
        auto f_z_pair = galaxy.get_f_z(x, debug);

        // Get the dimensions of the f_z_r and f_z_z arrays
        size_t rows = f_z_pair.first.size();
        size_t cols = f_z_pair.first[0].size();

        // Flatten the 2D f_z_r and f_z_z vectors
        std::vector<double> f_z_r_flat;
        std::vector<double> f_z_z_flat;
        for (size_t i = 0; i < rows; ++i) {
            f_z_r_flat.insert(f_z_r_flat.end(), f_z_pair.first[i].begin(), f_z_pair.first[i].end());
            f_z_z_flat.insert(f_z_z_flat.end(), f_z_pair.second[i].begin(), f_z_pair.second[i].end());
        }

        // Convert the result to NumPy arrays
        py::array_t<double> f_z_r({rows, cols}, f_z_r_flat.data());
        py::array_t<double> f_z_z({rows, cols}, f_z_z_flat.data());

        return std::make_pair(f_z_r, f_z_z);
    };


    std::vector<std::vector<double>> print_rotation_curve() {
        std::vector<std::vector<double>> rotation_curve;
        for (int i = 0; i < galaxy.n_rotation_points; i++) {
            std::vector<double> point{galaxy.x_rotation_points[i], galaxy.v_rotation_points[i]};
            rotation_curve.push_back(point);
        }
        return rotation_curve;
    };

    py::list print_simulated_curve() {
        py::list simulated_curve;
        for (int i = 0; i < galaxy.n_rotation_points; i++) {
            py::list point;
            point.append(galaxy.x_rotation_points[i]);
            point.append(galaxy.v_simulated_points[i]);
            simulated_curve.append(point);
        }
        return simulated_curve;
    };

    py::list print_density_parameters() {
        py::list density_params;
        density_params.append(galaxy.rho_0);
        density_params.append(galaxy.alpha_0);
        density_params.append(galaxy.rho_1);
        density_params.append(galaxy.alpha_1);
        density_params.append(galaxy.h0);
        return density_params;
    };



    py::array_t<double> simulate_rotation_curve() {
        // Calculate density at all radii
        std::vector<double> x0{galaxy.rho_0, galaxy.alpha_0, galaxy.rho_1, galaxy.alpha_1, galaxy.h0};
        int max_iter = 1000;
        double xtol_rel = 1e-6;
        std::vector<double> xout = nelder_mead(x0, galaxy, max_iter, xtol_rel);
        galaxy.rho_0 = xout[0];
        galaxy.alpha_0 = xout[1];
        galaxy.rho_1 = xout[2];
        galaxy.alpha_1 = xout[3];
        galaxy.h0 = xout[4];
        galaxy.rho = density(galaxy.rho_0, galaxy.alpha_0, galaxy.rho_1, galaxy.alpha_1, galaxy.r);
        // Calculate rotational velocity at all radii
        galaxy.v_simulated_points = calculate_rotational_velocity(galaxy.redshift, galaxy.dv0, galaxy.x_rotation_points,
                                                                  galaxy.r, galaxy.z, galaxy.costheta, galaxy.sintheta,
                                                                  galaxy.rho, false);
        double *data = galaxy.v_simulated_points.data(); // Get a pointer to the underlying data
        std::size_t size = galaxy.v_simulated_points.size(); // Get the size of the vector
        py::array_t<double> result(size, data);
        return result;
    }
private:
    Galaxy galaxy;
};

PYBIND11_MODULE(galaxy_wrapper, m) {
    py::class_<GalaxyWrapper>(m, "GalaxyWrapper")
            .def(py::init<double, double, double, double, double, double, double, int, int, int, int, int, double>(),
                 py::arg("GalaxyMass"), py::arg("rho_0"), py::arg("alpha_0"), py::arg("rho_1"), py::arg("alpha_1"), py::arg("h0"),
                 py::arg("R_max"), py::arg("nr"), py::arg("nz"), py::arg("nr_sampling"), py::arg("nz_sampling"), py::arg("ntheta"), py::arg("redshift") = 0.0)
            .def("get_f_z", &GalaxyWrapper::get_f_z, py::arg("x"), py::arg("debug") = false)
            .def("print_rotation_curve", &GalaxyWrapper::print_rotation_curve)
            .def("print_simulated_curve", &GalaxyWrapper::print_simulated_curve)
            .def("simulate_rotation_curve", &GalaxyWrapper::simulate_rotation_curve)
            .def("print_density_parameters", &GalaxyWrapper::print_density_parameters);
}

