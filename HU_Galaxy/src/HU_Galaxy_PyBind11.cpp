//
// Created by mp74207 on 4/22/23.
//

#include <Python.h>
#include <nlopt.hpp>
#include <memory>
#include <thread>
#include <iostream>
#include <vector>
#include <cmath>
#include <future>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../../LibGalaxy/include/lib0.h"
#include "../../LibGalaxyClass/include/Galaxy.h"

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
    }


private:
    Galaxy galaxy;
};

PYBIND11_MODULE(HU_Galaxy_GalaxyWrapper, m) {
    py::class_<GalaxyWrapper>(m, "GalaxyWrapper")
            .def(py::init<double, double, double, double, double, double, double, int, int, int, int, int, double>(),
                 py::arg("GalaxyMass"), py::arg("rho_0"), py::arg("alpha_0"), py::arg("rho_1"), py::arg("alpha_1"), py::arg("h0"),
                 py::arg("R_max"), py::arg("nr"), py::arg("nz"), py::arg("nr_sampling"), py::arg("nz_sampling"), py::arg("ntheta"), py::arg("redshift") = 0.0)
            .def("get_f_z", &GalaxyWrapper::get_f_z, py::arg("x"), py::arg("debug") = false);
}
