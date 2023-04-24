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
                  double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift = 0.0);

    std::pair<py::array_t<double>, py::array_t<double>> get_f_z(const std::vector<double> &x, bool debug = false) ;


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
