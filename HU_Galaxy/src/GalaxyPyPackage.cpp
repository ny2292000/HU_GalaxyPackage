#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "../../LibGalaxy/include/Galaxy.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Expose your C++ functions and classes to Python
PYBIND11_MODULE(HU_Galaxy_PyBind11, m) {
    m.doc() = "My Python module"; // Add a docstring for your module

    // Expose your C++ functions and classes to Python
    py::class_<Galaxy>(m, "HU_GalaxyModel")
            .def(py::init<double, double, double, double, double, double, double, int, int, int, int, int, double>())
            .def("get_f_z", &Galaxy::get_f_z, py::arg("x"), py::arg("debug")=false);
}
