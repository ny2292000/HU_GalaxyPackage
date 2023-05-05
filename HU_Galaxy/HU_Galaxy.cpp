//
// Created by mp74207 on 4/22/23.
//
//#pragma once

#include <Python.h>
#include <nlopt.hpp>
#include <vector>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Galaxy.h"

namespace py = pybind11;

#include <pybind11/numpy.h>
#include <vector>
#include <iostream>

namespace py = pybind11;

py::array_t<double> makeNumpy(const std::vector<std::vector<double>>& result) {
    // Get the dimensions of the data
    size_t nrows = result.size();
    size_t ncols = result[0].size();

    // Allocate a buffer to hold the data
    auto data = new double[nrows * ncols];
    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            data[i * ncols + j] = result[i][j];
        }
    }

    // Create a new NumPy array from the data
    auto capsule = py::capsule(data, [](void* ptr) { delete[] static_cast<double*>(ptr); });
    return py::array_t<double>({nrows, ncols}, {ncols * sizeof(double), sizeof(double)}, data, capsule);
}




class GalaxyWrapper {
public:
    GalaxyWrapper(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
                  double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift = 0.0)
            : galaxy(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0, R_max, nr, nz, nr_sampling, nz_sampling, ntheta, redshift) {}

    py::array_t<double> DrudePropagator(double epoch, double time_step_years, double eta, double temperature) {
        auto result = galaxy.DrudePropagator(epoch, time_step_years, eta, temperature);
        // Convert the 2D vector into a NumPy array using func
        return makeNumpy(result);
    }


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

    void read_galaxy_rotation_curve(py::array_t<double, py::array::c_style | py::array::forcecast> vin) {
        auto buf = vin.request();
        if (buf.ndim != 2 || buf.shape[1] != 2)
            throw std::runtime_error("Input should be a 2D array with 2 columns");

        std::vector<std::array<double, 2>> vec(buf.shape[0]);
        auto ptr = static_cast<double *>(buf.ptr);
        for (ssize_t i = 0; i < buf.shape[0]; ++i) {
            vec[i][0] = ptr[i * buf.shape[1]];
            vec[i][1] = ptr[i * buf.shape[1] + 1];
        }
        galaxy.read_galaxy_rotation_curve(vec);
    }

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
        // Get the galaxy object

        // Calculate density at all radii
        std::vector<double> xout = galaxy.simulate_rotation_curve();

        // Convert the result to a NumPy array
        double *data = xout.data(); // Get a pointer to the underlying data
        std::size_t size = xout.size(); // Get the size of the vector
        py::array_t<double> result(size, data);

        return result;
    }

    // Getter member functions
    double get_redshift() const { return galaxy.redshift; }
    double get_R_max() const { return galaxy.R_max; }
    int get_nz_sampling() const { return galaxy.nz_sampling; }
    int get_nr_sampling() const { return galaxy.nr_sampling; }
    int get_nz() const { return galaxy.nz; }
    int get_nr() const { return galaxy.nr; }
    double get_alpha_0() const { return galaxy.alpha_0; }
    double get_alpha_1() const { return galaxy.alpha_1; }
    double get_rho_0() const { return galaxy.rho_0; }
    double get_rho_1() const { return galaxy.rho_1; }
    double get_h0() const { return galaxy.h0; }
    const Galaxy& get_galaxy() const { return galaxy; }

private:
    Galaxy galaxy;
};

PYBIND11_MODULE(HU_Galaxy, m) {
    py::class_<GalaxyWrapper>(m, "GalaxyWrapper")
            .def(py::init<double, double, double, double, double, double, double, int, int, int, int, int, double>(),
                 py::arg("GalaxyMass"), py::arg("rho_0"), py::arg("alpha_0"), py::arg("rho_1"), py::arg("alpha_1"), py::arg("h0"),
                 py::arg("R_max"), py::arg("nr"), py::arg("nz"), py::arg("nr_sampling"), py::arg("nz_sampling"), py::arg("ntheta"), py::arg("redshift") = 0.0)
            .def("DrudePropagator", &GalaxyWrapper::DrudePropagator, py::arg("epoch"), py::arg("time_step_years"), py::arg("eta"), py::arg("temperature"),
                 "Propagate the mass distribution in a galaxy using the Drude model")
            .def("get_galaxy", &GalaxyWrapper::get_galaxy)
            .def_property_readonly("redshift", &GalaxyWrapper::get_redshift)
            .def_property_readonly("R_max", &GalaxyWrapper::get_R_max)
            .def_property_readonly("nz_sampling", &GalaxyWrapper::get_nz_sampling)
            .def_property_readonly("nr_sampling", &GalaxyWrapper::get_nr_sampling)
            .def_property_readonly("nz", &GalaxyWrapper::get_nz)
            .def_property_readonly("nr", &GalaxyWrapper::get_nr)
            .def_property_readonly("alpha_0", &GalaxyWrapper::get_alpha_0)
            .def_property_readonly("alpha_1", &GalaxyWrapper::get_alpha_1)
            .def_property_readonly("rho_0", &GalaxyWrapper::get_rho_0)
            .def_property_readonly("rho_1", &GalaxyWrapper::get_rho_1)
            .def_property_readonly("h0", &GalaxyWrapper::get_h0)
            .def("read_galaxy_rotation_curve", &GalaxyWrapper::read_galaxy_rotation_curve)
            .def("get_f_z", &GalaxyWrapper::get_f_z, py::arg("x"), py::arg("debug") = false)
            .def("print_rotation_curve", &GalaxyWrapper::print_rotation_curve)
            .def("print_simulated_curve", &GalaxyWrapper::print_simulated_curve)
            .def("simulate_rotation_curve", &GalaxyWrapper::simulate_rotation_curve)
            .def("print_density_parameters", &GalaxyWrapper::print_density_parameters);
    m.def("calculate_mass", &calculate_mass, py::arg("rho"), py::arg("alpha"), py::arg("h0"), "A function to calculate the mass of the galaxy");
}


