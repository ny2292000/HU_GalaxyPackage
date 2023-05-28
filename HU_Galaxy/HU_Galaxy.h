#ifndef GALAXYWRAPPER_H
#define GALAXYWRAPPER_H

//
// Created by mp74207 on 4/22/23.
//
#pragma once

#include <Python.h>
#include <nlopt.hpp>
#include <vector>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "Galaxy.h"

namespace py = pybind11;

py::array_t<double> density_wrapper(double rho_0, double alpha_0, double rho_1, double alpha_1,
                                    const py::array_t<double>& r, const py::array_t<double>& z);

py::array_t<double> makeNumpy(const std::vector<std::vector<double>>& result);




class GalaxyWrapper {
public:
    GalaxyWrapper(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
                  double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift = 0.0);

    py::array_t<double> DrudePropagator(double epoch, double time_step_years, double eta, double temperature);

    std::pair<py::array_t<double>, py::array_t<double>> get_f_z(const std::vector<double>&x, bool debug = false);

    void read_galaxy_rotation_curve(py::array_t<double, py::array::c_style | py::array::forcecast> vin);

    py::list print_rotation_curve();

    py::list print_simulated_curve();

    py::list print_density_parameters();

    py::array_t<double> simulate_rotation_curve();

    // Getter member functions
    double get_redshift() const;
    double get_R_max() const;
    int get_nz_sampling() const;
    int get_nr_sampling() const;
    int get_nz() const;
    int get_nr() const;
    double get_alpha_0() const;
    double get_alpha_1() const;
    double get_rho_0() const;
    double get_rho_1() const;
    double get_h0() const;
    const Galaxy& get_galaxy() const;

private:
    Galaxy galaxy;
};

#endif // GALAXYWRAPPER_H
