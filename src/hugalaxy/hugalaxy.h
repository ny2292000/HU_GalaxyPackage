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
#include "galaxy.h"

namespace py = pybind11;

py::array_t<double> density_wrapper(double rho_0, double alpha_0, double rho_1, double alpha_1,
                                    const py::array_t<double>& r, const py::array_t<double>& z);

py::array_t<double> makeNumpy(const std::vector<std::vector<double>>& result);




class GalaxyWrapper {
public:
    GalaxyWrapper(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
                  double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift = 0.0, int GPU_ID=0, bool cuda=false, bool debug=false);

    py::array_t<double> DrudePropagator(double redshift, double time_step_years, double eta, double temperature);

    std::pair<py::array_t<double>, py::array_t<double>> get_f_z(const std::vector<double>&x, bool debug = false);

    void read_galaxy_rotation_curve(py::array_t<double, py::array::c_style | py::array::forcecast> vin);

    void setCuda(bool value);

    py::bool_ getCuda  () const;

    py::int_ getGPU_ID() const;

    // Setter for cuda
    void setGPU_ID(int value);

    py::list print_rotation_curve();

    py::list print_simulated_curve();

    py::list print_density_parameters();

    py::array_t<double> simulate_rotation_curve();

    // Setter member functions
    void set_redshift(double redshift) ;
    void set_R_max(double R_max);
    void set_nz_sampling(int nz_sampling);
    void set_nr_sampling(int nr_sampling) ;
    void set_nz(int nz) ;
    void set_nr(int nr) ;
    void set_alpha_0(double alpha_0) ;
    void set_alpha_1(double alpha_1) ;
    void set_rho_0(double rho_0);
    void set_rho_1(double rho_1);
    void set_h0(double h0) ;
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
    const galaxy& get_galaxy() const;
    py::array_t<double> get_r() const ;
    void set_r(const py::array_t<double>& arr) ;
    py::array_t<double> get_rotation_curve() const;
    void set_rotation_curve(const py::array_t<double>& arr) ;

private:
    galaxy galaxy_;
};

#endif // GALAXYWRAPPER_H