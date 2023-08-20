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



py::array_t<double> makeNumpy_2D(const std::vector<std::vector<double>>& result);
py::array_t<double> makeNumpy_1D(const std::vector<double>& result);
py::array_t<double> calculate_density_parameters_py(double redshift);
py::array_t<double> move_rotation_curve_py(const py::array_t<double>& rotation_curve_py, double z1, double z2);
std::vector<double> to_std_vector(const pybind11::array_t<double>& input);

class GalaxyWrapper {
public:
    GalaxyWrapper(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
                  double R_max, int nr, int nz, int ntheta, double redshift = 0.0, int GPU_ID=0,
                  bool cuda=false, bool taskflow=false, double xtol_rel=1E-6, int max_iter=5000);

    py::list DrudePropagator(py::array_t<double>& redshifts, double deltaTime, double eta, double temperature);

    void DrudeGalaxyFormation(py::array_t<double> &epochs, double eta,
                              double temperature, const py::str &filename);

    void FreeFallGalaxyFormation(py::array_t<double> &epochs,  const py::str &filename);

    std::pair<py::array_t<double>, py::array_t<double>> get_f_z(const std::vector<std::vector<double>> &rho_, bool calc_vel,  double height);

    void set_rho_py(const py::array_t<double>& rho_py);
    [[nodiscard]] py::array_t<double> get_rho_py() const;

    [[nodiscard]] py::array_t<double> get_v_simulated_points() const;
    void set_v_simulated_points(const py::array_t<double>& arr);
    void read_galaxy_rotation_curve(py::array_t<double, py::array::c_style | py::array::forcecast> vin);
    [[nodiscard]] py::array_t<double> density_wrapper(double rho_0, double alpha_0, double rho_1, double alpha_1,
                                        const py::array_t<double>& r, const py::array_t<double>& z) const;
    py::array_t<double> density_wrapper_internal();
    void set_cuda(bool value);
    bool get_cuda  () const;
    void set_taskflow(bool value);
    bool get_taskflow  () const;
    int get_GPU_ID() const;
    void set_GPU_ID(int value);
    double calculate_total_mass();
    double calculate_mass(double rho, double alpha, double h);
    void  move_galaxy_redshift (double redshift);

    const py::array_t<double> calculate_rotational_velocity(py::array_t<double> rho_py, double height=0.0);
    const py::array_t<double> calculate_rotational_velocity_internal ();

    py::list print_simulated_curve();

    py::list print_density_parameters();

    py::array_t<double> simulate_rotation_curve();

    // Setter member functions
    void set_redshift(double redshift) ;
    void set_R_max(double R_max);
    void set_nz(int nz) ;
    void set_nr(int nr) ;
    void set_alpha_0(double alpha_0) ;
    void set_alpha_1(double alpha_1) ;
    void set_rho_0(double rho_0);
    void set_rho_1(double rho_1);
    void set_h0(double h0);
    void set_xtol_rel(double_t value);
    void set_max_iter(int value);
    // Getter member functions
    [[nodiscard]] double get_redshift() const;
    [[nodiscard]] double get_R_max() const;
    [[nodiscard]] int get_nz() const;
    [[nodiscard]] int get_nr() const;
    [[nodiscard]] double get_alpha_0() const;
    [[nodiscard]] double get_alpha_1() const;
    [[nodiscard]] double get_rho_0() const;
    [[nodiscard]] double get_rho_1() const;
    [[nodiscard]] double get_h0() const;
    [[nodiscard]] double_t get_xtol_rel() const;
    [[nodiscard]] int get_max_iter() const;
    [[nodiscard]] const galaxy& get_galaxy() const;
    galaxy& get_galaxy();
    [[nodiscard]] py::array_t<double> get_r() const ;
    void set_r(const py::array_t<double>& arr) ;
    [[nodiscard]] py::array_t<double> get_z() const ;
    void set_z(const py::array_t<double>& arr) ;
    [[nodiscard]] py::array_t<double> get_dv0() const ;
    void set_dv0(const py::array_t<double>& arr) ;
    [[nodiscard]] py::array_t<double> get_x_rotation_points() const;
    void set_x_rotation_points(const py::array_t<double>& arr) ;
    [[nodiscard]] py::array_t<double> get_v_rotation_points() const;
    void set_v_rotation_points(const py::array_t<double>& arr) ;
    [[nodiscard]] py::array_t<double> get_costheta() const ;
    void set_costheta(const py::array_t<double>& arr) ;
    [[nodiscard]] py::array_t<double> get_sintheta() const ;
    void set_sintheta(const py::array_t<double>& arr) ;

    void set_rotation_curve(const py::array_t<double>& arr) ;
    [[nodiscard]] py::array_t<double> get_rotation_curve() const ;
    void recalculate_masses();
    void recalculate_density();
    py::array_t<double> calibrate_df (py::array_t<double> vin, double redshift, int range_);
    double calculate_mass_gaussian(double rho, double alpha, double h);

private:
    galaxy galaxy_;
};

#endif // GALAXYWRAPPER_H
