#ifndef GALAXY_H
#define GALAXY_H
#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define _USE_MATH_DEFINES
#include <Python.h>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <nlopt.hpp>
#include <future>
#include <utility>
#include "torch/torch.h"

std::vector<std::vector<double>> density(double rho_0, double alpha_0, double rho_1, double alpha_1,
                                         const std::vector<double>& r, const std::vector<double>& z);


class Galaxy {
public:
    Galaxy(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
           double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift, int GPU_ID, bool cuda, bool debug);
    ~Galaxy();

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    get_f_z(const std::vector<double> &x, bool debug);
    void read_galaxy_rotation_curve(std::vector<std::array<double, 2>> vin);
    std::vector<double> simulate_rotation_curve();
    std::vector<std::vector<double>> print_rotation_curve();
    std::vector<std::vector<double>>print_simulated_curve();
    std::vector<double> print_density_parameters();
    std::vector<double> nelder_mead(const std::vector<double> &x0, Galaxy &myGalaxy, int max_iter=1000, double xtol_rel=1E-6);
    void recalculate_density(const std::vector<std::vector<double>>& currentMasses);
    std::vector<std::vector<double>>  DrudePropagator(double epoch, double time_step, double eta, double temperature);
    double get_R_max() const { return R_max; };
    void set_R_max(double value) { R_max = value; };
    int nr;
    int nz;
    int nr_sampling;
    int nz_sampling;
    double R_max;
    double alpha_0;
    double rho_0;
    double alpha_1;
    double rho_1;
    double h0;
    double dz;
    double dtheta;
    double redshift;
    int GPU_ID;
    bool cuda;
    bool debug;
    double GalaxyMass;
    std::vector<double> r;
    std::vector<double> dv0;
    std::vector<double> z;
    std::vector<double> r_sampling;
    std::vector<double> z_sampling;
    std::vector<std::vector<double>> rho;
    std::vector<double> theta;
    std::vector<double> costheta;
    std::vector<double> sintheta;
    std::vector<double> rotational_velocity_;
    std::vector<double> x_rotation_points;
    int n_rotation_points;
    std::vector<double> v_rotation_points;
    std::vector<double> v_simulated_points;
};


static double error_function(const std::vector<double> &x, Galaxy &myGalaxy);


#endif // GALAXY_H
