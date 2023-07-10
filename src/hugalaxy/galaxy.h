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
#include <torch/torch.h>

const double Radius_4D = 14.01E9;
const double Age_Universe = 14.01E9;



std::vector<std::vector<double>> density(double rho_0, double alpha_0, double rho_1, double alpha_1,
                                         const std::vector<double>& r, const std::vector<double>& z);


class galaxy {
public:
    galaxy(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
           double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift, int GPU_ID,
           bool cuda=false, bool debug=false, double xtol_rel=1E-6, int max_iter=5000);
    ~galaxy();

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    get_f_z(const std::vector<double> &x, bool debug);
    void read_galaxy_rotation_curve(std::vector<std::array<double, 2>> vin);
    std::vector<double> simulate_rotation_curve();
    std::vector<double> move_galaxy(double redshift=0.0, bool recalc=false);
    std::vector<double> move_galaxy_redshift(double redshift);
    std::vector<std::vector<double>> print_rotation_curve();
    std::vector<std::vector<double>>print_simulated_curve();
    std::vector<double> print_density_parameters();
    std::vector<double> nelder_mead(const std::vector<double> &x0, galaxy &myGalaxy, int max_iter=1000, double xtol_rel=1E-6);
    void recalculate_density();
    void recalculate_masses();
    void recalculate_dv0();
    std::vector<std::vector<double>>  DrudePropagator(double redshift, double deltaTime, double eta, double temperature);
    double get_R_max() const { return R_max; };
    void set_R_max(double value) { R_max = value; };
    int nr;
    int nz;
    int ntheta;
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
    double original_redshift;
    int GPU_ID;
    bool cuda;
    bool debug;
    double GalaxyMass;
    int max_iter;
    double xtol_rel;
    std::vector<double> r;
    std::vector<double> dv0;
    std::vector<double> z;
    std::vector<double> r_sampling;
    std::vector<double> z_sampling;
    std::vector<double> theta;
    std::vector<double> costheta;
    std::vector<double> sintheta;
    std::vector<double> rotational_velocity_;
    std::vector<double> x_rotation_points;
    int n_rotation_points;
    std::vector<double> v_rotation_points;
    std::vector<double> v_simulated_points;
    std::vector<std::vector<double>> current_masses;
    std::vector<std::vector<double>> rho;
};


static double error_function(const std::vector<double> &x, galaxy &myGalaxy);


#endif // GALAXY_H
