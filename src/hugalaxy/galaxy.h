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


class galaxy {
public:
    galaxy(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
           double R_max, int nr, int nz, int ntheta, double redshift, int GPU_ID,
           bool cuda= false, bool taskflow= false, double xtol_rel= 1E-6, int max_iter= 5000);
    ~galaxy();

    void DrudeGalaxyFormation(std::vector<double> epochs,  std::vector<double> redshifts,
                               double eta,
                               double temperature,
                               std::string filename);

    std::vector<std::vector<double>> density(double rho_0, double alpha_0, double rho_1, double alpha_1,
                                             const std::vector<double>& r, const std::vector<double>& z) const;
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    get_f_z(const std::vector<std::vector<double>> &rho_, bool calc_vel,  const double height) const;
    std::vector<double> calculate_rotational_velocity(const std::vector<std::vector<double>> &rho, const double height=0.0) const;
    std::vector<double> calculate_rotational_velocity_internal() const;
    double calculate_mass(double rho, double alpha, double h);
    std::vector<double> creategrid(double rho_0, double alpha_0, double rho_1, double alpha_1, int n);
    void read_galaxy_rotation_curve(std::vector<std::array<double, 2>> vin);
    std::vector<double> simulate_rotation_curve();
    void move_galaxy_redshift(double redshift);
    std::vector<std::vector<double>> print_rotation_curve();
    std::vector<double> print_density_parameters();
    std::vector<double> nelder_mead(const std::vector<double> &x0, int max_iter=1000, double xtol_rel=1E-6);
    void recalculate_density();
    void recalculate_masses();
    double calculate_total_mass();
    void recalculate_dv0();
    std::vector<std::vector<double>>  DrudePropagator(double redshift, double deltaTime, double eta, double temperature);
    double get_R_max() const { return R_max; };
    void set_R_max(double value) { R_max = value; };
    long unsigned nr;
    long unsigned nz;
    int ntheta;
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
    bool taskflow_;
    double GalaxyMass;
    int max_iter;
    double xtol_rel;
    std::vector<double> r;
    std::vector<double> dv0;
    std::vector<double> z;
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

    void density_internal();
};


static double error_function(const std::vector<double> &x, galaxy &myGalaxy);


#endif // GALAXY_H
