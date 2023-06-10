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

// Returns a vector of zeros with the given size
std::vector<double> zeros_1(int size) ;

std::vector<std::vector<double>> zeros_2(int nr, int nz) ;

std::vector<std::vector<double>> density(double rho_0, double alpha_0, double rho_1, double alpha_1,
                                         const std::vector<double>& r, const std::vector<double>& z);

void print(const std::vector<double> a) ;

void print_2D(const std::vector<std::vector<double>>& a);

double calculate_mass(double rho,double alpha, double h) ;

std::vector<std::vector<double>> calculate_tau(double effective_cross_section,
                       const std::vector<std::vector<double>>& local_density, double temperature);

std::vector<double> costhetaFunc(const std::vector<double> &theta);

std::vector<double> sinthetaFunc(const std::vector<double> &theta) ;

std::vector<double> linspace(double start, double end, size_t points) ;

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_all_torch(double redshift,
              const std::vector<double> &dv0_in,
              const std::vector<double> &r_sampling_in,
              const std::vector<double> &z_sampling_in,
              const std::vector<double> &r_in,
              const std::vector<double> &z_in,
              const std::vector<double> &costheta_in,
              const std::vector<double> &sintheta_in,
              const std::vector<std::vector<double>> &rho_in,
              bool debug);

std::pair<torch::Tensor, torch::Tensor> get_g_torch(
        double r_sampling_ii,
        double z_sampling_jj,
        const torch::Tensor& G,
        const torch::Tensor& dv0,
        const torch::Tensor& r,
        const torch::Tensor& z,
        const torch::Tensor& costheta,
        const torch::Tensor& sintheta,
        const torch::Tensor& rho,
        bool debug);

// # CPU functions
std::pair<double, double> get_g_cpu(double r_sampling_ii, double z_sampling_jj, double G,
                                    const std::vector<double> &dv0, const std::vector<double> &r,
                                    const std::vector<double> &z, const std::vector<double> &costheta,
                                    const std::vector<double> &sintheta, const std::vector<std::vector<double>> &rho, bool debug) ;

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_all_g(double redshift, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
          const std::vector<double> &z_sampling,
          const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
          const std::vector<double> &sintheta, const std::vector<std::vector<double>> &rho, bool debug) ;


std::vector<double> calculate_rotational_velocity(double redshift, const std::vector<double> &dv0,
                                                  std::vector<double> r_sampling,
                                                  const std::vector<double> &r,
                                                  const std::vector<double> &z,
                                                  const std::vector<double> &costheta,
                                                  const std::vector<double> &sintheta,
                                                  const std::vector<std::vector<double>> &rho, bool debug, bool cuda) ;


std::vector<double> creategrid(double rho_0, double alpha_0, double rho_1, double alpha_1, int n) ;



class Galaxy {
public:
    Galaxy(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
           double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift, bool cuda);
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
    bool cuda;
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





#endif // GALAXY_H
