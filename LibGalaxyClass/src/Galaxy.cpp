//
// Created by mp74207 on 4/17/23.
//
const double pi = 3.141592653589793;

#include "../include/Galaxy.h"
#include <vector>
#include <array>
#include <utility>
#include <cmath>
#include <iostream>
#include <nlopt.hpp>

#include "../../LibGalaxy/include/lib0.h"



Galaxy::Galaxy(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
               double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift)
        : R_max(R_max), nr(nr), nz(nz), nr_sampling(nr_sampling), nz_sampling(nz_sampling),
          alpha_0(alpha_0), rho_0(rho_0), alpha_1(alpha_1), rho_1(rho_1), h0(h0), redshift(redshift),
          GalaxyMass(GalaxyMass), n_rotation_points(0) {

    r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr);
    z = linspace(-h0 / 2.0, h0 / 2.0, nz);
    rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
    theta = linspace(0, 2 * pi, ntheta);
    costheta = costhetaFunc(theta);
    sintheta = sinthetaFunc(theta);
    z_sampling = linspace(-h0 / 2.0, h0 / 2.0, nz);
    r_sampling = linspace(1, R_max, nr_sampling);
    dz = h0 / nz;
    double dtheta = 2 * pi / ntheta;
    dv0.resize(1);
    dv0[0] = 0.0;
    for (int i = 1; i < nr; i++) {
        dv0.push_back((r[i] - r[i - 1]) * dz * dtheta);
    }

    // Initialize empty vectors for f_z, x_rotation_points, and v_rotation_points
    f_z = std::vector<std::vector<double>>(nr, std::vector<double>(nz, 0));
    x_rotation_points = std::vector<double>();
    v_rotation_points = std::vector<double>();
}


Galaxy::~Galaxy() {};

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
Galaxy::get_f_z(const std::vector<double> &x, bool debug) {
    // Calculate the rotation velocity using the current values of x
    double rho_0 = x[0];
    double alpha_0 = x[1];
    double rho_1 = x[2];
    double alpha_1 = x[3];
    double h0 = x[4];
    // Calculate the total mass of the galaxy
    std::vector<double> r_sampling = this->x_rotation_points;
    std::vector<double> z_sampling;
//    std::vector<double> z_sampling = this->z;
    if (debug) {
        z_sampling = {this->h0 / 2.0};
    } else {
        z_sampling = {this->z};
    }
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z = get_all_g(redshift, dv0,
                                                                                                  r_sampling,
                                                                                                  z_sampling, r, z,
                                                                                                  costheta,
                                                                                                  sintheta, rho,
                                                                                                  debug);
    return f_z;
}

void Galaxy::read_galaxy_rotation_curve(std::vector<std::array<double, 2>> vin) {
    n_rotation_points = vin.size();
    this->x_rotation_points.clear();
    this->v_rotation_points.clear();
    for (const auto &row: vin) {
        this->x_rotation_points.push_back(row[0]); // Extract the first column (index 0)
        this->v_rotation_points.push_back(row[1]); // Extract the first column (index 0)
    }


}


// Define the function to be minimized
static double error_function(const std::vector<double> &x, Galaxy &myGalaxy) {
    // Calculate the rotation velocity using the current values of x
    double rho_0 = x[0];
    double alpha_0 = x[1];
    double rho_1 = x[2];
    double alpha_1 = x[3];
    double h0 = x[4];
    // Calculate the total mass of the galaxy
    double Mtotal_si = massCalc(alpha_0, rho_0, h0);  // Mtotal in Solar Masses
    double error_mass = std::pow((myGalaxy.GalaxyMass - Mtotal_si) / myGalaxy.GalaxyMass, 2);
    bool debug = false;
    std::vector<double> rho = density(rho_0, alpha_0, rho_1, alpha_1, myGalaxy.r);
    std::vector<double> vsim = calculate_rotational_velocity(myGalaxy.redshift, myGalaxy.dv0,
                                                             myGalaxy.x_rotation_points,
                                                             myGalaxy.r,
                                                             myGalaxy.z,
                                                             myGalaxy.costheta,
                                                             myGalaxy.sintheta,
                                                             rho,
                                                             debug);
    double error = 0.0;
    for (int i = 0; i < myGalaxy.n_rotation_points; i++) { error += std::pow((myGalaxy.v_rotation_points[i] - vsim[i]), 2); }
    std::cout << "Total Error = " << (error + error_mass) << "\n";
    return error + error_mass;
}

// Define the objective function wrapper
auto objective_wrapper = [](const std::vector<double> &x, std::vector<double> &grad, void *data) {
    Galaxy *myGalaxy = static_cast<Galaxy *>(data);
    return error_function(x, *myGalaxy);
};

// Define the Nelder-Mead optimizer
std::vector<double>
nelder_mead(const std::vector<double> &x0, Galaxy &myGalaxy, int max_iter, double xtol_rel) {
    nlopt::opt opt(nlopt::LN_NELDERMEAD, x0.size());
    opt.set_min_objective(objective_wrapper, &myGalaxy);
    opt.set_xtol_rel(xtol_rel);
    std::vector<double> x = x0;
    double minf;
    nlopt::result result = opt.optimize(x, minf);
    if (result < 0) {
        std::cerr << "nlopt failed: " << strerror(result) << std::endl;
    }
    return x;
}