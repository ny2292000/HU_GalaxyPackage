//
// Created by mp74207 on 4/17/23.
//
const double pi = 3.141592653589793;

#include <Python.h>
#include <nlopt.hpp>
#include <vector>
#include <array>
#include <utility>
#include <cmath>
#include <iostream>
#include <future>
#include "Galaxy.h"


std::vector<std::vector<double>> density(double rho_0, double alpha_0, double rho_1, double alpha_1,
                                         const std::vector<double>& r, const std::vector<double>& z) {
    unsigned int nr = r.size();
    unsigned int nz = z.size();
    std::vector<std::vector<double>> density_(nr, std::vector<double>(nz));

    // to kg/lyr^3
    rho_0 *= 1.4171253E27;
    rho_1 *= 1.4171253E27;

    for (unsigned int i = 0; i < nr; i++) {
        for (unsigned int j = 0; j < nz; j++) {
            double r_ij = std::sqrt(std::pow(r[i], 2) + std::pow(z[j], 2));
            density_[i][j] = rho_0 * std::exp(-alpha_0 * r_ij) + rho_1 * std::exp(-alpha_1 * r_ij);
        }
    }

    return density_;
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
    double Mtotal_si = calculate_mass(alpha_0, rho_0, h0);  // Mtotal in Solar Masses
    double error_mass = std::pow((myGalaxy.GalaxyMass - Mtotal_si) / myGalaxy.GalaxyMass, 2);
    bool debug = false;
    std::vector<std::vector<double>> rho = density(rho_0, alpha_0, rho_1, alpha_1, myGalaxy.r, myGalaxy.z);

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
//    std::cout << "Total Error = " << (error + error_mass) << "\n";
    return error + error_mass;
}

// Define the objective function wrapper
auto objective_wrapper = [](const std::vector<double> &x, std::vector<double> &grad, void *data) {
    Galaxy *myGalaxy = static_cast<Galaxy *>(data);
    return error_function(x, *myGalaxy);
};

// Returns a vector of zeros with the given size
std::vector<double> zeros_1(int size) {
    return std::vector<double>(size, 0.0);
}

std::vector<std::vector<double>> zeros_2(int nr, int nz) {
    std::vector<std::vector<double>> vec(nr, std::vector<double>(nz, 0.0));
    return vec;
}



void print(const std::vector<double> a) {
    std::cout << "The vector elements are : " << "\n";
    for (int i = 0; i < a.size(); i++)
        std::cout << std::scientific << a.at(i) << '\n';
}

void print_2D(const std::vector<std::vector<double>>& a) {
    std::cout << "The 2D vector elements are : " << "\n";
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            std::cout << std::scientific << a[i][j] << " ";
        }
        std::cout << '\n';
    }
}




double massCalcX(double alpha, double rho, double h, double x) {
    double factor = 0.0007126927557971729; // factor takes care of moving from rho as atom/cc to kg/lyr^3, with alpha = 1/lyr and h0 = in lyr div sun_mass
    double M_si = -2 * pi * h * rho * x * exp(-alpha * x) / alpha - 2 * pi * h * rho * exp(-alpha * x) / std::pow(alpha, 2) +
                  2 * pi * h * rho / std::pow(alpha, 2);
    M_si = M_si * factor;
    return M_si;
}


double calculate_mass(double alpha, double rho, double h) {
    double factor = 0.0007126927557971729; // factor takes care of moving from rho as atom/cc to kg/lyr^3, with alpha = 1/lyr and h0 = in lyr div sun_mass
    double Mtotal_si = 2 * pi * h * rho / std::pow(alpha, 2);
    Mtotal_si = Mtotal_si * factor;
    return Mtotal_si;
}




std::vector<double> costhetaFunc(const std::vector<double> &theta) {
    unsigned int points = theta.size();
    std::vector<double> res(points);
    for (unsigned int i = 0; i < points; i++) {
        res[i] = std::cos(theta[i]);
    }
    return res;
}

std::vector<double> sinthetaFunc(const std::vector<double> &theta) {
    unsigned int points = theta.size();
    std::vector<double> res(points);
    for (unsigned int i = 0; i < points; i++) {
        res[i] = std::sin(theta[i]);
    }
    return res;
}


std::vector<double> linspace(double start, double end, size_t points) {
    std::vector<double> res(points);
    double step = (end - start) / (points - 1);
    size_t i = 0;
    for (auto &e: res) {
        e = start + step * i++;
    }
    return res;
}




// # CPU functions
std::pair<double, double> get_g_cpu(double r_sampling_ii, double z_sampling_jj, double G,
                                    const std::vector<double> &dv0, const std::vector<double> &r,
                                    const std::vector<double> &z, const std::vector<double> &costheta,
                                    const std::vector<double> &sintheta, const std::vector<std::vector<double>> &rho, bool debug) {
    unsigned int nr = r.size();
    unsigned int nz = z.size();
    unsigned int ntheta = costheta.size();
    double radial_value = 0.0;
    double thisradial_value = 0.0;
    double vertical_value = 0.0;
    double thisvertical_value = 0.0;
    for (unsigned int i = 0; i < nr; i++) {
        for (unsigned int j = 0; j < nz; j++) {
            for (unsigned int k = 0; k < ntheta; k++) {
                double d_2 = (z[j] - z_sampling_jj) * (z[j] - z_sampling_jj) +
                             (r_sampling_ii - r[i] * sintheta[k]) * (r_sampling_ii - r[i] * sintheta[k])+
                             r[i] * r[i] * costheta[k] * costheta[k];
                double d_1 = sqrt(d_2);
                double d_3 = d_1 * d_1 * d_1;
                double commonfactor  = G * rho[i][j] * r[i] * dv0[i]  / d_3;
                if ( r[i] < r_sampling_ii) {
                    thisradial_value = commonfactor * (r_sampling_ii - r[i] * sintheta[k]);
                    radial_value += thisradial_value;
                }
                thisvertical_value = commonfactor * (z[j] - z_sampling_jj);
                vertical_value += thisvertical_value;
                if (debug) {
                    if (i==5 && j==5 && k == 5){
                        printf("CPU \n");
                        printf("The value of f_z is %e\n", thisradial_value);
                        printf("The value of f_z is %e\n", thisvertical_value);
                        printf("The value of distance is %fd\n", sqrt(d_1));
                        printf("The value of r[i] is %fd\n", r[i]);
                        printf("The value of z[j] is %fd\n", z[j]);
                        printf("The value of costheta is %fd\n", costheta[k]);
                        printf("The value of sintheta is %fd\n", sintheta[k]);
                        printf("The value of dv0 is %fd\n", dv0[i]);
                        printf("The value of rho is %e\n", rho[i][0]);
                        printf("The value of rsampling is %fd\n", r_sampling_ii);
                        printf("The value of zsampling is %fd\n", z_sampling_jj);
                        printf("The value of G is %e\n", G);
                    }
                }
            }
        }
    }
    return std::make_pair(radial_value, vertical_value);
}


std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_all_g(double redshift, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
          const std::vector<double> &z_sampling,
          const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
          const std::vector<double> &sintheta, const std::vector<std::vector<double>> &rho, bool debug) {
    double G = 7.456866768350099e-46 * (1 + redshift);
    std::vector<std::future<std::pair<double, double>>> futures;
    int nr = r_sampling.size();
    int nz = z_sampling.size();
    futures.reserve(nr * nz);

    // Spawn threads
    std::vector<std::vector<double>> f_z_radial = zeros_2(nr, nz);
    std::vector<std::vector<double>> f_z_vertical = zeros_2(nr, nz);
    for (unsigned int i = 0; i < nr; i++) {
        for (unsigned int j = 0; j < nz; j++) {
            futures.emplace_back(
                    std::async(get_g_cpu, r_sampling[i], z_sampling[j], G, dv0, r, z, costheta, sintheta, rho, debug));
        }
    }

    // Collect results and populate f_z_radial and f_z_non_radial
    for (unsigned int i = 0; i < nr; i++) {
        for (unsigned int j = 0; j < nz; j++) {
            auto result_pair = futures[i * nz + j].get();
            f_z_radial[i][j] = result_pair.first;
            f_z_vertical[i][j] = result_pair.second;
        }
    }

    // Combine the two f_z vectors into a pair of two-dimensional vector and return it
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z_combined{f_z_radial, f_z_vertical};
    return f_z_combined;
}

std::vector<double> calculate_rotational_velocity(double redshift, const std::vector<double> &dv0,
                                                  std::vector<double> r_sampling,
                                                  const std::vector<double> &r,
                                                  const std::vector<double> &z,
                                                  const std::vector<double> &costheta,
                                                  const std::vector<double> &sintheta,
                                                  const std::vector<std::vector<double>> &rho, bool debug) {
    int nr_sampling = r_sampling.size();
    double km_lyr = 9460730472580.8; //uu.lyr.to(uu.km)
    // Allocate result vector
    std::vector<double> z_sampling = {0.0};
    std::vector<double> v_r(nr_sampling);
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z = get_all_g(redshift, dv0, r_sampling, z_sampling, r, z,
                                                                                                  costheta, sintheta, rho, debug);
    // Calculate velocities
    double v_squared;
    for (int i = 0; i < nr_sampling; i++) {
        v_squared = f_z.first[i][0] * r_sampling[i] * km_lyr; // Access radial values from the pair (first element)
        v_r[i] = sqrt(v_squared); // 9460730777119.56 km

        // Debugging output
        if (debug) {
            std::cout << "r_sampling[" << i << "]: " << r_sampling[i] << std::endl;
            std::cout << "f_z.first[" << i << "][0]: " << f_z.first[i][0] << std::endl;
            std::cout << "v_squared: " << v_squared << std::endl;
            std::cout << "v_r[" << i << "]: " << v_r[i] << std::endl;
        }
    }
    // Return result
    return v_r;
}


std::vector<double> creategrid(double rho_0, double alpha_0, double rho_1, double alpha_1, int n) {
    if (alpha_1 > alpha_0) {
        double alpha_ = alpha_0;
        double rho_ = rho_0;
        alpha_0 = alpha_1;
        rho_0 = rho_1;
        alpha_1 = alpha_;
        rho_1 = rho_;
    }
    int n_range = 4;
    double r_max_1 = n_range / alpha_0;
    double r_max_2 = n_range / alpha_1;
    double M1 = calculate_mass(alpha_0, rho_0, 1.0);
    double M2 = calculate_mass(alpha_1, rho_1, 1.0);
    int n1 = M1 / (M1 + M2) * n;
    int n2 = M2 / (M1 + M2) * n;
    double r_min1 = 1.0;
    double r_min2 = r_max_1 + 1.0;

    // Define the grid of n points using a geometric sequence
    std::vector<double> r(n1 + n2);
    for (int i = 0; i < n1; i++) {
        r[i] = r_min1 * std::pow(r_max_1 / r_min1, i / (double) (n1 - 1));
    }
    for (int i = 0; i < n2; i++) {
        r[i + n1] = r_min2 * std::pow(r_max_2 / r_min2, i / (double) (n2 - 1));
    }
    return r;
}

// #############################################################################
// #############################################################################
// #############################################################################
// #############################################################################
// #############################################################################
// #############################################################################

Galaxy::Galaxy(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
               double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift)
        : R_max(R_max), nr(nr), nz(nz), nr_sampling(nr_sampling), nz_sampling(nz_sampling),
          alpha_0(alpha_0), rho_0(rho_0), alpha_1(alpha_1), rho_1(rho_1), h0(h0), redshift(redshift),
          GalaxyMass(GalaxyMass), n_rotation_points(0) {

    r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr);
    z = linspace(-h0 / 2.0, h0 / 2.0, nz);
    rho = density(rho_0, alpha_0, rho_1, alpha_1, r, z);
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

std::vector<std::vector<double>> Galaxy::recalculate_density(const std::vector<std::vector<double>>& currentMasses) const {
    std::vector<std::vector<double>> updatedDensity(nr, std::vector<double>(nz, 0.0));
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nz; j++) {
            updatedDensity[i][j] = currentMasses[i][j] / dv0[i];
        }
    }
    return updatedDensity;
};

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

// Define the Nelder-Mead optimizer
std::vector<double>
Galaxy::nelder_mead(const std::vector<double> &x0, Galaxy &myGalaxy, int max_iter, double xtol_rel) {
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

std::vector<double> Galaxy::simulate_rotation_curve() {
    // Calculate density at all radii
    std::vector<double> x0{rho_0, alpha_0, rho_1, alpha_1, h0};
    int max_iter = 1000;
    double xtol_rel = 1e-6;
    std::vector<double> xout = nelder_mead(x0, *this, max_iter, xtol_rel);
    rho_0 = xout[0];
    alpha_0 = xout[1];
    rho_1 = xout[2];
    alpha_1 = xout[3];
    h0 = xout[4];
    rho = density(rho_0, alpha_0, rho_1, alpha_1, r, z);
    // Calculate rotational velocity at all radii
    v_simulated_points = calculate_rotational_velocity(redshift, dv0, x_rotation_points,
                                                       r, z, costheta, sintheta,
                                                       rho, false);
    return v_simulated_points;
}

std::vector<std::vector<double>> Galaxy::print_rotation_curve() {
    std::vector<std::vector<double>> rotation_curve;
    for (int i = 0; i < n_rotation_points; i++) {
        std::vector<double> point{x_rotation_points[i], v_rotation_points[i]};
        rotation_curve.push_back(point);
    }
    return rotation_curve;
}

std::vector<std::vector<double>> Galaxy::print_simulated_curve() {
    std::vector<std::vector<double>> simulated_curve;
    for (int i = 0; i < n_rotation_points; i++) {
        std::vector<double> point{x_rotation_points[i], v_simulated_points[i]};
        simulated_curve.push_back(point);
    }
    return simulated_curve;
}

std::vector<double> Galaxy::print_density_parameters() {
    std::vector<double> density_params{rho_0, alpha_0, rho_1, alpha_1, h0};
    return density_params;
}


