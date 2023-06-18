//
// Created by mp74207 on 4/17/23.
//
#define _USE_MATH_DEFINES
#include <cmath>
#include <nlopt.hpp>
#include <vector>
#include <array>
#include <utility>
#include <iostream>
#include <cuda_runtime.h>
#include "Galaxy.h"
#include "tensor_utils.h"

const double Radius_4D = 14.01;


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
            density_[i][j] = rho_0 * std::exp(-alpha_0 * r[i]) + rho_1 * std::exp(-alpha_1 * r[i]);
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
    // Ensure that all parameters are positive
    if (rho_0 <= 0.0 || alpha_0 <= 0.0 || rho_1 <= 0.0 || alpha_1 <= 0.0 || h0 <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    // Calculate the total mass of the galaxy
    double Mtotal_si = calculate_mass(rho_0, alpha_0, h0);
    double error_mass = (myGalaxy.GalaxyMass - Mtotal_si) / myGalaxy.GalaxyMass;
    error_mass *= error_mass ;
    bool debug = false;
    std::vector<std::vector<double>> rho = density(rho_0, alpha_0, rho_1, alpha_1, myGalaxy.r, myGalaxy.z);

    std::vector<double> vsim = calculate_rotational_velocity(myGalaxy.redshift, myGalaxy.dv0,
                                                             myGalaxy.x_rotation_points,
                                                             myGalaxy.r,
                                                             myGalaxy.z,
                                                             myGalaxy.costheta,
                                                             myGalaxy.sintheta,
                                                             rho,
                                                             debug,
                                                             myGalaxy.GPU_ID,
                                                             myGalaxy.cuda);
    double error = 0.0;
    for (int i = 0; i < myGalaxy.n_rotation_points; i++) {
        double a = myGalaxy.v_rotation_points[i] - vsim[i];
        error += a*a;
    }
    std::cout << "Total Error = " << error  << "\n";
    return error + error_mass/1000.0;
}

// Define the objective function wrapper
auto objective_wrapper = [](const std::vector<double> &x, std::vector<double> &grad, void *data) {
    auto *myGalaxy = static_cast<Galaxy *>(data);
    return error_function(x, *myGalaxy);
};

// #############################################################################
// #############################################################################
// #############################################################################
// #############################################################################
// #############################################################################
// #############################################################################

Galaxy::Galaxy(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
               double R_max, int nr_init, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift, int GPU_ID, bool cuda, bool debug)
        : R_max(R_max), nr(nr_init), nz(nz), nr_sampling(nr_sampling), nz_sampling(nz_sampling),
          alpha_0(alpha_0), rho_0(rho_0), alpha_1(alpha_1), rho_1(rho_1), h0(h0), redshift(redshift), GPU_ID(GPU_ID), cuda(cuda),debug(debug),
          GalaxyMass(GalaxyMass), n_rotation_points(0) {

    r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr);
    // Update nr
    nr = r.size();
    z = linspace(-h0 / 2.0, h0 / 2.0, nz);
    rho = density(rho_0, alpha_0, rho_1, alpha_1, r, z);
    theta = linspace(0, 2 * M_PI, ntheta);
    costheta = costhetaFunc(theta);
    sintheta = sinthetaFunc(theta);
    z_sampling = linspace(-h0 / 2.0, h0 / 2.0, nz);
    r_sampling = linspace(1, R_max, nr_sampling);
    dz = h0 / nz;
    dtheta = 2* M_PI/ntheta;
    dv0.resize(1);
    dv0[0] = r[0] * r[0]/2  * dz * dtheta;
    for (int i = 1; i < nr; i++) {
        dv0.push_back((r[i] - r[i - 1]) * (r[i] + r[i - 1]) /2* dz * dtheta);
    }

    // Initialize empty vectors for f_z, x_rotation_points, and v_rotation_points
    x_rotation_points = std::vector<double>();
    v_rotation_points = std::vector<double>();
}


Galaxy::~Galaxy() {};


std::vector<std::vector<double>>  Galaxy::DrudePropagator(double epoch, double time_step_years, double eta, double temperature) {
    // Calculate the effective cross-section
    double time_step_seconds = time_step_years * 365*3600*24;
    double lyr_to_m = 9.46073047258E+15;
    double H_cross_section = 3.53E-20;  //m^2
    double effective_cross_section = eta * H_cross_section;
    double scaling_factor = epoch/Radius_4D;
    double redshift = 1.0/scaling_factor-1;
    std::vector<double> r_drude = create_subgrid(r, scaling_factor);
    std::vector<double> dv0_drude = create_subgrid(dv0, scaling_factor* scaling_factor);
    std::vector<std::vector<double>> rho_drude(nr, std::vector<double>(nz, 0.0));;
    for (size_t i = 0; i < nr; i++) {
        for (size_t j = 0; j < nz; j++) {
            rho_drude[i][j] = rho[i][j]/scaling_factor/scaling_factor;
        }
    }
    // Half of the vertical points
    int half_nz = nz / 2;

    // Calculate the mass in each volume element
    std::vector<std::vector<double>> current_masses(nr, std::vector<double>(nz, 0.0));
    for (size_t i = 0; i < nr; i++) {
        for (size_t j = 0; j < nz; j++) {
            current_masses[i][j] = rho_drude[i][j] * dv0_drude[i];
        }
    }

    // Get the vertical acceleration array
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z =
            get_all_g(redshift, dv0_drude, r_drude, z, r_drude, z, costheta, sintheta, rho_drude, false);

    if(cuda){
        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z = get_all_torch(redshift, dv0_drude, r_drude, z,
                                                                                        r_drude, z, costheta, sintheta, rho_drude, GPU_ID, debug);
    }
    else {
        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z = get_all_g(redshift, dv0_drude, r_drude, z,
                                                                                      r_drude, z, costheta, sintheta, rho_drude, debug);
    }
    auto z_acceleration = f_z.second;

    std::vector<std::vector<double>> tau = calculate_tau(effective_cross_section, rho_drude, temperature);

    // Loop through radial points
    for (size_t i = 0; i < nr; i++) {
        double volume_cross_section = dv0_drude[i]/dz;
        // Loop through positive vertical points and zero
        for (size_t j = 1; j <= half_nz; j++) {
            double local_acceleration = z_acceleration[i][j];


            // Calculate drift velocity and mass drift
            double drift_velocity = tau[i][j] * local_acceleration;
            double mass_drift = drift_velocity /lyr_to_m * time_step_seconds * volume_cross_section * rho_drude[i][j];

            // Update mass in the current volume
            if(current_masses[i][j]<mass_drift){
                mass_drift= current_masses[i][j];
                current_masses[i][j] = 0.0;
            } else{
                current_masses[i][j] -= mass_drift;
            }
            current_masses[i][j-1] += mass_drift;
        }

        // Handle mass gain at z=0 (from both sides of the disk)
        double local_acceleration_z0 = z_acceleration[i][0];
        double tau_z0 = tau[i][0];
        double drift_velocity_z0 = tau_z0 * local_acceleration_z0;
        double mass_drift_z0 =  drift_velocity_z0 * time_step_seconds * volume_cross_section;
        current_masses[i][0] += mass_drift_z0;
    }

    // Update mass for negative z values
    for (size_t i = 0; i < nr; i++) {
        for (size_t j = half_nz + 1; j < nz; j++) {
            current_masses[i][j] = current_masses[i][nz - j];
        }
    }

    // Recalculate density from updated mass
    recalculate_density(current_masses);
    return current_masses;
}



void Galaxy::recalculate_density(const std::vector<std::vector<double>>& currentMasses) {
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nz; j++) {
            rho[i][j] = currentMasses[i][j] / dv0[i];
        }
    }
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
        z_sampling = {0.0};
    } else {
        z_sampling = {this->z};
    }
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z;
    if(cuda){
        f_z = get_all_torch(redshift, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, GPU_ID, debug);
    }
    else {
        f_z = get_all_g(redshift, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, debug);
    }
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
    opt.set_maxeval(max_iter);
    std::vector<double> x = x0;
    double minf=0.0;
    nlopt::result result = opt.optimize(x, minf);
    if (result < 0) {
        std::cerr << "nlopt failed: " << strerror(result) << std::endl;
    }
    std::cout << result << std::endl;
    return x;
}

std::vector<double> Galaxy::simulate_rotation_curve() {
    // Calculate density at all radii
    std::vector<double> x0{rho_0, alpha_0, rho_1, alpha_1, h0};
    int max_iter = 5000;
    double xtol_rel = 1e-6;
    std::vector<double> xout = nelder_mead(x0, *this, max_iter, xtol_rel);
    rho_0 = (rho_0 + xout[0])/2.0;
    alpha_0 = (alpha_0 + xout[1])/2.0;
    rho_1 = (rho_1 + xout[2])/2.0;
    alpha_1 = (alpha_1 + xout[3])/2.0;
    h0 = (h0 + xout[4])/2.0;
    rho = density(rho_0, alpha_0, rho_1, alpha_1, r, z);
    // Calculate rotational velocity at all radii
    v_simulated_points = calculate_rotational_velocity(redshift, dv0, x_rotation_points,
                                                       r, z, costheta, sintheta,
                                                       rho, GPU_ID, false, cuda);
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


