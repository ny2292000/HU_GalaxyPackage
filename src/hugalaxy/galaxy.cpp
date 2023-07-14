//
// Created by mp74207 on 4/17/23.
//
#define _USE_MATH_DEFINES
#include "tensor_utils.h"
#include <cmath>
#include <nlopt.hpp>
#include <vector>
#include <array>
#include <utility>
#include <iostream>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include "galaxy.h"

#include <c10/cuda/CUDACachingAllocator.h>




// Define the function to be minimized
static double error_function(const std::vector<double> &x, galaxy &myGalaxy) {
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
    // Calculate the total mass of the galaxy_
    double Mtotal_si = myGalaxy.calculate_mass(rho_0, alpha_0, h0);
    double error_mass = (myGalaxy.GalaxyMass - Mtotal_si) / myGalaxy.GalaxyMass;
    error_mass *= error_mass ;
    std::vector<std::vector<double>> rho = myGalaxy.density(rho_0, alpha_0, rho_1, alpha_1, myGalaxy.r, myGalaxy.z);

    std::vector<double> vsim = myGalaxy.calculate_rotational_velocity(rho);
    double error = 0.0;
    for (int i = 0; i < myGalaxy.n_rotation_points; i++) {
        double a = myGalaxy.v_rotation_points[i] - vsim[i];
        error += a*a;
    }
    std::cout << "Total Error = " << error  << "\n";
    return error + error_mass*10;
}

// Define the objective function wrapper
auto objective_wrapper = [](const std::vector<double> &x, std::vector<double> &grad, void *data) {
    auto *myGalaxy = static_cast<galaxy *>(data);
    return error_function(x, *myGalaxy);
};

// #############################################################################
// #############################################################################
// #############################################################################
// #############################################################################
// #############################################################################
// #############################################################################

galaxy::galaxy(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
               double R_max, int nr, int nz, int ntheta, double redshift, int GPU_ID,
               bool cuda, bool taskflow, double xtol_rel, int max_iter)
        : R_max(R_max), nr(nr), nz(nz), ntheta(ntheta),
          alpha_0(alpha_0), rho_0(rho_0), alpha_1(alpha_1), rho_1(rho_1), h0(h0), redshift(redshift), GPU_ID(GPU_ID),
          cuda(cuda), taskflow_(taskflow), xtol_rel(xtol_rel), max_iter(max_iter),
          GalaxyMass(GalaxyMass), n_rotation_points(0) {

    r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr);
    // Update nr
    nr = r.size();
    z = linspace(-h0 / 2.0, h0 / 2.0, nz);
    rho = density(rho_0, alpha_0, rho_1, alpha_1, r, z);
    theta = linspace(0, 2 * M_PI, ntheta);
    costheta = costhetaFunc(theta);
    sintheta = sinthetaFunc(theta);
    current_masses = zeros_2(nr, nz);
    rho = zeros_2(nr,nz);
    dz = h0 / nz;
    dtheta = 2* M_PI/ntheta;
    recalculate_dv0();
    // Initialize empty vectors for f_z, x_rotation_points, and v_rotation_points
    x_rotation_points = std::vector<double>();
    v_rotation_points = std::vector<double>();
    v_simulated_points = std::vector<double>();
}


galaxy::~galaxy() {};


// Define the Nelder-Mead optimizer
std::vector<double>
galaxy::nelder_mead(const std::vector<double> &x0, int max_iter, double xtol_rel) {
    nlopt::opt opt(nlopt::LN_NELDERMEAD, x0.size());
    opt.set_min_objective(objective_wrapper, this);
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

std::vector<std::vector<double>> galaxy::density(double rho_0, double alpha_0, double rho_1, double alpha_1,
                                         const std::vector<double>& r, const std::vector<double>& z) const {
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

double galaxy::calculate_mass(double rho, double alpha, double h) {
    double factor = 0.0007126927557971729; // factor takes care of moving from rho as atom/cc to kg/lyr^3, with alpha = 1/lyr and h0 = in lyr div sun_mass
    double Mtotal_si = 2 * M_PI * h * rho /(alpha*alpha); //where h is in lyr and alpha is in 1/lyr
    return Mtotal_si*factor;
}

std::vector<double> galaxy::creategrid(double rho_0, double alpha_0, double rho_1, double alpha_1, int n) {
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
    double M1 = calculate_mass(rho_0, alpha_0, 1.0);
    double M2 = calculate_mass(rho_1, alpha_1, 1.0);
    int n1 = M1 / (M1 + M2) * n;
    int n2 = n - n1;
    double r_min1 = 1.0;
    double r_min2 = r_max_1 + 1.0;

    // Define the grid of n points using a geometric sequence
    std::vector<double> r(n1 + n2);
    for (int i = 0; i < n1; i++) {
        r[i] = r_min1 * std::pow(r_max_1 / r_min1, i / (double) (n1 - 1));
    }
    for (int i = n1; i < n; i++) {
        r[i] = r_min2 * std::pow(r_max_2 / r_min2,(i-n1) / (double) (n - n1));
    }
    return r;
}

std::vector<double> galaxy::calculate_rotational_velocity(const std::vector<std::vector<double>> &rho, const double height) const {
    int nr_sampling = x_rotation_points.size();
    double km_lyr = 9460730472580.8; //uu.lyr.to(uu.km)
    // Allocate result vector
    std::vector<double> z_sampling = {height};
    std::vector<double> v_r(nr_sampling,0.0);
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z;
    std::string compute_choice = getCudaString(cuda, taskflow_);
    if(compute_choice=="GPU_Torch_Chunks"){
        f_z = get_all_torch_chunks(redshift, dv0, x_rotation_points, z_sampling,
                                   r, z, costheta, sintheta, rho, GPU_ID);
    } else if(compute_choice=="GPU_Torch_No_Chunks"){
        f_z = get_all_torch_no_chunks(redshift, dv0, x_rotation_points, z_sampling,
                                      r, z, costheta, sintheta, rho, GPU_ID);
    }
    else if (compute_choice=="CPU_TaskFlow") {
        tf::Taskflow tf;
        f_z = get_all_g_thread(tf, redshift, dv0, x_rotation_points, z_sampling,
                               r, z, costheta, sintheta, rho);
    } else {
        f_z = get_all_g(redshift, dv0, x_rotation_points, z_sampling,
                        r, z, costheta, sintheta, rho);
    }
    // Calculate velocities
    double v_squared;
    for (int i = 0; i < nr_sampling; i++) {
        v_squared = f_z.first[i][0] * x_rotation_points[i] * km_lyr; // Access radial values from the pair (first element)
        v_r[i] = sqrt(v_squared); // 9460730777119.56 km
    }
    // Return result
    return v_r;
}

std::vector<std::vector<double>>  galaxy::DrudePropagator(double redshift, double deltaTime, double eta, double temperature) {
    // Calculate the effective cross-section
//    double radius_of_cmb = 11E6; // 11 million light-years
//    double density_at_cmb = 1E3; // hydrogen atoms per cubic centimeter
    move_galaxy_redshift(redshift);
    double time_step_seconds = deltaTime * 365 * 3600 * 24;
    double lyr_to_m = 9.46073047258E+15;
    double H_cross_section = 3.53E-20;  //m^2
    double effective_cross_section = eta * H_cross_section;
//    double radius_of_epoch = Radius_4D/(1+redshift);
//    double rho_at_epoch = density_at_cmb*pow(radius_of_cmb/radius_of_epoch,3);
//    //////////////
    // redshift 13 means 4D radius of 1 billion light years or 1 billion years after the Universe Creation.
    // The current distance ladder tells you 332 million years.
    // https://www.astro.ucla.edu/~wright/CosmoCalc.html
    // Table 3.  The Universe goes from a superfluid neutron matter Neutronium phase to a plasma phase during the
    // process of Neutronium decay. Plasma Gamma is the average gamma coefficient until the recombination event.
    // The recombination event happened when the Universe was 11.1 million years old, the plasma temperature
    // was 3443 Kelvin and the gas density was 8.30E-17 kg/m3. The redshift for the CMB is z=1263.
    // So, 1 atom of Hydrogen weighs 1.008/(6.023x10^23) g = 0.167x10^(-23) g = 1.67x10^(-27) kg
    // 49700 atoms/cubic centimeter
    // Density at redshift 139 (at 100 million years old universe) =((1 + redshift)/(1+1263))**3*49700=67.5 atoms/cc
    //////////////
    recalculate_masses();
    // Half of the vertical points
    int half_nz = nz / 2;


    // Get the double acceleration array
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z;
    std::string compute_choice = getCudaString(cuda, taskflow_);
    if(compute_choice=="GPU_Torch_Chunks"){
        f_z = get_all_torch_chunks(redshift, dv0, r, z,
                                   r, z, costheta, sintheta, rho, GPU_ID);
    } else if(compute_choice=="GPU_Torch_No_Chunks"){
        f_z = get_all_torch_no_chunks(redshift, dv0, r, z,
                                   r, z, costheta, sintheta, rho, GPU_ID);
    }
    else if (compute_choice=="CPU_TaskFlow") {
        tf::Taskflow tf;
        f_z = get_all_g_thread(tf, redshift, dv0, r, z,
                               r, z, costheta, sintheta, rho);
    } else {
        f_z = get_all_g(redshift, dv0, r, z,
                        r, z, costheta, sintheta, rho);
    }
    auto z_acceleration = f_z.second;

    std::vector<std::vector<double>> tau = calculate_tau(effective_cross_section, rho, temperature);

    // Loop through radial points
    for (size_t i = 0; i < nr; i++) {
        double volume_cross_section = dv0[i]/dz;
        // Loop through positive vertical points and zero
        for (size_t j = 1; j <= half_nz; j++) {
            double local_acceleration = z_acceleration[i][j];


            // Calculate drift velocity and mass drift
            double drift_velocity = tau[i][j] * local_acceleration;
            double mass_drift = drift_velocity /lyr_to_m * time_step_seconds * volume_cross_section * rho[i][j];

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
    recalculate_density();
    return current_masses;
}


std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
galaxy::get_f_z(const std::vector<double> &x) {
    // Calculate the rotation velocity using the current values of x
    double rho_0 = x[0];
    double alpha_0 = x[1];
    double rho_1 = x[2];
    double alpha_1 = x[3];
    double h0 = x[4];
    // Calculate the total mass of the galaxy_
    std::vector<double> r_sampling = this->x_rotation_points;
    std::vector<double> z_sampling = this->z;

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z;
    std::string compute_choice = getCudaString(cuda, taskflow_);
    if(compute_choice=="GPU_Torch_Chunks"){
        f_z = get_all_torch_chunks(redshift, dv0, x_rotation_points, z_sampling,
                                   r, z, costheta, sintheta, rho, GPU_ID);
    } else if(compute_choice=="GPU_Torch_No_Chunks"){
        f_z = get_all_torch_no_chunks(redshift, dv0, x_rotation_points, z_sampling,
                                      r, z, costheta, sintheta, rho, GPU_ID);
    }
    else if (compute_choice=="CPU_TaskFlow") {
        tf::Taskflow tf;
        f_z = get_all_g_thread(tf, redshift, dv0, x_rotation_points, z_sampling,
                               r, z, costheta, sintheta, rho);
    } else {
        f_z = get_all_g(redshift, dv0, x_rotation_points, z_sampling,
                        r, z, costheta, sintheta, rho);
    }
    return f_z;
}

void galaxy::read_galaxy_rotation_curve(std::vector<std::array<double, 2>> vin) {
    n_rotation_points = vin.size();
    this->x_rotation_points.clear();
    this->v_rotation_points.clear();
    for (const auto &row: vin) {
        this->x_rotation_points.push_back(row[0]); // Extract the first column (index 0)
        this->v_rotation_points.push_back(row[1]); // Extract the first column (index 0)
    }
}




std::vector<double> galaxy::simulate_rotation_curve() {
    // Calculate density at all radii
    std::vector<double> x0{rho_0, alpha_0, rho_1, alpha_1, h0};
    std::vector<double> xout = nelder_mead(x0, max_iter, xtol_rel);
    rho_0 = xout[0];
    alpha_0 = xout[1];
    rho_1 = xout[2];
    alpha_1 = xout[3];
    h0 = xout[4];
    rho = density(rho_0, alpha_0, rho_1, alpha_1, r, z);
    // Calculate rotational velocity at all radii
    auto vin = calculate_rotational_velocity(rho);
    n_rotation_points = x_rotation_points.size();
    v_simulated_points.clear();
    for (const auto &row: vin) {
        v_simulated_points.push_back(row); // Extract the first column (index 0)
    }
    return v_simulated_points;
}

std::vector<double> galaxy::move_galaxy_redshift(double redshift) {
    std::vector<double> x0 = calculate_density_parameters(redshift);
    rho_0 = x0[0]; //z=0
    alpha_0 = x0[1];
    rho_1 = x0[2];
    alpha_1 = x0[3];
    h0 = x0[4];
    r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr);
    // Update nr
    nr = r.size();
    ntheta = sintheta.size();
    z = linspace(-h0 / 2.0, h0 / 2.0, nz);
    recalculate_dv0();
    rho = density(rho_0, alpha_0, rho_1, alpha_1, r, z);
    return x0;
}





std::vector<std::vector<double>> galaxy::print_rotation_curve() {
    std::vector<std::vector<double>> rotation_curve;
    for (int i = 0; i < n_rotation_points; i++) {
        std::vector<double> point{x_rotation_points[i], v_rotation_points[i]};
        rotation_curve.push_back(point);
    }
    return rotation_curve;
}

std::vector<double> galaxy::print_density_parameters() {
    std::vector<double> density_params{rho_0, alpha_0, rho_1, alpha_1, h0};
    return density_params;
}


void galaxy::recalculate_density() {
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nz; j++) {
            rho[i][j] = current_masses[i][j] / dv0[i];
        }
    }
};

void galaxy::recalculate_masses() {
    for (int i=0; i<nr; i++){
        for (int j=0; j<nz; j++){
            current_masses[i][j] = rho[i][j] * dv0[i];
        }
    }
}

void galaxy::recalculate_dv0() {
    dz = h0 / nz;
    dtheta = 2* M_PI/ntheta;
    dv0.resize(1);
    dv0[0] = r[0] * r[0]/2  * dz * dtheta;
    for (int i = 1; i < nr; i++) {
        dv0.push_back((r[i] - r[i - 1]) * (r[i] + r[i - 1]) /2* dz * dtheta);
    }
}

