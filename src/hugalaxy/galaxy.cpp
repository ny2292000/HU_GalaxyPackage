//
// Created by mp74207 on 4/17/23.
//
#include "tensor_utils.h"
#include <cmath>
#include <nlopt.hpp>
#include <vector>
#include <array>
#include <utility>
#include <iostream>
#include <torch/torch.h>

#include <exception>
#include "galaxy.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <sstream>
#include <iomanip>


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

    double MGas_si = myGalaxy.calculate_mass(rho_1, alpha_1, 100000);
    double error_gas = MGas_si/ myGalaxy.GalaxyMass;


    std::vector<std::vector<double>> rho = myGalaxy.density(rho_0, alpha_0, rho_1, alpha_1, myGalaxy.r, myGalaxy.z);

    std::vector<double> vsim = myGalaxy.calculate_rotational_velocity(rho);
    double error = 0.0;
    for (int i = 0; i < myGalaxy.n_rotation_points; i++) {
        double a = myGalaxy.v_rotation_points[i] - vsim[i];
        error += a*a;
    }
//    std::cout << "Total Error = " << error  << "\n";
    return error + error_mass*50 + error_gas*50;
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
               double R_max, int nr_init, int nz, int ntheta, double redshift, int GPU_ID,
               bool cuda, bool taskflow, double xtol_rel, int max_iter)
        : R_max(R_max), nr(nr_init), nz(nz), ntheta(ntheta),
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
    dv0 = std::vector<double>();
    recalculate_dv0();
    // Initialize empty vectors for f_z, x_rotation_points, and v_rotation_points
    x_rotation_points = std::vector<double>();
    v_rotation_points = std::vector<double>();
    v_simulated_points = std::vector<double>();
}


galaxy::~galaxy() {}


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
//    std::cout << result << std::endl;
//    std::cout << "Total Error = " << minf/(1+redshift)/(1+redshift) << "\n";
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
//            density_[i][j] = rho_0 * std::exp(-alpha_0 * r[i]) + rho_1 * std::exp(-alpha_1 * r[i] * r[i]);
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

double galaxy::calculate_mass_gaussian(double rho, double alpha, double h) {
    double factor = 0.0007126927557971729; // factor takes care of moving from rho as atom/cc to kg/lyr^3, with alpha = 1/lyr and h0 = in lyr div sun_mass
    double Mtotal_si = h * rho * M_PI /alpha ; //where h is in lyr and alpha is in 1/lyr
    return Mtotal_si*factor;
}

std::vector<double> galaxy::creategrid(double rho_0, double alpha_0, double rho_1, double alpha_1, unsigned int n) {
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
    int n1 = n/2;
//    int n1 = M1 / (M1 + M2) * n;
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

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
galaxy::get_f_z(const std::vector<std::vector<double>> &rho_, bool calc_vel,  const double height) const {
    // Calculate the total mass of the galaxy_
    std::vector<double> r_sampling = this->x_rotation_points;
    std::vector<double> z_sampling;
    if (calc_vel or height!=0.0) {
        z_sampling = {height};
    } else {
        z_sampling = {this->z};
    }
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z;
    std::string compute_choice = getCudaString(cuda, taskflow_);
    if(compute_choice=="GPU_Torch_Chunks"){
        f_z = get_all_torch_chunks(redshift, dv0, x_rotation_points, z_sampling,
                                   r, z, costheta, sintheta, rho_, GPU_ID);
    } else if(compute_choice=="GPU_Torch_No_Chunks"){
        f_z = get_all_torch_no_chunks(redshift, dv0, x_rotation_points, z_sampling,
                                      r, z, costheta, sintheta, rho_, GPU_ID);
    }
    else if (compute_choice=="CPU_TaskFlow") {
        tf::Taskflow tf;
        f_z = get_all_g_thread(tf, redshift, dv0, x_rotation_points, z_sampling,
                               r, z, costheta, sintheta, rho_);
    } else {
        f_z = get_all_g(redshift, dv0, x_rotation_points, z_sampling,
                        r, z, costheta, sintheta, rho_);
    }
    c10::cuda::CUDACachingAllocator::emptyCache();
    return f_z;
}



std::vector<double> galaxy::calculate_rotational_velocity(const std::vector<std::vector<double>> &rho, const double height) {
    int nr_sampling = x_rotation_points.size();
    double km_lyr = 9460730472580.8; //uu.lyr.to(uu.km)
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z = get_f_z(rho, true, height);
    // Allocate result vector
    std::vector<double> v_r(nr_sampling,0.0);
    // Calculate velocities
    v_simulated_points.clear();
    double v_squared;
    for (int i = 0; i < nr_sampling; i++) {
        v_squared = f_z.first[i][0] * x_rotation_points[i] * km_lyr; // Access radial values from the pair (first element)
        v_simulated_points.push_back(sqrt(v_squared)); // 9460730777119.56 km
    }
    // Return result
    return v_simulated_points;
}

std::vector<double> galaxy::calculate_rotational_velocity_internal() {
    int nr_sampling = x_rotation_points.size();
    double km_lyr = 9460730472580.8; //uu.lyr.to(uu.km)
    rho = density(rho_0, alpha_0, rho_1, alpha_1, r, z);
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z = get_f_z(rho, true,0.0);
    // Allocate result vector
    std::vector<double> v_r(nr_sampling,0.0);
    // Calculate velocities
    v_simulated_points.clear();
    double v_squared;
    for (int i = 0; i < nr_sampling; i++) {
        v_squared = f_z.first[i][0] * x_rotation_points[i] * km_lyr; // Access radial values from the pair (first element)
        v_simulated_points.push_back(sqrt(v_squared)); // 9460730777119.56 km
    }
    // Return result
    return v_simulated_points;
}

double galaxy::calculate_total_mass() {
    double total_mass = 0;
    for (size_t i = 0; i < nr; i++) {
        for (size_t j = 0; j < nz; j++) {
            total_mass += current_masses[i][j];
        }
    }
    return total_mass/1.9884099E40*ntheta;  // total mass in 1E10 Solar Masses
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
    auto vin = calculate_rotational_velocity_internal();
    n_rotation_points = x_rotation_points.size();
    v_simulated_points.clear();
    for (const auto &row: vin) {
        v_simulated_points.push_back(row); // Extract the first column (index 0)
    }
    return xout;
}


std::vector<std::vector<double>>  galaxy::DrudePropagator(double redshift, double deltaTime, double eta, double temperature) {
    // Calculate the effective cross-section
    //    double radius_of_cmb = 11E6; // 11 million light-years
    //    double density_at_cmb = 1E3; // hydrogen atoms per cubic centimeter
    //    move_galaxy_redshift(redshift);
    double initial_total_mass = calculate_total_mass();
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
    move_galaxy_redshift_drude(redshift);
    // Half of the vertical points
    int half_nz = nz / 2;
    int km_to_m = 1000;

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
    for (size_t i = 0; i < tau[0].size(); i++) {
        for (size_t j = 0; j < tau.size(); j++) {   // <-- change "i++" to "j++" here
            if (tau[i][j] > time_step_seconds) {
                tau[i][j] = time_step_seconds;
            }
        }
    }
    // Loop through radial points
    for (size_t i = 0; i < nr; i++) {
        double volume_cross_section = dv0[i]/dz;
        // Loop through positive vertical points and zero
        for (size_t j = 1; j <= half_nz; j++) {
            double local_acceleration = z_acceleration[i][j-1]* km_to_m;


            // Calculate drift velocity and mass drift
            double drift_velocity = std::abs(tau[i][j-1] * local_acceleration);
            double mass_drift = std::abs(drift_velocity /lyr_to_m * time_step_seconds * volume_cross_section * rho[i][j-1]);

            // Update mass in the current volume
            if(current_masses[i][j-1]< mass_drift){
                mass_drift= current_masses[i][j-1];
                current_masses[i][j-1] = 0.0;
                current_masses[i][j] += mass_drift;
            } else{
                current_masses[i][j] += mass_drift;
                current_masses[i][j-1] -= mass_drift;
            }
// Double check this
// TODO double check
            if (j==half_nz){
                current_masses[i][j] += mass_drift;
            }

        }
    }

    // Update mass for negative z values
    for (size_t i = 0; i < nr; i++) {
        for (size_t j = half_nz + 1 ; j < nz; j++) {
            current_masses[i][j] = current_masses[i][nz - j - 1 ];
        }
    }
    double final_total_mass = calculate_total_mass(); // Calculate final total mass
    if (std::abs(initial_total_mass/final_total_mass-1.0) > 1e-3) {
        std::cout << "Warning: mass not conserved! Initial: " << initial_total_mass << ", Final: " << final_total_mass << std::endl;
    }
    return current_masses;
}


void galaxy::DrudeGalaxyFormation(std::vector<double> epochs,
                                  double eta,
                                  double temperature,
                                  std::string filename_base)  // Changed filename to filename_base
{
    long unsigned n_epochs = epochs.size();
    // 3D vectors to store data for each epoch
    std::vector<std::vector<std::vector<double>>> all_current_masses;
    std::vector<std::vector<double>> all_dv0;
    std::vector<std::vector<double>> all_r;
    std::vector<std::vector<double>> all_z;
    std::vector<double> all_total_mass;
    std::vector<double> all_redshifts;
    double redshift_0 = Radius_4D/epochs[0] - 1.0;
    std::string redshift_str;
    std::stringstream ss;
    ss << round(redshift_0);
    redshift_str = ss.str();
    move_galaxy_redshift(redshift_0);
    density_internal();
    recalculate_masses();
    all_current_masses.push_back(current_masses);
    all_dv0.push_back(dv0);  // Save dv0 for each epoch
    all_r.push_back(r);  // Save r for each epoch
    all_z.push_back(z);  // Save z for each epoch
    all_total_mass.push_back(calculate_total_mass());
    all_redshifts.push_back(redshift_0);  // Save z for each epoch
    for(int i=1; i<n_epochs; i++) {
        double redshift = Radius_4D/epochs[i] - 1.0;
        double delta_time = epochs[i]-epochs[i-1];
        std::vector<std::vector<double>> current_masses = DrudePropagator(redshift, delta_time, eta, temperature);
        double final_total_mass_1 = calculate_total_mass();
        all_total_mass.push_back(final_total_mass_1);
        if (has_nan(current_masses)) {
            std::cout << "There are NaNs in the array.\n";
        } else {
            std::cout << "There are no NaNs in the array.\n";
        }
        all_redshifts.push_back(redshift);
        all_current_masses.push_back(current_masses);  // Save current_masses for each epoch

        // Assuming get_dv0(), get_r(), get_z() return the dv0, r, and z values respectively for the current epoch
        all_dv0.push_back(dv0);  // Save dv0 for each epoch

        all_r.push_back(r);  // Save r for each epoch

        all_z.push_back(z);  // Save z for each epoch

        if (has_nan(current_masses)) {
            std::cout << "There are NaNs in the current_masses array at epoch " << i << ".\n";
        } else {
            std::cout << "There are no NaNs in the current_masses array at epoch " << i << ".\n";
            std::cout << "Drude Simulation redshift " << redshift_str<< " - total mass of galaxy = " << final_total_mass_1 << "\n";
        }
    }

    // Save data as numpy arrays
    try {
// Save 1D vectors
        save_npy(filename_base + "_epochs_" + redshift_str + ".npy", epochs);
        save_npy(filename_base + "_all_redshifts_" + redshift_str + ".npy", all_redshifts);
        save_npy(filename_base + "_all_total_mass_" + redshift_str + ".npy", all_total_mass);

// Save 2D vectors
        save_npy(filename_base + "_all_dv0_" + redshift_str + ".npy", all_dv0);
        save_npy(filename_base + "_all_r_" + redshift_str + ".npy", all_r);
        save_npy(filename_base + "_all_z_" + redshift_str + ".npy", all_z);

// Save 3D vectors
        save_npy(filename_base + "_all_current_masses_" + redshift_str + ".npy", all_current_masses);

    } catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << '\n';
    }
    c10::cuda::CUDACachingAllocator::emptyCache();
}


std::vector<std::vector<double>> galaxy::FreeFallPropagator(double redshift, double deltatime) {
    // Calculate the effective cross-section
    //    double radius_of_cmb = 11E6; // 11 million light-years
    //    double density_at_cmb = 1E3; // hydrogen atoms per cubic centimeter
    //    move_galaxy_redshift(redshift);
    double initial_total_mass = calculate_total_mass();
    double time_step_seconds = deltatime * 365 * 3600 * 24;
    double lyr_to_m = 9.46073047258E+15;
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
    move_galaxy_redshift_drude(redshift);
    // Half of the vertical points
    int half_nz = nz / 2;
    int km_to_m = 1000;

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


    for (size_t i = 0; i < nr; i++) {
        for (size_t j = 0; j <= half_nz; j++) {
            double local_acceleration = z_acceleration[i][j] * km_to_m;
            double delta_z = -0.5 * local_acceleration * time_step_seconds * time_step_seconds;
            double fall_velocity = local_acceleration * time_step_seconds/3E8;
            if (std::abs(fall_velocity) > 0.1) {
                std::cout << "Relativistic Speed Reached: " << fall_velocity << std::endl;
            }
            int dz_cells = std::round(delta_z / dz/lyr_to_m);
            if (dz_cells > 0){
                // Transfer all the mass from current cell to the cell below by dz_cells
                if (j + dz_cells < half_nz) {
                    current_masses[i][j+dz_cells] += current_masses[i][j];
                } else {
                    current_masses[i][half_nz] += 2 * current_masses[i][j];  // double the mass as it's the accumulation point
                }
                current_masses[i][j] = 0.0;  // Clearing the mass from the current cell
            }
        }
    }

// Mirror the mass for negative z values
    for (size_t i = 0; i < nr; i++) {
        for (size_t j = half_nz + 1; j < nz; j++) {
            current_masses[i][j] = current_masses[i][nz - j - 1];
        }
    }

    double final_total_mass = calculate_total_mass(); // Calculate final total mass
    if (std::abs(initial_total_mass/final_total_mass-1.0) > 1e-3) {
        std::cout << "Warning: mass not conserved! Initial: " << initial_total_mass << ", Final: " << final_total_mass << std::endl;
    }
    return current_masses;
}


void galaxy::FreeFallGalaxyFormation(std::vector<double> epochs,
                                     std::string filename_base)  // Changed filename to filename_base
{
    long unsigned n_epochs = epochs.size();

    // 3D vectors to store data for each epoch
    std::vector<std::vector<std::vector<double>>> all_current_masses;
    std::vector<std::vector<double>> all_dv0;
    std::vector<std::vector<double>> all_r;
    std::vector<std::vector<double>> all_z;
    std::vector<double> all_total_mass;
    std::vector<double> all_redshifts;
    double redshift_0 = Radius_4D/epochs[0] - 1.0;
    std::string redshift_str;
    std::stringstream ss;
    ss << round(redshift_0);
    redshift_str = ss.str();
    move_galaxy_redshift(redshift_0);
    density_internal();
    recalculate_masses();
    all_current_masses.push_back(current_masses);
    all_dv0.push_back(dv0);  // Save dv0 for each epoch
    all_r.push_back(r);  // Save r for each epoch
    all_z.push_back(z);  // Save z for each epoch
    all_total_mass.push_back(calculate_total_mass());
    all_redshifts.push_back(redshift_0);  // Save dv0 for each epoch
    for(int i=1; i<n_epochs; i++) {
        double redshift = Radius_4D/epochs[i] - 1.0;
        double delta_time = epochs[i]-epochs[0];
        std::vector<std::vector<double>> current_masses = FreeFallPropagator(redshift, delta_time);
        double final_total_mass_1 = calculate_total_mass();
        all_total_mass.push_back(final_total_mass_1);
        if (has_nan(current_masses)) {
            std::cout << "There are NaNs in the array.\n";
        } else {
            std::cout << "There are no NaNs in the array.\n";
        }
        all_current_masses.push_back(current_masses);  // Save current_masses for each epoch
        all_redshifts.push_back(redshift);
        // Assuming get_dv0(), get_r(), get_z() return the dv0, r, and z values respectively for the current epoch
        all_dv0.push_back(dv0);  // Save dv0 for each epoch

        all_r.push_back(r);  // Save r for each epoch

        all_z.push_back(z);  // Save z for each epoch

        if (has_nan(current_masses)) {
            std::cout << "There are NaNs in the current_masses array at epoch " << i << ".\n";
        } else {
            std::cout << "There are no NaNs in the current_masses array at epoch " << i << ".\n";
            std::cout << "FreeFall Simulation redshift " << redshift_str<< " - total mass of galaxy = " << final_total_mass_1 << "\n";
        }
    }

    // Save data as numpy arrays
    try {
// Save 1D vectors
        save_npy(filename_base + "_freefall_epochs_" + redshift_str + ".npy", epochs);
        save_npy(filename_base + "_freefall_redshifts_" + redshift_str + ".npy", all_redshifts);
        save_npy(filename_base + "_freefall__all_total_mass_" + redshift_str + ".npy", all_total_mass);

// Save 2D vectors
        save_npy(filename_base + "_freefall_all_dv0_" + redshift_str + ".npy", all_dv0);
        save_npy(filename_base + "_freefall_all_r_" + redshift_str + ".npy", all_r);
        save_npy(filename_base + "_freefall_all_z_" + redshift_str + ".npy", all_z);

// Save 3D vectors
        save_npy(filename_base + "_freefall_all_current_masses_" + redshift_str + ".npy", all_current_masses);

    } catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << '\n';
    }
    c10::cuda::CUDACachingAllocator::emptyCache();
}

void galaxy::move_galaxy_redshift(double redshift_) {
    std::vector<double> x0 = calculate_density_parameters(redshift_);
    rho_0 = x0[0]; //z=0
    alpha_0 = x0[1];
    rho_1 = x0[2];
    alpha_1 = x0[3];
    h0 = x0[4];
    r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr);
    // Update nr
    if (nr!=r.size()){
        nr = r.size();
        std::cout << "number of radial points changed" <<std::endl;
    }
    ntheta = sintheta.size();
    z = linspace(-h0 / 2.0, h0 / 2.0, nz);
    redshift=redshift_;
    recalculate_dv0();
    double M0= calculate_mass(rho_0, alpha_0, h0)/1E10;
    double M1= calculate_mass(rho_1, alpha_1, h0)/1E10;
    density_internal();
    recalculate_masses();
    double M_total = calculate_total_mass();
//    std::cout << "M0 = " << M0 << std::endl  <<  "M1 = " << M1 << std::endl;
//    std::cout << "Total SummedUp Mass Calculated from summing up cells "  << M_total << std::endl;
}


void galaxy::move_galaxy_redshift_drude(double redshift_) {
    std::vector<double> x0 = calculate_density_parameters(redshift_);
    double rho_0 = x0[0]; //z=0
    double alpha_0 = x0[1];
    double rho_1 = x0[2];
    double alpha_1 = x0[3];
    double h0 = x0[4];
    r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr);
    // Update nr
    if (nr!=r.size()){
        nr = r.size();
        std::cout << "number of radial points changed" <<std::endl;
    }
    ntheta = sintheta.size();
    z = linspace(-h0 / 2.0, h0 / 2.0, nz);
    redshift=redshift_;
    recalculate_dv0();
    recalculate_density();
    double M_total = calculate_total_mass();
//    std::cout << "M0 = " << M0 << std::endl  <<  "M1 = " << M1 << std::endl;
//    std::cout << "Total SummedUp Mass Calculated from summing up cells "  << M_total << std::endl;
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
}

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

void galaxy::density_internal() {
    rho = density(rho_0, alpha_0, rho_1, alpha_1, r, z);
}

std::vector<std::vector<double>> galaxy::calibrate_df(std::vector<std::array<double, 2>> m33_rotation_curve, double m33_redshift, int range_) {
    // A map is chosen here to store data temporarily
    std::map<double, std::vector<double>> df_map;

    std::vector<double> redshift_births;
    for (int i = 0; i <= range_; i++) redshift_births.push_back(i); // np.arange(0,20,1)
    for (double redshift_birth : redshift_births) {
        // Replace this with your actual function to get new rotation curve
        std::vector<std::array<double, 2>> new_m33_rotation_curve = move_rotation_curve(m33_rotation_curve, m33_redshift,redshift_birth);
        read_galaxy_rotation_curve(new_m33_rotation_curve);
        move_galaxy_redshift(redshift_birth);
        std::vector<double> values = simulate_rotation_curve();
        // Appending the two double values to the end of the values vector
        values.push_back(calculate_mass(rho_0, alpha_0, h0));
        values.push_back(calculate_mass(rho_1, alpha_1, h0));
        values.push_back(redshift_birth);
        df_map[redshift_birth] = values;
    }

    // Convert map to 2D vector for output
    std::vector<std::vector<double>> df;
    for (const auto& pair : df_map) {
        df.push_back(pair.second);
    }
    return df;
}