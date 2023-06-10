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
#include <future>
#include "torch/torch.h"
#include <cuda_runtime.h>
#include "Galaxy.h"

const double Radius_4D = 14.01;

std::vector<double> create_subgrid(const std::vector<double>& original_grid, double scaling_factor) {
    std::vector<double> subgrid;
    for (auto r : original_grid) {
            subgrid.push_back(r*scaling_factor);
    }
    return subgrid;
}

std::vector<std::vector<double>> calculate_tau(double effective_cross_section, const std::vector<std::vector<double>>& local_density, double temperature) {
    // Constants
    const double boltzmann_constant = 1.380649e-23; // J/K
    const double hydrogen_mass = 1.6737236e-27;
    const double lyr3_to_m3 = 8.4678666e+47; // kg

    // Calculate the average velocity of gas particles based on temperature
    double average_velocity = std::sqrt((3 * boltzmann_constant * temperature) / hydrogen_mass);

    // Calculate the number density of gas particles
    std::vector<std::vector<double>> number_density(local_density.size(), std::vector<double>(local_density[0].size(), 0.0));
    for (size_t i = 0; i < local_density.size(); i++) {
        for (size_t j = 0; j < local_density[0].size(); j++) {
            number_density[i][j] = local_density[i][j] / lyr3_to_m3 / hydrogen_mass;
        }
    }

    // Calculate the time between collisions
    std::vector<std::vector<double>> tau(local_density.size(), std::vector<double>(local_density[0].size(), 0.0));
    for (size_t i = 0; i < local_density.size(); i++) {
        for (size_t j = 0; j < local_density[0].size(); j++) {
            tau[i][j] = 1.0 / (number_density[i][j] * effective_cross_section * average_velocity);
        }
    }

    return tau;
}


double calculate_mass(double rho, double alpha, double h) {
    double factor = 0.0007126927557971729; // factor takes care of moving from rho as atom/cc to kg/lyr^3, with alpha = 1/lyr and h0 = in lyr div sun_mass
    double Mtotal_si = 2 * M_PI * h * rho /(alpha*alpha); //where h is in lyr and alpha is in 1/lyr
    return Mtotal_si*factor;
}


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

void printTensorShape(const torch::Tensor& tensor) {
    torch::IntArrayRef shape = tensor.sizes();

    std::cout << "Shape of the tensor: ";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;
}



void print_tensor_point(const torch::Tensor& tensor, int i, int j, int k) {
    auto tensor_acc = tensor.accessor<double,3>();
    std::cout << "Tensor point at (" << i << ", " << j << ", " << k << "): " << tensor_acc[i][j][k] << std::endl;
}



std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> get_g_torch(
        const torch::Tensor& r_sampling,
        const torch::Tensor& z_sampling,
        const torch::Tensor& G,
        const torch::Tensor& dv0,
        const torch::Tensor& r,
        const torch::Tensor& z,
        const torch::Tensor& costheta,
        const torch::Tensor& sintheta,
        const torch::Tensor& rho,
        bool debug) {

    auto options = r.options();

    // Get the sizes for each dimension
    int r_size = r_sampling.size(0);
    int z_size = z_sampling.size(0);
    int n_r = r.size(0);
    int n_theta = sintheta.size(0);
    int n_z = z.size(0);

    // Reshape tensors for broadcasting
    auto r_broadcasted = r.unsqueeze(1).unsqueeze(2);
    auto dv0_broadcasted = dv0.unsqueeze(1).unsqueeze(2);
    auto rho_broadcasted = rho.unsqueeze(1);
    auto G_broadcasted = G.unsqueeze(1).unsqueeze(2);
    auto sintheta_broadcasted = sintheta.unsqueeze(0).unsqueeze(2);
    auto costheta_broadcasted = costheta.unsqueeze(0).unsqueeze(2);
    auto z_broadcasted = z.unsqueeze(0).unsqueeze(1);

    // Initialize the output vectors with the correct dimensions
    std::vector<std::vector<double>> radial_value_2d(r_size, std::vector<double>(z_size));
    std::vector<std::vector<double>> vertical_value_2d(r_size, std::vector<double>(z_size));

    // Loop over r_sampling values
    for (int i = 0; i < r_size; ++i) {
        auto r_sampling_ii = r_sampling[i].unsqueeze(0).unsqueeze(1).unsqueeze(2);
        // Create masks for radial value calculation
        auto mask = (r_broadcasted >= r_sampling_ii);

        // Loop over z_sampling values
        for (int j = 0; j < z_size; ++j) {
            auto z_sampling_jj = z_sampling[j].unsqueeze(0).unsqueeze(1).unsqueeze(2);

            // Calculate the common factor without the mask
            // Calculate the distances
            auto d_3 = ( (z_broadcasted - z_sampling_jj).pow(2) +
                       (r_sampling_ii - r_broadcasted * sintheta_broadcasted).pow(2) +
                       (r_broadcasted * costheta_broadcasted).pow(2) ).pow(3/2);

            // Calculate the common factor
            auto commonfactor = G_broadcasted * rho_broadcasted * dv0_broadcasted;

            // Perform the summation over the last three dimensions
            vertical_value_2d[i][j] = (commonfactor/d_3 * (z_broadcasted - z_sampling_jj)).view({-1}).sum().item<double>();

            // Apply the mask to commonfactor before the division
            auto commonfactor_masked = (commonfactor/d_3).masked_fill_(mask, 0);

            auto commonfactor_masked_cpu = commonfactor_masked.to(torch::kCPU);

            auto tensor_acc = commonfactor_masked_cpu.accessor<double,3>();
            std::cout << "Tensor point at " << tensor_acc[5][5][5] << std::endl;

            // Then divide by d_3
            radial_value_2d[i][j] = (commonfactor_masked).view({-1}).sum().item<double>();
        }
    }

    return std::make_pair(radial_value_2d, vertical_value_2d);
}


torch::Tensor move_data_to_gpu(const std::vector<double>& host_data, const torch::Device& device) {
    try {
        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
        int sizes = host_data.size();
        // Allocate GPU memory for the tensor
        auto gpu_tensor = torch::empty(sizes, options);

        // Copy host data to GPU tensor
        gpu_tensor.copy_(torch::from_blob(const_cast<double*>(host_data.data()), sizes));

        return gpu_tensor;
    } catch (const std::exception& e) {
        // Exception handling
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return torch::Tensor();
    }
}

torch::Tensor move_data_to_gpu2D(const std::vector<std::vector<double>>& host_data, const torch::Device& device) {
    try {
        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);

        if (host_data.empty()) {
            return torch::empty({0}, options);  // Return an empty tensor
        }

        // Get the size of the host data
        int64_t rows = host_data.size();
        int64_t cols = (host_data[0].size() > 0) ? host_data[0].size() : 1;

        // Allocate GPU memory for the tensor
        auto gpu_tensor = torch::empty({rows, cols}, options);

        // Copy host data to GPU tensor
        for (int64_t i = 0; i < rows; i++) {
            gpu_tensor[i].copy_(torch::from_blob(const_cast<double*>(host_data[i].data()), {cols}));
        }

        return gpu_tensor;
    } catch (const std::exception& e) {
        // Exception handling
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return torch::Tensor();
    }
}

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
              bool debug) {

    // Define options for GPU tensors
    // Create tensors and move to GPU

    if (torch::cuda::is_available()) {
//        std::cout << "CUDA is available! Training on GPU." << std::endl;
        int num_devices = torch::cuda::device_count();
//        std::cout << "Number of GPUs: " << num_devices << std::endl;

        for (int i = 0; i < num_devices; ++i) {
            torch::Device device(torch::kCUDA, i);
//            std::cout << "GPU " << i << ": " << device << std::endl;

            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, i);
//            std::cout << "GPU name: " << props.name << std::endl;
//            std::cout << "GPU capability: " << props.major << "." << props.minor << std::endl;
            // Access more information about the GPU using other fields in props
        }
    } else {
        std::cout << "No GPUs available." << std::endl;
    }

    int GPU_N = 0;
    torch::Device device(torch::kCUDA, GPU_N);

// Move data to GPU
    torch::Tensor dv0 = move_data_to_gpu(dv0_in, device);
    torch::Tensor r_sampling = move_data_to_gpu(r_sampling_in, device);
    torch::Tensor z_sampling = move_data_to_gpu(z_sampling_in, device);
    torch::Tensor r = move_data_to_gpu(r_in, device);
    torch::Tensor z = move_data_to_gpu(z_in, device);
    torch::Tensor costheta = move_data_to_gpu(costheta_in, device);
    torch::Tensor sintheta = move_data_to_gpu(sintheta_in, device);
    torch::Tensor rho = move_data_to_gpu2D(rho_in, device);

    // Create G tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    auto G = torch::full({1}, 7.456866768350099e-46 * (1 + redshift), options);

    // Initialize the 2D arrays
    std::vector<std::vector<double>> radial_values_2d(r_sampling_in.size(), std::vector<double>(z_sampling_in.size()));
    std::vector<std::vector<double>> vertical_values_2d(r_sampling_in.size(), std::vector<double>(z_sampling_in.size()));

    auto [radial_value_2d, vertical_value_2d] = get_g_torch(r_sampling, z_sampling, G, dv0, r, z, costheta, sintheta, rho, debug);

    return {radial_values_2d, vertical_values_2d};
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
                double commonfactor  = G * rho[i][j] *  dv0[i]  / d_3;
                if ( r[i] < r_sampling_ii) {
                    thisradial_value = commonfactor * (r_sampling_ii - r[i] * sintheta[k]);
                    radial_value += thisradial_value;
                }
                thisvertical_value = commonfactor * (z[j] - z_sampling_jj);
                vertical_value += thisvertical_value;
                if (false) {
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
    int nr_local = r_sampling.size();
    int nz_local = z_sampling.size();
    futures.reserve(nr_local * nz_local);

    // Spawn threads
    std::vector<std::vector<double>> f_z_radial = zeros_2(nr_local, nz_local);
    std::vector<std::vector<double>> f_z_vertical = zeros_2(nr_local, nz_local);
    for (unsigned int i = 0; i < nr_local; i++) {
        for (unsigned int j = 0; j < nz_local; j++) {
            futures.emplace_back(
                    std::async(get_g_cpu, r_sampling[i], z_sampling[j], G, dv0, r, z, costheta, sintheta, rho, debug));
        }
    }

    // Collect results and populate f_z_radial and f_z_non_radial
    for (unsigned int i = 0; i < nr_local; i++) {
        for (unsigned int j = 0; j < nz_local; j++) {
            auto result_pair = futures[i * nz_local + j].get();
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
                                                  const std::vector<std::vector<double>> &rho, bool debug, bool cuda) {
    int nr_sampling = r_sampling.size();
    double km_lyr = 9460730472580.8; //uu.lyr.to(uu.km)
    // Allocate result vector
    std::vector<double> z_sampling = {0.0};
    std::vector<double> v_r(nr_sampling,0.0);
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z;
    if(cuda){
        f_z = get_all_torch(redshift, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, debug);
    }
    else {
        f_z = get_all_g(redshift, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, debug);
    }

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
    double M1 = calculate_mass(rho_0, alpha_0, 1.0);
    double M2 = calculate_mass(rho_1, alpha_1, 1.0);
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
               double R_max, int nr_init, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift, bool cuda)
        : R_max(R_max), nr(nr_init), nz(nz), nr_sampling(nr_sampling), nz_sampling(nz_sampling),
          alpha_0(alpha_0), rho_0(rho_0), alpha_1(alpha_1), rho_1(rho_1), h0(h0), redshift(redshift), cuda(cuda),
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
                                                                                        r_drude, z, costheta, sintheta, rho_drude, false);
    }
    else {
        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z = get_all_g(redshift, dv0_drude, r_drude, z,
                                                                                      r_drude, z, costheta, sintheta, rho_drude, false);
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
        f_z = get_all_torch(redshift, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, debug);
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
    double minf;
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
                                                       rho, false, cuda);
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


