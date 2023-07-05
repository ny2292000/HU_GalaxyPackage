#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <future>
#include <torch/torch.h>
#include "tensor_utils.h"

inline std::vector<std::vector<double>> tensor_to_vec_of_vec(const torch::Tensor& tensor){
    std::vector<std::vector<double>> vec_of_vec;

    // Assumption: tensor is 2D
    int rows = tensor.size(0);
    int cols = tensor.size(1);

    vec_of_vec.resize(rows, std::vector<double>(cols, 0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            try {
                vec_of_vec[i][j] = tensor[i][j].item<double>();
            }catch (const char* e) {
                std::cout << e << std::endl;
            }
        }
    }
    return vec_of_vec;
}

torch::Tensor move_data_to_gpu(const std::vector<double>& host_data, const torch::Device& device) {
    try {
        torch::Tensor cpu_tensor = torch::from_blob(const_cast<double*>(host_data.data()), {static_cast<int64_t>(host_data.size())}, torch::kFloat64);
        return cpu_tensor.to(device, torch::kFloat64);
    } catch (const std::exception& e) {
        // Exception handling
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return torch::Tensor();
    }
}


torch::Tensor move_data_to_gpu2D(const std::vector<std::vector<double>>& host_data, const torch::Device& device) {
    try {
        if (host_data.empty()) {
            return torch::empty({0}, torch::kFloat64);  // Return an empty tensor
        }

        // Get the size of the host data
        int64_t rows = host_data.size();
        int64_t cols = (host_data[0].size() > 0) ? host_data[0].size() : 0;

        // Create a flat vector from host_data
        std::vector<double> flat_data(rows * cols);
        for(int64_t i = 0; i < rows; ++i) {
            std::copy(host_data[i].begin(), host_data[i].end(), flat_data.begin() + i * cols);
        }

        // Create CPU tensor and copy the data
        torch::Tensor cpu_tensor = torch::from_blob(flat_data.data(), {rows, cols}, torch::kFloat64);

        // Move to GPU
        return cpu_tensor.to(device, torch::kFloat64);
    } catch (const std::exception& e) {
        // Exception handling
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return torch::Tensor();
    }
}


// Returns a vector of zeros with the given size
std::vector<double> zeros_1(int size) {
    return std::vector<double>(size, 0.0);
}

std::vector<std::vector<double>> zeros_2(int nr, int nz) {
    std::vector<std::vector<double>> vec(nr, std::vector<double>(nz, 0.0));
    return vec;
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

void print_1D(const std::vector<double> a) {
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

void print_tensor(const torch::Tensor& tensor) {
    auto tensor_flat = tensor.view({-1});
    tensor_flat = tensor_flat.to(torch::kCPU);
    for (int64_t i = 0; i < tensor_flat.size(0); ++i) {
        std::cout << "Tensor point at index " << i << ": " << tensor_flat[i].item<double>() << std::endl;
    }
}

void print_mask (const torch::Tensor& tensor) {
    auto tensor_flat = tensor.view({-1});
    tensor_flat = tensor_flat.to(torch::kCPU);
    for (int64_t i = 0; i < tensor_flat.size(0); ++i) {
        std::cout << "Tensor point at index " << i << ": " << tensor_flat[i].item<bool>() << std::endl;
    }
}


void print_tensor_point(const torch::Tensor& tensor, int i, int j, int k) {
    auto tensor_acc = tensor.accessor<double,3>();
    std::cout << "Tensor point at (" << i << ", " << j << ", " << k << "): " << tensor_acc[i][j][k] << std::endl;
}

void printTensorShape(const torch::Tensor& tensor) {
    torch::IntArrayRef shape = tensor.sizes();

    std::cout << "Shape of the tensor: ";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;
}

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

    // Get the sizes for each dimension
    int r_size = r_sampling.size(0);
    int z_size = z_sampling.size(0);
    int n_r = r.size(0);
    int n_theta = sintheta.size(0);
    int n_z = z.size(0);

    // Reshape tensors for broadcasting
    // tensor alignment r_sampling, z_sampling, r, theta, z = (0,1,2,3,4)
    // Reshape r_sampling for broadcasting
    auto r_sampling_broadcasted = r_sampling.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4);
    auto z_sampling_broadcasted = z_sampling.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4);

    auto r_broadcasted =     r.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4);
    auto dv0_broadcasted = dv0.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4);
    auto rho_broadcasted = rho.unsqueeze(0).unsqueeze(1).unsqueeze(3);
    auto G_broadcasted = G.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4);
    auto sintheta_broadcasted = sintheta.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4);
    auto costheta_broadcasted = costheta.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4);
    auto z_broadcasted = z.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3);

    // Create masks for radial value calculation
    auto mask = (r_broadcasted <= r_sampling_broadcasted).to(r_sampling_broadcasted.dtype());

    // Initialize the output tensors with the correct dimensions
    torch::Tensor radial_values_2d = torch::zeros({r_size, z_size});
    torch::Tensor vertical_values_2d = torch::zeros({r_size, z_size});


    // Calculate the common factor without the mask
    // Calculate the distances
    // Calculate the distances
    auto d_3 = ( (z_sampling_broadcasted - z_broadcasted).pow(2) +
                 (r_sampling_broadcasted - r_broadcasted * sintheta_broadcasted).pow(2) +
                 (r_broadcasted * costheta_broadcasted).pow(2) ).pow(1.5);


    // Calculate the common factor
    auto commonfactor = G_broadcasted * rho_broadcasted * dv0_broadcasted/d_3;

    // Perform the summation over the last three dimensions
    vertical_values_2d = (commonfactor * (z_sampling_broadcasted - z_broadcasted)).sum({2,3,4});

    // Apply the mask to commonfactor before the division
    radial_values_2d = (commonfactor * mask * (r_sampling_broadcasted - r_broadcasted * sintheta_broadcasted)).sum({2,3,4});


    // Convert the result to vector of vector of doubles
    auto radial_values = tensor_to_vec_of_vec(radial_values_2d);
    auto vertical_values = tensor_to_vec_of_vec(vertical_values_2d);

    return std::make_pair(radial_values, vertical_values);
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
              int GPU_ID,
              bool debug) {


    torch::Device device(torch::kCUDA, GPU_ID);
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
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

    auto G = torch::full({1}, 7.456866768350099e-46 * (1 + redshift), options);
    return get_g_torch(r_sampling, z_sampling, G, dv0, r, z, costheta, sintheta, rho, debug);
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

//std::vector<double> calculate_rotational_velocity(double redshift, const std::vector<double> &dv0,
//                                                  std::vector<double> r_sampling,
//                                                  const std::vector<double> &r,
//                                                  const std::vector<double> &z,
//                                                  const std::vector<double> &costheta,
//                                                  const std::vector<double> &sintheta,
//                                                  const std::vector<std::vector<double>> &rho, bool debug, int GPU_ID, bool cuda) {
//    int nr_sampling = r_sampling.size();
//    double km_lyr = 9460730472580.8; //uu.lyr.to(uu.km)
//    // Allocate result vector
//    std::vector<double> z_sampling = {0.0};
//    std::vector<double> v_r(nr_sampling,0.0);
//    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z;
//    if(cuda){
//        f_z = get_all_torch(redshift, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, GPU_ID, debug);
//    }
//    else {
//        f_z = get_all_g(redshift, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, debug);
//    }
//
//    // Calculate velocities
//    double v_squared;
//    for (int i = 0; i < nr_sampling; i++) {
//        v_squared = f_z.first[i][0] * r_sampling[i] * km_lyr; // Access radial values from the pair (first element)
//        v_r[i] = sqrt(v_squared); // 9460730777119.56 km
//
//        // Debugging output
//        if (debug) {
//            std::cout << "r_sampling[" << i << "]: " << r_sampling[i] << std::endl;
//            std::cout << "f_z.first[" << i << "][0]: " << f_z.first[i][0] << std::endl;
//            std::cout << "v_squared: " << v_squared << std::endl;
//            std::cout << "v_r[" << i << "]: " << v_r[i] << std::endl;
//        }
//    }
//    // Return result
//    return v_r;
//}

std::vector<double> calculate_rotational_velocity(const galaxy& galaxy, const std::vector<std::vector<double>> &rho) {
    int nr_sampling = galaxy.x_rotation_points.size();
    double km_lyr = 9460730472580.8; //uu.lyr.to(uu.km)
    // Allocate result vector
    std::vector<double> z_sampling = {0.0};
    std::vector<double> v_r(nr_sampling,0.0);
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z;

    if(galaxy.cuda){
        f_z = get_all_torch(galaxy.redshift, galaxy.dv0, galaxy.x_rotation_points, z_sampling,
                            galaxy.r, galaxy.z, galaxy.costheta, galaxy.sintheta, rho, galaxy.GPU_ID, galaxy.debug);
    }
    else {
        f_z = get_all_g(galaxy.redshift, galaxy.dv0, galaxy.x_rotation_points, z_sampling,
                        galaxy.r, galaxy.z, galaxy.costheta, galaxy.sintheta, rho, galaxy.debug);
    }

    // Calculate velocities
    double v_squared;
    for (int i = 0; i < nr_sampling; i++) {
        v_squared = f_z.first[i][0] * galaxy.x_rotation_points[i] * km_lyr; // Access radial values from the pair (first element)
        v_r[i] = sqrt(v_squared); // 9460730777119.56 km

        // Debugging output
        if (galaxy.debug) {
            std::cout << "r_sampling[" << i << "]: " << galaxy.x_rotation_points[i] << std::endl;
            std::cout << "f_z.first[" << i << "][0]: " << f_z.first[i][0] << std::endl;
            std::cout << "v_squared: " << v_squared << std::endl;
            std::cout << "v_r[" << i << "]: " << v_r[i] << std::endl;
        }
    }
    // Return result
    return v_r;
}



double calculate_mass(double rho, double alpha, double h) {
    double factor = 0.0007126927557971729; // factor takes care of moving from rho as atom/cc to kg/lyr^3, with alpha = 1/lyr and h0 = in lyr div sun_mass
    double Mtotal_si = 2 * M_PI * h * rho /(alpha*alpha); //where h is in lyr and alpha is in 1/lyr
    return Mtotal_si*factor;
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
