#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <future>
#include <torch/types.h>
#include <typeinfo>
#include "tensor_utils.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <taskflow/taskflow.hpp>


std::vector<double> geomspace(double start, double stop, int num) {
    std::vector<double> result;

    if (num <= 0) return result; // Empty vector if num is not positive

    double base = std::log(stop / start) / (num - 1);

    for (int i = 0; i < num; ++i) {
        result.push_back(start * std::exp(i * base));
    }

    return result;
}

bool has_nan(const std::vector<std::vector<double>>& v) {
    for (const auto& inner_vector : v) {
        for (const auto& element : inner_vector) {
            if (std::isnan(element)) {
                return true;
            }
        }
    }
    return false;
}




std::vector<std::array<double, 2>> interpolate(const std::vector<std::array<double, 2>>& input, size_t num_points) {
    std::vector<std::array<double, 2>> output(num_points);
    double x_min = input[0][0];
    double x_max = input.back()[0];
    double dx = (x_max - x_min) / (num_points - 1);

    for(size_t i = 0; i < num_points; ++i) {
        double x = x_min + dx * i;
        auto it = std::lower_bound(input.begin(), input.end(), std::array<double, 2>{x, 0.0}, [](const std::array<double, 2>& a, const std::array<double, 2>& b){ return a[0] < b[0]; });
        if(it == input.end()) {
            output[i] = {x, input.back()[1]};
        } else if(it == input.begin()) {
            output[i] = {x, input[0][1]};
        } else {
            double x1 = (it-1)->at(0);
            double y1 = (it-1)->at(1);
            double x2 = it->at(0);
            double y2 = it->at(1);
            output[i] = {x, y1 + (y2 - y1) * (x - x1) / (x2 - x1)}; // linear interpolation
        }
    }

    return output;
}



std::string getCudaString(bool cuda, bool taskflow) {
    if (cuda) {
        if(taskflow){
            return "GPU_Torch_Chunks";
        } else{
            return "GPU_Torch_No_Chunks";
        }
    } else if(taskflow) {
        return "CPU_TaskFlow";
    } else{
        return "CPU_Futures";
    }
}


std::string get_device_util(at::Tensor tensor) {
    if (tensor.device() == at::kCPU) {
        return "Tensor is on the CPU.";
    } else if (tensor.device() == at::kCUDA) {
        return "Tensor is on the GPU.";
    } else {
        return "Tensor is on an unknown device.";
    }
}



std::vector<double> calculate_density_parameters(double redshift){
//    Fitting coefficients for log(rho_0) versus log(r4d):
//    Slope: -2.958649379487641
//    Intercept: 4.66200406232113
//
//    Fitting coefficients for log(alpha_0) versus log(r4d):
//    Slope: -0.9895204320261537
//    Intercept: -2.198902650547498
//
//    Fitting coefficients for log(rho_1) versus log(r4d):
//    Slope: -3.0101301080291396
//    Intercept: 2.567935487552146
//
//    Fitting coefficients for log(alpha_1) versus log(r4d):
//    Slope: -1.0271431297841869
//    Intercept: -3.572070693277129
//
//    Fitting coefficients for log(h0) versus log(r4d):
//    Slope: 0.9780589482263441
//    Intercept: 3.9804724134564493
    double r4d = 1 / (1 + redshift);
    std::vector<double> values{
//            pow(r4d, -2.9438845606949298) * 10.102708507336727,
//            pow(r4d, -0.9591355997183575) * 0.00036652348928414505,
//            pow(r4d, -2.8081721846475487) * 0.18453605105707863,
//            pow(r4d, -0.7229538634491799) * 3.432419089184575e-05,
//            pow(r4d, 1.0252075729451395) * 148647.66101686007,
//
//            pow(r4d, -3.000000000000001) * 10.748808195190971,
//            pow(r4d, -1.0000000000000004) * 0.0003828200322686644,
//            pow(r4d, -2.9999999999999996) * 0.20985872041386153,
//            pow(r4d, -0.9999999999999968) * 3.4931539070167785e-05,
//            pow(r4d, 0.9999999999999983) * 152301.7081641162,

            pow(r4d, -2.942043736411602) * 10.130004656180187,
            pow(r4d, -0.9586430173804117) * 0.00036679488581632983,
            pow(r4d, -2.8116752243088303) * 0.18361646621629862,
            pow(r4d, -0.7259713529991378) * 3.417717871690614e-05,
            pow(r4d, 1.024993059207669) * 148468.15438408472,
    };
    return values;
}


std::vector<std::array<double, 2>> move_rotation_curve(std::vector<std::array<double, 2>>& rotation_curve, double z1, double z2) {
    double rescaling_factor = (1 + z2) / (1 + z1);
    std::vector<std::array<double, 2>> result(rotation_curve.size());

    for (size_t i = 0; i < rotation_curve.size(); ++i) {
        result[i][0]=rotation_curve[i][0]/rescaling_factor;
        result[i][1]=rotation_curve[i][1]*rescaling_factor;
    }
    return result;
}


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

void print_tensor_shape(const torch::Tensor& tensor) {
    torch::IntArrayRef shape = tensor.sizes();

    std::cout << "Shape of the tensor: ";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;
}

void print_tensor_dimensionality(const torch::Tensor& tensor){
    std::cout << "Dimensionality: " << tensor.dim() << std::endl;
    std::cout << "Shape: ";
    for (int i = 0; i < tensor.dim(); ++i) {
        std::cout << tensor.size(i) << " ";
    }
    std::cout << std::endl;
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
            number_density[i][j] = local_density[i][j] / lyr3_to_m3 / hydrogen_mass + 1E-6;
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

std::pair<torch::Tensor, torch::Tensor> compute_chunk(
        const torch::Tensor& r_sampling,
        const torch::Tensor& z_sampling,
        const torch::Tensor& r_broadcasted,
        const torch::Tensor& dv0_broadcasted,
        const torch::Tensor& G_broadcasted,
        const torch::Tensor& rho_broadcasted,
        const torch::Tensor& sintheta_broadcasted,
        const torch::Tensor& costheta_broadcasted,
        const torch::Tensor& z_broadcasted
        ) {

    auto r_sampling_broadcasted = r_sampling.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4);
    auto z_sampling_broadcasted = z_sampling.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4);

    // Create masks for radial value calculation
    auto mask = (r_broadcasted <= r_sampling_broadcasted).to(r_sampling_broadcasted.dtype());

    // Calculate the distances
//    print_tensor_shape(z_sampling_broadcasted);
//    print_tensor_shape(z_broadcasted);
//    print_tensor_shape(r_sampling_broadcasted);
//    print_tensor_shape(r_broadcasted);
//    print_tensor_shape(sintheta_broadcasted);
//    print_tensor_shape(costheta_broadcasted);
//    print_tensor_shape(rho_broadcasted);
//    print_tensor_shape(G_broadcasted);

    auto d_3 = ( (z_sampling_broadcasted - z_broadcasted).pow(2) +
                 (r_sampling_broadcasted - r_broadcasted * sintheta_broadcasted).pow(2) +
                 (r_broadcasted * costheta_broadcasted).pow(2) ).pow(1.5);

    // Calculate the common factor
    auto commonfactor = G_broadcasted * rho_broadcasted * dv0_broadcasted/d_3;

    // Perform the summation over the last three dimensions
    auto vertical_values_2d = (commonfactor * (z_sampling_broadcasted - z_broadcasted)).sum({2,3,4});
    // Apply the mask to commonfactor before the division
    auto radial_values_2d = (commonfactor * mask * (r_sampling_broadcasted - r_broadcasted * sintheta_broadcasted)).sum({2,3,4}) ;

    // Force immediate deletion of intermediate tensors
    d_3 = torch::Tensor();
    commonfactor = torch::Tensor();

    torch::Device device_cpu(torch::kCPU);
    return std::make_pair(radial_values_2d.to(device_cpu), vertical_values_2d.to(device_cpu));
//
//    // Move to CPU
//    auto radial_values_2d_cpu = radial_values_2d.to(torch::kCPU);
//    auto vertical_values_2d_cpu = vertical_values_2d.to(torch::kCPU);
//// Delete from GPU
//    radial_values_2d = torch::Tensor();
//    vertical_values_2d = torch::Tensor();
//    return std::make_pair(radial_values_2d_cpu, vertical_values_2d_cpu);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_all_torch_chunks(double redshift,
                     const std::vector<double> &dv0_in,
                     const std::vector<double> &r_sampling_in,
                     const std::vector<double> &z_sampling_in,
                     const std::vector<double> &r_in,
                     const std::vector<double> &z_in,
                     const std::vector<double> &costheta_in,
                     const std::vector<double> &sintheta_in,
                     const std::vector<std::vector<double>> &rho_in,
                     int GPU_ID
) {
    //###########################################
    //    Acceleration calculated in km/s^2
    //###########################################
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

    // Get results from get_g_torch
    // Get the sizes for each dimension
    int r_size = r_sampling.size(0);
    int z_size = z_sampling.size(0);
    int n_r = r.size(0);
    int n_z = z.size(0);
    int ntheta = costheta.size(0);

    // Initialize the output tensors with the correct dimensions
    torch::Tensor radial_values_2d = torch::zeros({r_size, z_size});
    torch::Tensor vertical_values_2d = torch::zeros({r_size, z_size});
    // Ensure that the tensors are on the CPU
    torch::Device device_cpu(torch::kCPU);
    radial_values_2d = radial_values_2d.to(device_cpu);
    vertical_values_2d = vertical_values_2d.to(device_cpu);


    // Broadcasting other tensors before chunking
    auto r_broadcasted = r.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4);
    auto dv0_broadcasted = dv0.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4);
    auto rho_broadcasted = rho.unsqueeze(0).unsqueeze(1).unsqueeze(3);
    auto G_broadcasted = G.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4);
    auto sintheta_broadcasted = sintheta.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4);
    auto costheta_broadcasted = costheta.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(4);
    auto z_broadcasted = z.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3);



    int chunk_r_size =  r_size;
    int chunk_z_size=1;

    double available_memory_bytes = 11.0 * 1024 * 1024 * 1024; // 12 GB
    int n_memory = 7;
    double total_memory_bytes = (n_memory * n_r * n_z * ntheta * chunk_r_size * chunk_z_size) * 4;
    while (total_memory_bytes > available_memory_bytes) {
        chunk_r_size -= 1;
//        std::cout << chunk_r_size << std::endl;
        total_memory_bytes = (n_memory * n_r * n_z * ntheta * chunk_r_size * chunk_z_size) * 4; // times 2 for two tensors
    }

    // Split r_sampling and z_sampling tensors into chunks and process each chunk separately
    for (int i = 0; i < r_size; i += chunk_r_size) {
        for (int j = 0; j < z_size; j += 1) {
            int r_end = std::min(i + chunk_r_size, r_size);
            int z_end = std::min(j + chunk_z_size, z_size);

            auto r_sampling_chunk = r_sampling.slice(0, i, r_end).clone();
            auto z_sampling_chunk = z_sampling.slice(0, j, z_end).clone();

            auto radial_vertical_chunk = compute_chunk(
                    r_sampling_chunk, z_sampling_chunk, r_broadcasted, dv0_broadcasted, G_broadcasted,
                    rho_broadcasted, sintheta_broadcasted, costheta_broadcasted, z_broadcasted
            );

//            std::cout << get_device_util(radial_vertical_chunk.first) << std::endl; //cpu
//            std::cout << get_device_util(radial_vertical_chunk.second) << std::endl; //cpu

            try {
                // code that might throw an exception
                radial_values_2d.slice(0, i, r_end).slice(1, j, z_end).add_(radial_vertical_chunk.first);
                vertical_values_2d.slice(0, i, r_end).slice(1, j, z_end).add_(radial_vertical_chunk.second);
            } catch (const std::exception& e) {
                // handle exception
                std::cerr << "Caught exception: " << e.what() << std::endl;
            } catch (...) {
                // catch-all handler: can catch any exception not caught by earlier handlers
                std::cerr << "Caught unknown exception" << std::endl;
            }

            // Force immediate deletion of chunk tensors
            r_sampling_chunk = torch::Tensor();
            z_sampling_chunk = torch::Tensor();
            radial_vertical_chunk.first = torch::Tensor();
            radial_vertical_chunk.second = torch::Tensor();
            // Empty the cache inside the try block
            c10::cuda::CUDACachingAllocator::emptyCache(); // This line moved

        }
    }

    // Convert the result to vector of vector of doubles
    auto radial_values = tensor_to_vec_of_vec(radial_values_2d);
    auto vertical_values = tensor_to_vec_of_vec(vertical_values_2d);



//    std::cout << get_device_util(dv0 ) << std::endl;
//    std::cout << get_device_util(r_sampling ) << std::endl;
//    std::cout << get_device_util(z_sampling ) << std::endl;
//    std::cout << get_device_util(r ) << std::endl;
//    std::cout << get_device_util(z ) << std::endl;
//    std::cout << get_device_util(costheta ) << std::endl;
//    std::cout << get_device_util(sintheta ) << std::endl;
//    std::cout << get_device_util(rho ) << std::endl;
//    std::cout << get_device_util(G ) << std::endl;
//    std::cout << get_device_util(r_broadcasted ) << std::endl;
//    std::cout << get_device_util(dv0_broadcasted ) << std::endl;
//    std::cout << get_device_util(rho_broadcasted ) << std::endl;
//    std::cout << get_device_util(G_broadcasted ) << std::endl;
//    std::cout << get_device_util(sintheta_broadcasted ) << std::endl;
//    std::cout << get_device_util(costheta_broadcasted ) << std::endl;
//    std::cout << get_device_util(z_broadcasted ) << std::endl;
//    std::cout << get_device_util(radial_values_2d ) << std::endl; //cpu
//    std::cout << get_device_util(vertical_values_2d ) << std::endl; //cpu



    // Delete tensors
    dv0 = torch::Tensor();
    r_sampling = torch::Tensor();
    z_sampling = torch::Tensor();
    r = torch::Tensor();
    z = torch::Tensor();
    costheta = torch::Tensor();
    sintheta = torch::Tensor();
    rho = torch::Tensor();
    G = torch::Tensor();
    r_broadcasted = torch::Tensor();
    dv0_broadcasted = torch::Tensor();
    rho_broadcasted = torch::Tensor();
    G_broadcasted = torch::Tensor();
    sintheta_broadcasted = torch::Tensor();
    costheta_broadcasted = torch::Tensor();
    z_broadcasted = torch::Tensor();

    // Delete tensors and empty cache
    dv0.reset();
    r_sampling.reset();
    z_sampling.reset();
    r.reset();
    z.reset();
    costheta.reset();
    sintheta.reset();
    rho.reset();
    G.reset();
    c10::cuda::CUDACachingAllocator::emptyCache();
    return std::make_pair(radial_values, vertical_values);
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
        const torch::Tensor& rho
        ) {

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

    c10::cuda::CUDACachingAllocator::emptyCache();
    return std::make_pair(radial_values, vertical_values);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_all_torch_no_chunks(double redshift,
              const std::vector<double> &dv0_in,
              const std::vector<double> &r_sampling_in,
              const std::vector<double> &z_sampling_in,
              const std::vector<double> &r_in,
              const std::vector<double> &z_in,
              const std::vector<double> &costheta_in,
              const std::vector<double> &sintheta_in,
              const std::vector<std::vector<double>> &rho_in,
              int GPU_ID
              ) {
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

    // Get results from get_g_torch
    auto result = get_g_torch(r_sampling, z_sampling, G, dv0, r, z, costheta, sintheta, rho);

    // Delete tensors and empty cache
    dv0.reset();
    r_sampling.reset();
    z_sampling.reset();
    r.reset();
    z.reset();
    costheta.reset();
    sintheta.reset();
    rho.reset();
    G.reset();
    c10::cuda::CUDACachingAllocator::emptyCache();
    return result;
}



// # CPU functions
std::pair<double, double> get_g_cpu(double r_sampling_ii, double z_sampling_jj, double G,
                                    const std::vector<double> &dv0, const std::vector<double> &r,
                                    const std::vector<double> &z, const std::vector<double> &costheta,
                                    const std::vector<double> &sintheta, const std::vector<std::vector<double>> &rho) {
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
                double d_3 =pow( (z[j] - z_sampling_jj) * (z[j] - z_sampling_jj) +
                             (r_sampling_ii - r[i] * sintheta[k]) * (r_sampling_ii - r[i] * sintheta[k])+
                             r[i] * r[i] * costheta[k] * costheta[k], 1.5);
                double commonfactor  = G * rho[i][j] *  dv0[i]  / d_3;
                if ( r[i] < r_sampling_ii) {
                    thisradial_value = commonfactor * (r_sampling_ii - r[i] * sintheta[k]);
                    radial_value += thisradial_value;
                }
                thisvertical_value = commonfactor * (z_sampling_jj - z[j]);
                vertical_value += thisvertical_value;
            }
        }
    }
    return std::make_pair(radial_value, vertical_value);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_all_g_thread(tf::Taskflow& tf, double redshift, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
                 const std::vector<double> &z_sampling,
                 const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
                 const std::vector<double> &sintheta, const std::vector<std::vector<double>> &rho) {
    double G = 7.456866768350099e-46 * (1 + redshift);

    int nr_local = r_sampling.size();
    int nz_local = z_sampling.size();

    std::vector<std::vector<double>> f_z_radial(nr_local, std::vector<double>(nz_local, 0));
    std::vector<std::vector<double>> f_z_vertical(nr_local, std::vector<double>(nz_local, 0));

    std::vector<std::vector<tf::Task>> tasks(nr_local, std::vector<tf::Task>(nz_local));

    for (unsigned int i = 0; i < nr_local; i++) {
        for (unsigned int j = 0; j < nz_local; j++) {
            tasks[i][j] = tf.emplace([&, i, j]() {
                auto result_pair = get_g_cpu(r_sampling[i], z_sampling[j], G, dv0, r, z, costheta, sintheta, rho);
                f_z_radial[i][j] = result_pair.first;
                f_z_vertical[i][j] = result_pair.second;
            });
        }
    }

    tf::Executor executor;
    executor.run(tf).get();

    return std::make_pair(f_z_radial, f_z_vertical);
}


std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_all_g(double redshift, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
          const std::vector<double> &z_sampling,
          const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
          const std::vector<double> &sintheta, const std::vector<std::vector<double>> &rho) {
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
                    std::async(get_g_cpu, r_sampling[i], z_sampling[j], G, dv0, r, z, costheta, sintheta, rho));
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




