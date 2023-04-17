





void deleteCUDAVector(double *x_vector_device) {
    cudaFree(x_vector_device);
}

std::pair<dim3, dim3> get_block_size(int n_i, int n_j, int threads_per_block) {
    int blocks_per_grid_x = (n_j + threads_per_block - 1) / threads_per_block;
    int blocks_per_grid_y = (n_i + threads_per_block - 1) / threads_per_block;
    dim3 block_size(threads_per_block, threads_per_block, 1);
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y, 1);
    return std::make_pair(num_blocks, block_size);
}

//std::pair<dim3, dim3> get_block_size3(int n_i, int n_j, int n_k, int threads_per_block) {
//    int blocks_per_grid_x = (n_j + threads_per_block - 1) / threads_per_block;
//    int blocks_per_grid_y = (n_i + threads_per_block - 1) / threads_per_block;
//    int blocks_per_grid_z = (n_k + threads_per_block - 1) / threads_per_block;
//    dim3 block_size(threads_per_block, threads_per_block, threads_per_block);
//    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y, threads_per_block);
//    return std::make_pair(num_blocks, block_size);
//}


void check_device_memory_allocation(const void *device_pointer, const std::string &name) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Error: Unable to allocate memory on device for " << name << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Allocated memory on device for " << name << std::endl;
    }
}

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}




double *createCUDAVector(const double *hostPointer, int n) {
    double *devicePointer;
    cudaMalloc((void **) &devicePointer, n * sizeof(double));
    cudaMemcpy(devicePointer, hostPointer, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Error copying host pointer to device: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    return devicePointer;
}

double *createCUDAVector(const std::vector<double> &hostVector) {
    double *devicePointer;
    cudaMalloc((void **) &devicePointer, hostVector.size() * sizeof(double));
    cudaMemcpy(devicePointer, hostVector.data(), hostVector.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Error copying host vector to device: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    return devicePointer;
}

















std::pair<dim3, dim3> get_block_size(int n, int threads_per_block) {
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    dim3 block_size(threads_per_block, 1, 1);
    dim3 num_blocks(blocks_per_grid, 1, 1);
    return std::make_pair(num_blocks, block_size);
}



void deleteCUDAVector(double *x_vector_host, double *x_vector_device) {
    delete[] x_vector_host;
    cudaFree(x_vector_device);
}


std::vector<std::vector<double>>
get_all_g(double redshift, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
          const std::vector<double> &z_sampling,
          const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
          const std::vector<double> &sintheta, const std::vector<double> &rho, bool debug, bool radial = true, bool cuda = true) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    // G = cc.G.to(uu.km / uu.s ** 2 / uu.kg * uu.lyr ** 2).value *(1+redshift)
    double G = 7.456866768350099e-46 * (1 + redshift);
//    if (device_count > 0 && cuda) {
//        // If CUDA devices are available, use the CUDA implementation
//        return get_all_g_impl_cuda(G, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, debug, radial);
//    } else {
    // If no CUDA devices are available, use the CPU implementation
    return get_all_g_impl_cpu(G, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, debug);
//    }
}



__global__ void get_all_g_kernel(int nr, int nz, int ntheta, int nr_sampling, int nz_sampling, double G,
                                 bool radial, double *f_z, bool debug) {

    // Get the indices of the point in r and z for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Use shared memory for caching intermediate results
    extern __shared__ double shared_data[];

    if (i < nr_sampling && j < nz_sampling) {
        int idx = i + j * nr_sampling;
        // Initialize the result variable for this thread
        double res = 0.0;
        double thisres = 0.0;
        // Loop over all r and z points in the grid
        for (int ir = 0; ir < nr; ir++) {
            double r_ir = const_grid_data[ir];
            double dv0_ir = const_grid_data[nr + nz + 2 * ntheta + ir];
            double rho_ir = const_grid_data[2 * nr + nz + 2 * ntheta + ir];
            if (radial && (r_ir > const_grid_data[3 * nr + nz + 2 * ntheta + i])) {
                break;
            }
            for (int iz = 0; iz < nz; iz++) {
                double z_iz = const_grid_data[nr + iz];
                for (int k = 0; k < ntheta; k++) {
                    double costheta_k = const_grid_data[nr + nz + k];
                    double sintheta_k = const_grid_data[nr + nz + ntheta + k];
                    double r_sampling_i = const_grid_data[3 * nr + nz + 2 * ntheta + i];
                    double z_sampling_j = const_grid_data[3 * nr + nr_sampling + nz + 2 * ntheta + j];

                    // Compute the distance between the sampling point and the point in r and z for this thread
                    double d_2 = (z_iz - z_sampling_j) * (z_iz - z_sampling_j) +
                                 (r_sampling_i - r_ir * sintheta_k) * (r_sampling_i - r_ir * sintheta_k) +
                                 r_ir * r_ir * costheta_k * costheta_k;
                    double d_1 = sqrt(d_2);
                    double d_3 = d_1 * d_1 * d_1;

                    //                    d = pow(z[j] - z_sampling_jj, 2.0) + pow(r_sampling_ii - r[i] * sintheta[k], 2.0) +
//                        r[i] * r[i] * costheta[k] * costheta[k];
                    if (radial) {
                        thisres= G * rho_ir * r_ir * dv0_ir *(r_sampling_i - r_ir * sintheta_k)/d_3;
//                        thisres = G * rho[i] * r[i] * dv0[i] * (r_sampling_ii - r[i] * sintheta[k]) / pow(d, 1.5);
                        res += thisres;
                    } else {
                        thisres = G * rho_ir * r_ir * dv0_ir *(z_iz - z_sampling_j )/d_3;;
                        res += thisres;
//                        thisres = G * rho[i] * r[i] * dv0[i] * (z[j] - z_sampling_jj) / pow(d, 1.5);
                    }

                    if(debug){
                        if (ir==5 && iz==5 && k == 5) {
//                            std::vector<double> shared_data(r);
//                            shared_data.insert(shared_data.end(), z.begin(), z.end());
//                            shared_data.insert(shared_data.end(), costheta.begin(), costheta.end());
//                            shared_data.insert(shared_data.end(), sintheta.begin(), sintheta.end());
//                            shared_data.insert(shared_data.end(), dv0.begin(), dv0.end());
//                            shared_data.insert(shared_data.end(), rho.begin(), rho.end());
//                            shared_data.insert(shared_data.end(), r_sampling.begin(), r_sampling.end());
//                            shared_data.insert(shared_data.end(), z_sampling.begin(), z_sampling.end());
                            printf("GPU \n");
                            printf("The value of f_z is %e\n", thisres);
                            printf("The value of distance is %fd\n", d_1);
                            printf("The value of r[i] is %fd\n", r_ir);
                            printf("The value of z[j] is %fd\n", z_iz);
                            printf("The value of costheta is %fd\n", costheta_k);
                            printf("The value of sintheta is %fd\n", sintheta_k);
                            printf("The value of dv0 is %fd\n", dv0_ir);
                            printf("The value of rho is %e\n", rho_ir);
                            printf("The value of rsampling is %fd\n", r_sampling_i);
                            printf("The value of zsampling is %fd\n", z_sampling_j);
                            printf("The value of G is %e\n", G);
                        }
                    }

                }
            }
        }
        // Store the result variable for this thread in the output array
        f_z[idx] = res;
    }
}




std::vector<double> get_all_g_impl_cuda(double G, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
                                        const std::vector<double> &z_sampling, const std::vector<double> &r,
                                        const std::vector<double> &z,
                                        const std::vector<double> &costheta, const std::vector<double> &sintheta,
                                        const std::vector<double> &rho, bool debug,
                                        bool radial = true) {
    // Combine r, z, costheta, sintheta, and dv0 into a single vector (grid_data) for cudaMemcpy
    std::vector<double> grid_data(r);
    grid_data.insert(grid_data.end(), z.begin(), z.end());
    grid_data.insert(grid_data.end(), costheta.begin(), costheta.end());
    grid_data.insert(grid_data.end(), sintheta.begin(), sintheta.end());
    grid_data.insert(grid_data.end(), dv0.begin(), dv0.end());
    grid_data.insert(grid_data.end(), rho.begin(), rho.end());
    grid_data.insert(grid_data.end(), r_sampling.begin(), r_sampling.end());
    grid_data.insert(grid_data.end(), z_sampling.begin(), z_sampling.end());

    if (grid_data.size() > MAX_GRID_DATA_SIZE) {
        // Handle error, e.g., return an empty vector or throw an exception
        std::cout << "too large grid_data ";
        return std::vector<double>();
    }

    int nr_sampling = r_sampling.size();
    int nz_sampling = z_sampling.size();
    int nr = r.size();
    int nz = z.size();

    int gridsize = grid_data.size();
    std::cout << "gridsize " << gridsize <<"\n";
    std::cout << "nr " << nr <<"\n";
    std::cout << "nz " << nz <<"\n";
    std::cout << "nr_sampling " << nr_sampling <<"\n";
    std::cout << "nz_sampling " << nz_sampling <<"\n";
    std::cout << "rho_size " << nr <<"\n";
    std::cout << "dv0_size " << nr <<"\n";
    std::cout << "costheta.size " << costheta.size() <<"\n";
    std::cout << "sintheta.size " << costheta.size() <<"\n";

    // Copy grid_data to const_grid_data
    cudaError_t err = cudaMemcpyToSymbol(const_grid_data, grid_data.data(), grid_data.size() * sizeof(double));
    if (err != cudaSuccess) {
        printf("Error copying data to constant memory: %s\n", cudaGetErrorString(err));
        return std::vector<double>();
    }



    // Allocate and copy device memory
    double *f_z = new double(nr_sampling * nz_sampling);
//    double *dev_grid_data = createCUDAVector(grid_data);
//    double *dev_rho = createCUDAVector(rho);
    double *dev_f_z = createCUDAVector(f_z, nr_sampling * nz_sampling);
//    double *dev_dv0 = createCUDAVector(dv0);



// Get the block size for the tensor product kernel
    // Get the block size for the tensor product kernel
    int ntheta = costheta.size();
    int threads_per_block = 32;
    std::pair<dim3, dim3> block_info = get_block_size(nr_sampling * nz_sampling, threads_per_block);
    dim3 num_blocks = block_info.first;
    dim3 block_size = block_info.second;


    // Launch kernel
    get_all_g_kernel<<<num_blocks, block_size>>>(nr, nz,ntheta, nr_sampling, nz_sampling, G,
                                                 radial, dev_f_z, debug);
    // Copy results back to host
    std::vector<double> f_z_vec(nr_sampling * nz_sampling);
    cudaMemcpy(f_z_vec.data(), dev_f_z, nr_sampling * nz_sampling * sizeof(double), cudaMemcpyDeviceToHost);


    // Free device memory
    cudaFree(dev_f_z);

    // Return result
    return f_z_vec;
}

// CUDA kernel to compute the gravitational acceleration f_z
// for points in r_sampling and z_sampling
//__global__ void get_all_g_kernelNew(int nr, int nz, int ntheta, int nr_sampling, int nz_sampling, double G,
//                                 bool radial, double *f_z, bool debug) {
//
//    // Get the indices of the point in r and z for this thread
//    int i = blockIdx.x * blockDim.x + threadIdx.x;  // r
//    int j = blockIdx.y * blockDim.y + threadIdx.y;  // z
//
//    if (i < nr_sampling && j < nz_sampling) {
//        int idx = i + j * nr_sampling;
//        // Initialize the result variable for this thread
//        double res = 0.0;
//        double thisres = 0.0;
//        // Loop over all r, z, and theta points in the grid
//        for (int ir = 0; ir < nr; ir++) {
//            double r_ir = const_grid_data[i];
//            double dv0_ir = const_grid_data[nr + nz + 2 * ntheta + i];
//            double rho_ir = const_grid_data[2 * nr + nz + 2 * ntheta + i];
//            if (radial && (r_ir > const_grid_data[3 * nr + nz + 2 * ntheta + i])) {
//                break;
//            }
//            for (int iz = 0; iz < nz; iz++) {
//                double z_iz = const_grid_data[nr + j];
//                for (int k = 0; k < ntheta; k++) {
//                    double costheta_k = const_grid_data[nr + nz + k];
//                    double sintheta_k = const_grid_data[nr + nz + ntheta + k];
//                    double r_sampling_i = const_grid_data[3 * nr + nz + 2 * ntheta + i];
//                    double z_sampling_j = const_grid_data[3 * nr + nr_sampling + nz + 2 * ntheta + j];
//
//                    // Compute the distance between the sampling point and the point in r and z for this thread
//                    double d_2 = (z_iz - z_sampling_j) * (z_iz - z_sampling_j) +
//                                 (r_sampling_i - r_ir * sintheta_k) * (r_sampling_i - r_ir * sintheta_k) +
//                                 r_ir * r_ir * costheta_k * costheta_k;
//                    double d_1 = sqrt(d_2);
//                    double d_3 = d_1 * d_1 * d_1;
//
//                    //                    d = pow(z[j] - z_sampling_jj, 2.0) + pow(r_sampling_ii - r[i] * sintheta[k], 2.0) +
////                        r[i] * r[i] * costheta[k] * costheta[k];
//                    if (radial) {
//                        thisres = G * rho_ir * r_ir * dv0_ir * (r_sampling_i - r_ir * sintheta_k) / d_3;
////                        thisres = G * rho[i] * r[i] * dv0[i] * (r_sampling_ii - r[i] * sintheta[k]) / pow(d, 1.5);
//                        res += thisres;
//                    } else {
//                        thisres = G * rho_ir * r_ir * dv0_ir * (z_iz - z_sampling_j) / d_3;;
//                        res += thisres;
////                        thisres = G * rho[i] * r[i] * dv0[i] * (z[j] - z_sampling_jj) / pow(d, 1.5);
//                    }
//
//                    if (debug) {
//                        if (ir == 5 && iz == 5 && k == 5) {
////                            std::vector<double> shared_data(r);
////                            shared_data.insert(shared_data.end(), z.begin(), z.end());
////                            shared_data.insert(shared_data.end(), costheta.begin(), costheta.end());
////                            shared_data.insert(shared_data.end(), sintheta.begin(), sintheta.end());
////                            shared_data.insert(shared_data.end(), dv0.begin(), dv0.end());
////                            shared_data.insert(shared_data.end(), rho.begin(), rho.end());
////                            shared_data.insert(shared_data.end(), r_sampling.begin(), r_sampling.end());
////                            shared_data.insert(shared_data.end(), z_sampling.begin(), z_sampling.end());
//                            printf("GPU \n");
//                            printf("The value of f_z is %e\n", thisres);
//                            printf("The value of distance is %fd\n", d_1);
//                            printf("The value of r[i] is %fd\n", r_ir);
//                            printf("The value of z[j] is %fd\n", z_iz);
//                            printf("The value of costheta is %fd\n", costheta_k);
//                            printf("The value of sintheta is %fd\n", sintheta_k);
//                            printf("The value of dv0 is %fd\n", dv0_ir);
//                            printf("The value of rho is %e\n", rho_ir);
//                            printf("The value of rsampling is %fd\n", r_sampling_i);
//                            printf("The value of zsampling is %fd\n", z_sampling_j);
//                            printf("The value of G is %e\n", G);
//                        }
//                    }
//                }
//            }
//        }
//        atomicAddDouble(&f_z[idx], res);
//    }
//}

//std::vector<double> get_all_g_impl_cudaNew(double G, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
//                                        const std::vector<double> &z_sampling, const std::vector<double> &r,
//                                        const std::vector<double> &z,
//                                        const std::vector<double> &costheta, const std::vector<double> &sintheta,
//                                        const std::vector<double> &rho, bool debug,
//                                        bool radial = true) {
//    // Combine r, z, costheta, sintheta, and dv0 into a single vector (grid_data) for cudaMemcpy
//    std::vector<double> grid_data(r);
//    grid_data.insert(grid_data.end(), z.begin(), z.end());
//    grid_data.insert(grid_data.end(), costheta.begin(), costheta.end());
//    grid_data.insert(grid_data.end(), sintheta.begin(), sintheta.end());
//    grid_data.insert(grid_data.end(), dv0.begin(), dv0.end());
//    grid_data.insert(grid_data.end(), rho.begin(), rho.end());
//    grid_data.insert(grid_data.end(), r_sampling.begin(), r_sampling.end());
//    grid_data.insert(grid_data.end(), z_sampling.begin(), z_sampling.end());
//
//    if (grid_data.size() > MAX_GRID_DATA_SIZE) {
//        // Handle error, e.g., return an empty vector or throw an exception
//        std::cout << "too large grid_data ";
//        return std::vector<double>();
//    }
//
//    int nr_sampling = r_sampling.size();
//    int nz_sampling = z_sampling.size();
//    int nr = r.size();
//    int nz = z.size();
//
//    // Copy grid_data to const_grid_data
//    cudaError_t err = cudaMemcpyToSymbol(const_grid_data, grid_data.data(), grid_data.size() * sizeof(double));
//    if (err != cudaSuccess) {
//        printf("Error copying data to constant memory: %s\n", cudaGetErrorString(err));
//        return std::vector<double>();
//    }
//
//
//
//    // Allocate and copy device memory
//    double *f_z = new double(nr_sampling * nz_sampling);
////    double *dev_grid_data = createCUDAVector(grid_data);
////    double *dev_rho = createCUDAVector(rho);
//    double *dev_f_z = createCUDAVector(f_z, nr_sampling * nz_sampling);
////    double *dev_dv0 = createCUDAVector(dv0);
//
//
//
//// Get the block size for the tensor product kernel
//    // Get the block size for the tensor product kernel
//    int ntheta = costheta.size();
//    int threads_per_block = 32;
//    dim3 block_size(32, 32, 1);
//    dim3 grid_size((nr + block_size.x - 1) / block_size.x,
//                   (nz + block_size.y - 1) / block_size.y,
//                   (ntheta + block_size.z - 1) / block_size.z);
//
//
//    cudaDeviceProp prop;
//    int device_id=0;
//    cudaGetDeviceProperties(&prop, device_id);
//    int max_threads_per_block = prop.maxThreadsPerBlock;
//
//
//
//    // Launch kernel
//    get_all_g_kernel<<<grid_size, block_size>>>(nr, nz,ntheta, nr_sampling, nz_sampling, G,
//                                                 radial, dev_f_z, debug);
//    // Copy results back to host
//    std::vector<double> f_z_vec(nr_sampling * nz_sampling);
//    cudaMemcpy(f_z_vec.data(), dev_f_z, nr_sampling * nz_sampling * sizeof(double), cudaMemcpyDeviceToHost);
//
//
//    // Free device memory
//    cudaFree(dev_f_z);
//
//    // Return result
//    return f_z_vec;
//}
//