//============================================================================
// Name        : GalaxyFormation.cpp
// Author      : Marco Pereira
// Version     : 1.0.0
// Copyright   : Your copyright notice
// Description : Hypergeometrical Universe Galaxy Formation in C++, Ansi-style
//============================================================================
#include <vector>
#include <array>
#include "../src/hugalaxy/tensor_utils.h"




int main() {

    std::vector<std::array<double, 2>> m33_rotational_curve = {
            {0.0f,       0.0f},
            {1508.7187f, 38.674137f},
            {2873.3889f, 55.65067f},
            {4116.755f,  67.91063f},
            {5451.099f,  79.22689f},
            {6846.0957f, 85.01734f},
            {8089.462f,  88.38242f},
            {9393.48f,   92.42116f},
            {10727.824f, 95.11208f},
            {11880.212f, 98.342697f},
            {13275.208f, 99.82048f},
            {14609.553f, 102.10709f},
            {18521.607f, 104.25024f},
            {22403.336f, 107.60643f},
            {26406.369f, 115.40966f},
            {30379.076f, 116.87875f},
            {34382.107f, 116.05664f},
            {38354.813f, 117.93005f},
            {42266.87f,  121.42091f},
            {46300.227f, 128.55017f},
            {50212.285f, 132.84966f}
    };
    auto interpolated_data = interpolate(m33_rotational_curve, 100);
    const int nr = 320;
    const int nz = 101;
    const int ntheta = 180;
    const double R_max = 50000.0;
    const double GalaxyMass = 5E10;
    const double M33_Distance = 3.2E6;
    // CURRENT M33
    double redshift = M33_Distance / (Radius_4D - M33_Distance);
    // TARGET BIRTH OF M33 Z=13
    double redshift_birth = 13.0;
//    double redshift_birth = redshift;
    std::vector<std::array<double,2>> new_m33_rotational_curve = move_rotation_curve(m33_rotational_curve, redshift, redshift_birth);
    std::vector<double> x0 = calculate_density_parameters(redshift_birth);
    double rho_0 = x0[0]; //z=0
    double alpha_0 = x0[1];
    double rho_1 = x0[2];
    double alpha_1 = x0[3];
    double h0 = x0[4];
    bool cuda= true;
    bool taskflow = true;
    int GPU_ID = 0;
    double xtol_rel = 1E-6;
    int max_iter = 5000;
// TARGET BIRTH OF M33 Z=10
    galaxy M33(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0, R_max, nr, nz, ntheta, redshift_birth, GPU_ID, cuda, taskflow, xtol_rel, max_iter );
    M33.read_galaxy_rotation_curve(new_m33_rotational_curve);
    std::vector<std::vector<double>> rho = M33.density(rho_0, alpha_0, rho_1, alpha_1, M33.r, M33.z);
    std::string compute_choice = getCudaString(M33.cuda, M33.taskflow_);
    std::cout << compute_choice << std::endl;
    auto velo_1 = M33.simulate_rotation_curve();
    double density_at_cmb= 1000;
    double radius_of_cmb = 11E6;
    std::basic_string<char> filename_base = "/home/mp74207/CLionProjects/HU_GalaxyPackage/notebooks/data/";
    double eta = 1E-4;
    double temperature = 1.0;
    double current_time = Radius_4D/(1 + redshift_birth);
    double final_time =current_time+4E9;
    unsigned long n_epochs =50;
    std::vector<double> epochs = logspace(current_time,final_time,n_epochs);
    double time_step = (final_time-current_time)/n_epochs;
    std::vector<double> redshifts(n_epochs+1);
    for(int i=0; i<n_epochs+1; i++) {
        redshifts[i] = Radius_4D/epochs[i] -1;
    }

    M33.DrudeGalaxyFormation(epochs, redshifts,eta, temperature,filename_base);

}




//int main() {
//    // Example usage
//    std::vector<int> arr1D = {1, 2, 3, 4, 5};
//    save_npy("arr1D.npy", arr1D);
//
//    std::vector<std::vector<int>> arr2D = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//    save_npy("arr2D.npy", arr2D);
//
//    std::vector<std::vector<std::vector<int>>> arr3D = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
//    save_npy("arr3D.npy", arr3D);
//
//    return 0;
//}
