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

void simulateFreeFall(double redshift_birth){
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
    const int nr = 320;
    const int nz = 101;
    const int ntheta = 180;
    const double R_max = 50000.0;
    const double GalaxyMass = 5E10;
    const double M33_Distance = 3.2E6;
    // CURRENT M33
    double redshift = M33_Distance / (Radius_4D - M33_Distance);
    // TARGET BIRTH OF M33 Z=10
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
    std::vector<std::array<double, 2>> new_m33_rotational_curve = move_rotation_curve(m33_rotational_curve, redshift, redshift_birth);
    galaxy M33(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0, R_max, nr, nz, ntheta, redshift_birth, GPU_ID, cuda, taskflow, xtol_rel, max_iter );
    M33.read_galaxy_rotation_curve(new_m33_rotational_curve);
    std::basic_string<char> filename_base = "./notebooks/data/";
    double current_time = Radius_4D/(1 + redshift_birth);
    double final_time =current_time + 3E6*10/redshift_birth;
    if (final_time>Radius_4D){final_time=Radius_4D;}
    unsigned long n_epochs =4;
    std::vector<double> epochs = geomspace(current_time, final_time, n_epochs);
    M33.FreeFallGalaxyFormation(epochs,filename_base);
}

void simulateDrude(double redshift_birth){
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
    const int nr = 320;
    const int nz = 101;
    const int ntheta = 180;
    const double R_max = 50000.0;
    const double GalaxyMass = 5E10;
    const double M33_Distance = 3.2E6;
    // CURRENT M33
    double redshift = M33_Distance / (Radius_4D - M33_Distance);
    // TARGET BIRTH OF M33 Z=10
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
    std::vector<std::array<double, 2>> new_m33_rotational_curve = move_rotation_curve(m33_rotational_curve, redshift, redshift_birth);
    galaxy M33(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0, R_max, nr, nz, ntheta, redshift_birth, GPU_ID, cuda, taskflow, xtol_rel, max_iter );
    M33.read_galaxy_rotation_curve(new_m33_rotational_curve);
    std::basic_string<char> filename_base = "./notebooks/data/";
    double eta = 1E-2;
    double temperature = 1.0;
    double current_time = Radius_4D/(1 + redshift_birth);
    double final_time =current_time + 3E9*10/redshift_birth;
    if (final_time>Radius_4D){final_time=Radius_4D;}
    unsigned long n_epochs =20;
    std::vector<double> epochs = geomspace(current_time, final_time, n_epochs);
    M33.DrudeGalaxyFormation(epochs,eta, temperature,filename_base);
}

int main() {
    for (int i = 5; i > 3; i--){
        simulateFreeFall(((double)i));
    };
//    for (int i = 10; i > 3; i--){
//        simulateDrude((double) i);
//    };
}
