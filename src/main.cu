//============================================================================
// Name        : GalaxyFormation.cpp
// Author      : Marco Pereira
// Version     : 1.0.0
// Copyright   : Your copyright notice
// Description : Hypergeometrical Universe Galaxy Formation in C++, Ansi-style
//============================================================================
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include <vector>
#include <thread>
#include <memory>  // for std::unique_ptr
#include <cmath>
#include <stdexcept>
#include <cstring>
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include <array>
#include "/usr/include/boost/python.hpp"
#include <iostream>
#include <future>
#include "lib0.hpp"





class Galaxy {       // The class
public:             // Access specifier
    int nr;
    int nz;
    int nr_sampling;
    int nz_sampling;
    float G;
    float R_max;
    const float pi = 3.141592653589793;
    float dr;
    float alpha_0;
    float rho_0;
    float alpha_1;
    float rho_1;
    float h0;
    float dz;
    float dv0;
    float redshift;
    double GalaxyMass;
    float  sun_mass = 1.9885e30; // mass of the Sun in kilograms
    std::vector<float> r;
    std::vector<float> z;
    std::vector<float> r_sampling;
    std::vector<float> z_sampling;
    std::vector<float> rho;
    std::vector<float> theta;
    std::vector<float> costheta;
    std::vector<float> sintheta;
    std::vector<float> f_z;
    std::vector<float> rotational_velocity_;
    // ######################################
    std::vector<float> x_rotation_points;
    int n_rotation_points = 0;
    std::vector<float> v_rotation_points;
    // ######################################
    bool radial = true;


    Galaxy(float G, float dv0, double GalaxyMass, float rho_0, float alpha_0, float rho_1, float alpha_1, float h0,
           float R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta,
           float redshift = 0.0)
            : G(G * (1 + redshift)), R_max(R_max), nr(nr), nz(nz), nr_sampling(nr_sampling), nz_sampling(nz_sampling),
              alpha_0(alpha_0), rho_0(rho_0), alpha_1(alpha_1), rho_1(rho_1), h0(h0), dv0(dv0), redshift(redshift),
              GalaxyMass(GalaxyMass)
           {
        // ######################################
        r = linspace(1, R_max, nr);
        z = linspace(-h0 / 2.0, h0 / 2.0, nz);
        rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
        theta = linspace(0, 2 * pi, ntheta);
        costheta = costhetaFunc(theta);
        sintheta = sinthetaFunc(theta);
        // Allocate result vector
        z_sampling = linspace(-h0 / 2.0, h0 / 2.0, nz);
        r_sampling = linspace(1,R_max, nr_sampling);
    }

    std::vector<float> _rotational_velocity() {
        // Allocate result vector
        std::vector<float> z_sampling;
        z_sampling.push_back(0.0);
        std::vector<float> res = calculate_rotational_velocity(G, dv0,
                                                             r_sampling,
                                                             r,
                                                             z,
                                                             costheta,
                                                             sintheta,
                                                             rho,
                                                             redshift);
        return res;
    }


    void find_galaxy_initial_density(const std::vector<float>  vin, std::vector<float> x0){
        rotational_velocity_ = vin;
        float alpha_0 =x0[0];
        float rho_0 = x0[1];
        float alpha_1 = x0[2];
        float rho_1 = x0[3];


    }

    // Define the function to be minimized
    float error_function(const std::vector<float>& x) {
        // Calculate the rotation velocity using the current values of x
        float rho_0 = x[0];
        float alpha_0 = x[1];
        float rho_1 = x[2];
        float alpha_1 = x[3];
        float h0 = x[4];
        // Calculate the total mass of the galaxy
        float Mtotal_si;
        float error_mass;
        Mtotal_si = massCalc(alpha_0, rho_0,h0)/sun_mass;
        error_mass = pow( this->GalaxyMass-Mtotal_si,2);
        // Allocate result vector
        std::vector<float> z_sampling;
        z_sampling.push_back(0.0);
        std::vector<float> rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
        std::vector<float> vsim = calculate_rotational_velocity(G, dv0,
                                                               x_rotation_points,
                                                               r,
                                                               z,
                                                               costheta,
                                                               sintheta,
                                                               rho,
                                                               redshift); ;
        float error=0.0;
        for (int i=0; i< n_rotation_points; i++){ error += pow( (v_rotation_points[i]-vsim[i]), 2); }
        return error + error_mass;
    }


// Define the gradient function of the error function
    std::vector<float> gradient(const std::vector<float>& x) {
        const float h = 1e-5;
        const int n = x.size();
        std::vector<float> grad(n);
        float error_0 = error_function(x);
        std::vector<float> xh = x;
        for (int i = 0; i < n; i++) {
            xh[i] += h;
            float error_h = error_function(xh);
            grad[i] = (error_h - error_0) / h;
        }
        return grad;
    }

    void read_galaxy_rotation_curve(std::vector<std::array<float, 2>> vin){
        n_rotation_points = vin.size();
        for (const auto &row : vin) {
            x_rotation_points.push_back(row[0]); // Extract the first column (index 0)
            v_rotation_points.push_back(row[1]); // Extract the first column (index 0)
        }
    }

// Define the gradient descent algorithm
    std::vector<float> gradient_descent(const std::vector<float>& x0, float lr = 0.01, int max_iter = 1000, float eps = 1e-6) {
        std::vector<float> x = x0;
        for (int iter = 0; iter < max_iter; iter++) {
            // Compute the gradient of the error function
            std::vector<float> grad = gradient(x0);
            // Update the parameter values in the direction of steepest descent
            for (int i = 0; i < x.size(); i++) {
                x[i] -= lr * grad[i];
            }
            // Check for convergence
            if (std::abs(grad[0]) < eps && std::abs(grad[1]) < eps && std::abs(grad[2]) < eps && std::abs(grad[3]) < eps) {
                std::cout << "Converged after " << iter << " iterations." << std::endl;
                break;
            }
        }
        return x;
    }


    void set_radial(bool radial){
        radial = radial;
    }

    std::vector<float> get_g(bool radial){
        f_z = get_all_g(G, dv0, r_sampling, z_sampling, r, z,
                             costheta, sintheta, rho, redshift, radial);
        return f_z;
    }

};



int main(){
    std::vector<std::array<float, 2>> m33_rotational_curve = {
            {0.0f, 0.0f},
            {1508.7187f, 38.674137f},
            {2873.3889f, 55.65067f},
            {4116.755f, 67.91063f},
            {5451.099f, 79.22689f},
            {6846.0957f, 85.01734f},
            {8089.462f, 88.38242f},
            {9393.48f, 92.42116f},
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
            {42266.87f, 121.42091f},
            {46300.227f, 128.55017f},
            {50212.285f, 132.84966f}
    };

    const float M33_Distance = 3.2E6;
    const float Radius_Universe_4D = 14.03E9;
    float redshift = M33_Distance/(Radius_Universe_4D - M33_Distance);
    const int nr = 100;
    const int nz = 101;
    const int ntheta = 102;
    const int nr_sampling=103;
    const int nz_sampling=104;
    const float G = 7.456866768350099e-46;
    const float R_max = 50000.0;
    const float dr = R_max/nr;
    const float pi = 3.141592653589793;
    const float alpha_0 =0.00042423668409927005;
    const float rho_0 =  12.868348904393013;
    const float alpha_1 = 2.0523892233327836e-05;
    const float rho_1 = 0.13249804158174094;
    const float h0 = 156161.88949004377;
    const float dz = h0/nz;
    const float dtheta = 2*pi/ntheta;
    const float dv0 = dr * dtheta * dz;
    const float GalaxyMass = 5E10;
    const std::vector<float> r = linspace(1,R_max, nr);
    const std::vector<float> z = linspace(-h0/2.0 , h0/2.0, nz);
    const std::vector<float> rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
    const std::vector<float> theta = linspace(0,2*pi, ntheta);
    std::vector<float> f_z = zeros(nr*nz);
    Galaxy M33 = Galaxy(G, dv0, GalaxyMass, rho_0, alpha_0, rho_1, alpha_1,h0,
                        R_max, nr, nz, nr_sampling, nz_sampling, ntheta,
                        redshift);
    M33.read_galaxy_rotation_curve(m33_rotational_curve);
    std::vector<float> x0 = {alpha_0, rho_0, alpha_1, rho_1, h0};
    M33.gradient_descent(x0);
}
