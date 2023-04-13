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
    double G;
    double R_max;
    const double pi = 3.141592653589793;
    double dr;
    double alpha_0;
    double rho_0;
    double alpha_1;
    double rho_1;
    double h0;
    double dz;
    double dv0;
    double redshift;
    double GalaxyMass;
    double  sun_mass = 1.9885e30; // mass of the Sun in kilograms
    std::vector<double> r;
    std::vector<double> z;
    std::vector<double> r_sampling;
    std::vector<double> z_sampling;
    std::vector<double> rho;
    std::vector<double> theta;
    std::vector<double> costheta;
    std::vector<double> sintheta;
    std::vector<double> f_z;
    std::vector<double> rotational_velocity_;
    // ######################################
    std::vector<double> x_rotation_points;
    int n_rotation_points = 0;
    std::vector<double> v_rotation_points;
    // ######################################
    bool radial = true;


    Galaxy(double G, double dv0, double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
           double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta,
           double redshift = 0.0)
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

    std::vector<double> _rotational_velocity(double rho_0, double alpha_0, double rho_1, double alpha_1,  double h0) {
        // Allocate result vector
        std::vector<double> z_sampling;
        z_sampling.push_back(0.0);
        std::vector<double> res = calculate_rotational_velocity(this->G, this->dv0,
                                                             this->r_sampling,
                                                             this->r,
                                                             this->z,
                                                             this->costheta,
                                                             this->sintheta,
                                                             density(rho_0, alpha_0, rho_1, alpha_1, this->r_sampling));
        return res;
    }


//    void find_galaxy_initial_density(const std::vector<double>  vin, std::vector<double> x0){
//        rotational_velocity_ = vin;
//        double alpha_0 =x0[0];
//        double rho_0 = x0[1];
//        double alpha_1 = x0[2];
//        double rho_1 = x0[3];
//
//
//    }

    // Define the function to be minimized
    double error_function(const std::vector<double>& x) {
        // Calculate the rotation velocity using the current values of x
        double rho_0 = x[0];
        double alpha_0 = x[1];
        double rho_1 = x[2];
        double alpha_1 = x[3];
        double h0 = x[4];
        // Calculate the total mass of the galaxy
        double Mtotal_si=0.0;
        double error_mass=0.0;
        Mtotal_si = massCalc(alpha_0, rho_0,h0);  // Mtotal in Solar Masses
        error_mass = pow( (this->GalaxyMass-Mtotal_si)/this->GalaxyMass,2);
        std::vector<double> rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
        std::vector<double> vsim = calculate_rotational_velocity(this->G, this->dv0,
                                                               this->x_rotation_points,
                                                               this->r,
                                                               this->z,
                                                               this->costheta,
                                                               this->sintheta,
                                                               rho) ;
        double error=0.0;
        for (int i=0; i< n_rotation_points; i++){ error += pow( (v_rotation_points[i]-vsim[i]), 2); }
        return error + error_mass;
    }


// Define the gradient function of the error function
    std::vector<double> gradient(const std::vector<double>& x) {
        const double h = 1e-5;
        const int n = x.size();
        std::vector<double> grad(n);
        double error_0 = error_function(x);
        std::vector<double> xh = x;
        for (int i = 0; i < n; i++) {
            xh[i] += h;
            double error_h = error_function(xh);
            grad[i] = (error_h - error_0) / h;
        }
        return grad;
    }

    void read_galaxy_rotation_curve(std::vector<std::array<double, 2>> vin){
        n_rotation_points = vin.size();
        this->x_rotation_points.clear();
        this->v_rotation_points.clear();
        for (const auto &row : vin) {
            this->x_rotation_points.push_back(row[0]); // Extract the first column (index 0)
            this->v_rotation_points.push_back(row[1]); // Extract the first column (index 0)
        }
    }

// Define the gradient descent algorithm
    std::vector<double> gradient_descent(const std::vector<double>& x0, double lr = 0.01, int max_iter = 1000, double eps = 1e-6) {
        std::vector<double> x = x0;
        for (int iter = 0; iter < max_iter; iter++) {
            // Compute the gradient of the error function
            std::vector<double> grad = gradient(x0);
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

    std::vector<double> get_g(bool radial){
        f_z = get_all_g(G, dv0, r_sampling, z_sampling, r, z,
                             costheta, sintheta, rho, redshift, radial);
        return f_z;
    }

};



int main(){
    std::vector<std::array<double, 2>> m33_rotational_curve = {
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

    const double M33_Distance = 3.2E6;
    const double Radius_Universe_4D = 14.03E9;
    double redshift = M33_Distance/(Radius_Universe_4D - M33_Distance);
    const int nr = 100;
    const int nz = 101;
    const int ntheta = 102;
    const int nr_sampling=103;
    const int nz_sampling=104;
    const double G = 6.6743e-11;
    const double R_max = 50000.0;
    const double dr = R_max/nr;
    const double pi = 3.141592653589793;
    const double alpha_0 =0.00042423668409927005;
    const double rho_0 =  12.868348904393013;
    const double alpha_1 = 2.0523892233327836e-05;
    const double rho_1 = 0.13249804158174094;
    const double h0 = 156161.88949004377;
    const double dz = h0/nz;
    const double dtheta = 2*pi/ntheta;
    const double dv0 = dr * dtheta * dz;
    const double GalaxyMass = 5E10;
    const std::vector<double> r = linspace(1,R_max, nr);
    const std::vector<double> z = linspace(-h0/2.0 , h0/2.0, nz);
    const std::vector<double> rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
    const std::vector<double> theta = linspace(0,2*pi, ntheta);
    std::vector<double> f_z = zeros(nr*nz);
    Galaxy M33 = Galaxy(G, dv0, GalaxyMass, rho_0, alpha_0, rho_1, alpha_1,h0,
                        R_max, nr, nz, nr_sampling, nz_sampling, ntheta,
                        redshift);
    M33.read_galaxy_rotation_curve(m33_rotational_curve);
    std::vector<double> x0 = { rho_0, alpha_0, rho_1, alpha_1,  h0};
    M33.gradient_descent(x0);
}
