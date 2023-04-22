#ifndef LIB0_HPP
#define LIB0_HPP
//============================================================================
// Name        : GalaxyFormation.cpp
// Author      : Marco Pereira
// Version     : 1.0.0
// Copyright   : Your copyright notice
// Description : Hypergeometrical Universe Galaxy Formation in C++, Ansi-style
//============================================================================
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdexcept>
#include <cstring>
#include <array>
#include <iostream>
#include <future>
#include <nlopt.hpp>
#include "Galaxy.h"

// Returns a vector of zeros with the given size
std::vector<double> zeros_1(int size) ;

std::vector<std::vector<double>> zeros_2(int nr, int nz) ;


void print(const std::vector<double> a) ;

void print_2D(const std::vector<std::vector<double>>& a);

double massCalcX(double alpha, double rho, double h, double x) ;

double massCalc(double alpha, double rho, double h) ;


std::vector<double> costhetaFunc(const std::vector<double> &theta);

std::vector<double> sinthetaFunc(const std::vector<double> &theta) ;

std::vector<double> linspace(double start, double end, size_t points) ;

std::vector<double> density(double rho_0, double alpha_0, double rho_1, double alpha_1, std::vector<double> r) ;

// # CPU functions
std::pair<double, double> get_g_cpu(double r_sampling_ii, double z_sampling_jj, double G,
                                    const std::vector<double> &dv0, const std::vector<double> &r,
                                    const std::vector<double> &z, const std::vector<double> &costheta,
                                    const std::vector<double> &sintheta, const std::vector<double> &rho, bool debug) ;

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_all_g_impl_cpu(double G, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
                   const std::vector<double> &z_sampling,
                   const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
                   const std::vector<double> &sintheta, const std::vector<double> &rho, bool debug);

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_all_g(double redshift, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
          const std::vector<double> &z_sampling,
          const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
          const std::vector<double> &sintheta, const std::vector<double> &rho, bool debug) ;


std::vector<double> calculate_rotational_velocity(double redshift, const std::vector<double> &dv0,
                                                  std::vector<double> r_sampling,
                                                  const std::vector<double> &r,
                                                  const std::vector<double> &z,
                                                  const std::vector<double> &costheta,
                                                  const std::vector<double> &sintheta,
                                                  const std::vector<double> &rho, bool debug) ;


std::vector<double> creategrid(double rho_0, double alpha_0, double rho_1, double alpha_1, int n) ;


std::vector<double> nelder_mead(const std::vector<double> &x0, Galaxy &myGalaxy, int max_iter = 1000, double xtol_rel = 1e-6);

#endif //LIB0_HPP