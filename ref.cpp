#include <iostream>
#include <cmath>
#include "utils.h"
#include <omp.h>
#include <vector>
#include <cstdint>
#include <chrono>
using namespace std;

FunctionPtr functions[] = {f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20};

double romberg_integration(double lower_limit, double upper_limit, uint32_t p, uint32_t q, FunctionPtr f) {
    std::vector<std::vector<double>> t(p + 1, std::vector<double>(q + 1, 0));
    std::vector<double> local(p + 1, 0);
    double h,sm,sl,a;

    h = upper_limit - lower_limit;
    t[0][0] = h / 2 * ((f(lower_limit)) + (f(upper_limit)));
#pragma omp parallel for
    for (std::size_t i = 1; i <= p; i++) {
        sl = pow(2, i - 1);
        sm = 0;
        for (std::size_t j = 1; j <= (std::size_t)sl; j++) {
            a = lower_limit + (2 * (double)j - 1) * h / pow(2, i);
            sm = sm + (f(a));
        }
        t[i][0] = t[i - 1][0] / 2 + sm * h / pow(2, i);
    }

    for(std::size_t i=1; i<=p; ++i){
        for(std::size_t j=1;j<=i && j<=q; ++j){
            t[i][j] = (pow(4, j) * t[i][j - 1] - t[i - 1][j - 1]) / (pow(4, j) - 1);
        }
    }
//        printf("Romberg estimate of integration =%e\n",t[p][q]);
    return t[p][q];
}

int main(int argc, char **argv) {
    int num_functions = sizeof(functions) / sizeof(functions[0]);
    double lower_limit = -10, upper_limit = 11;
    uint32_t p = 27, q = 27;
    double results[num_functions];

    // Process command-line arguments
    if (argc == 5) {
        lower_limit = atof(argv[1]);
        upper_limit = atof(argv[2]);
        p = atoi(argv[3]);
        q = atoi(argv[4]);
    } else {
        std::cout << "Usage: " << argv[0] << " <lower_limit> <upper_limit> <p> <q>" << std::endl;
        std::cout << "Using default values: lower_limit = -1, upper_limit = 4, p = 24, q = 24" << std::endl;
    }

    std::cout << "Number of functions: " << num_functions << "\n";
    std::cout << "Interval: [" << lower_limit << ":" << upper_limit << "]\n";
    std::cout << "p = " << p << "; q = " << q << "\n";

    printf("Starting Romberg Integration\n");
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (std::size_t i = 0; i < num_functions; ++i) {
//        std::cout << "Integrating f" << i << "\n";
        results[i] = romberg_integration(lower_limit, upper_limit, p, q, functions[i]);
    }

    printf("===============RESULTS==================\n");
    for (std::size_t i = 0; i < num_functions; ++i) {
        printf("Estimate of integration for f%ld =%e\n", i, results[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "\nExecution time: " << duration << " seconds\n";
    return 0;
}
