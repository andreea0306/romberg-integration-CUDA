#pragma once
#include <cmath>

typedef double (*FunctionPtr)(double);

double f1(double x) { return 1/x; }
double f2(double x) { return x * x; }
double f3(double x) { return x * x * x; }
double f4(double x) { return pow(x, 4); }
double f5(double x) { return pow(x, 5); }
double f6(double x) { return x + 2 * x * x; }
double f7(double x) { return x - 3 * x * x + 2 * pow(x, 3); }
double f8(double x) { return 5 * x * x - 4 * x + 1; }
double f9(double x) { return x * x + 6 * x + 9; }
double f10(double x) { return x * x - 4 * x + 4; }
double f11(double x) { return 3 * x * x - 2 * x + 7; }
double f12(double x) { return pow(x, 3) - x * x + x - 1; }
double f13(double x) { return pow(x, 4) - 2 * pow(x, 3) + x * x - x + 3; }
double f14(double x) { return 2 * pow(x, 5) - 3 * pow(x, 3) + x - 5; }
double f15(double x) { return pow(x, 2) + 2 * x + 1; }
double f16(double x) { return 4 * pow(x, 4) - 2 * pow(x, 2) + 6 * x - 3; }
double f17(double x) { return x * x * x + 3 * x * x - x + 5; }
double f18(double x) { return 2 * pow(x, 6) - 4 * pow(x, 4) + x * x - 1; }
double f19(double x) { return pow(x, 7) - 3 * pow(x, 5) + 2 * pow(x, 3) - x + 4; }
double f20(double x) { return pow(x, 8) - 2 * pow(x, 6) + 3 * pow(x, 4) - 4 * x + 5; }
