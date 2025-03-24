#include <math.h>
#include <lapacke.h>
#include <cblas.h>

void serial_gmres(double* a, int n, double* b, int m, double* x);
