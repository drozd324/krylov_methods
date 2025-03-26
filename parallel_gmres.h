#include <math.h>
#include <lapacke.h>
#include <cblas.h>
#include <string.h>
#include <stdlib.h>

void parallel_gmres(double* a, int n, double* b, int m, double* x);
