#include <stdio.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>
#include <mpi.h>
#include "gmres.h"

int gmres(double* a, int n, double* b, int m){

	int proc;
	int num_procs;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &proc);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);	
	
	double* x = malloc(n * sizeof(double));
	double* r = malloc(n * sizeof(double));

	// writes r = b - Ax
	cblas_dgemv('N',
		n, n, 1, a, n,
		x, 1, 1,
		r, 1);

	// writes beta = ||r||_2
	double beta = cblas_dnrm2(n, r, 1);
	
	// v[:, 0] = r / beta	
	double* v = malloc(n * (m+1) * sizeof(double));
	for (int i=0; i<n; i++){
		v[n*i + 0] = r[i] / beta
	}			
	
	double* h = calloc((m+1)*m, sizof(double));
	double* w = malloc(n * sizof(double));
	
		
	for (int j=0; j<m; j++){

		// writes w = A @ v[:, j]
		double* v_j = malloc(n * sizeof(double));
		cblas_dgemv('N',
			n, n, 1, a, n,
			v, 1, 1,
			w, 1);
		
		for (int i=0; i<j; i++){
			// h[i, j] = w.T @ v[:, i]
			h[i*(m+1) + j] = cblas_ddot(n, w, 1, v_j, 1);
			// w = w - h[i, j] * v[:, i]
			cblas_daxpy(n, -h[i*(m+1) + j], v_j, 1, w, 1);
		}
		
		h[(j+1)*(m+1) + j] = cblas_dnrm2(n, w, 1);
		if (h[(j+1)*(m+1) + j] == 0){
			break;	
		} 
	
		for (int i=0; i<n; i++){
			v[n*i + (j+1)] = w[i] / h[(j+1)*(m+1) + j];
		}
	}
	
	double* be_1 = ralloc(n, sizeof(double));
	be_1[0] = beta;

	int lwork = 3 * (m+1) * m;  // Conservative estimation
    double *work = (double*)malloc(lwork * sizeof(double));
	y_m = LAPACKE_dgels('N', m+1, m, 1, h, m+1, be_1, 1, work, lwork)
		
	
}
