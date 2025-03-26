#include "serial_gmres.h"

void print_vec(double* v, int n){
	for (int i=0; i<n; i++){
		printf("%lf, ", v[i]);
	}
	printf("\n\n");
}

void print_mat(double* a, int m, int n, int ld){
	for (int i=0; i<m; i++){
		for (int j=0; j<n; j++){
			printf("%lf, ", a[i*ld  + j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

/**
 * @brief  
 *
 * @param a  Pointer to array of doubles for matrix of size n x n.
 * @param n  Dimension of matrix.
 * @param b  Pointer to array of doubles for vector b of lenght n.
 * @param m  Number of iterations.
 * @param x  Pointer to array of doubles for vector x for the solution to linear system Ax = b.
*/	
void serial_gmres(double* a, int n, double* b, int m, double* x){
	
	double* r_0 = calloc(n       , sizeof(double));
	double* v   = calloc(n*(m+1) , sizeof(double));
	double* h   = calloc((m+1)*m , sizeof(double));
	double* w   = calloc(n       , sizeof(double));

	// writes r_0 = b - Ax
	memcpy(r_0, b, n * sizeof(double));
	cblas_dgemv(CblasRowMajor, CblasNoTrans, 
		n, n, 
		-1, a, n, 
		x, 1, 
		1, r_0, 1);
	
	printf("r_0 = \n");
	print_vec(r_0, n);
	
	// writes beta = ||r||_2
	double beta = cblas_dnrm2(n, r_0, 1);
	
	printf("beta = %lf\n", beta);		

	// v[:, 0] = r / beta	
	for (int i=0; i<n; i++){
		v[i*(m+1) + 0] = r_0[i] / beta;
	}			
	free(r_0);
	
	double* y   = calloc((m+1)   , sizeof(double));
	y[0] = beta;
	
	for (int j=0; j<m; j++){ //cant paralellise this loop
		printf("v = \n");
		print_mat(v, n, m+1, m+1);
		// writes w = A @ v[:, j]
		cblas_dgemv(CblasRowMajor, CblasNoTrans,
			n, n, 1, a, n,  
			&(v[j]), m+1, 
			0, w, 1);			 
		
		printf("w = \n");
		print_vec(w, n);

		for (int i=0; i<j; i++){ // parallelise this?
			h[i*m + j] = cblas_ddot(n, w, 1, &(v[i]), m+1);  // h[i, j] = w.T @ v[:, i] 
			cblas_daxpy(n, -h[i*m + j], &(v[i]), m+1, w, 1); // w = w - h[i, j] * v[:, i]
		}
	
		printf("h = \n");
		print_mat(h, m+1, m, m);
		
		h[(j+1)*m + j] = cblas_dnrm2(n, w, 1);
		if (fabs(h[(j+1)*m + j]) < 1e-14){
			m = j;
			printf("AAAAAAAAAAAAAAAAAA");
			break;	
		} 
		printf("h[j+1, j] = %lf\n", h[(j+1)*m + j]);
	
		// parallelise this?
		for (int i=0; i<n; i++){
			v[i*(m+1) + (j+1)] = w[i] / h[(j+1)*m + j];
		}
		
		LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', j+2, j+1, 1, h, m, y, 1);

		printf("y = \n");
		print_vec(y, m+1);
		
		cblas_dgemv(CblasRowMajor, CblasNoTrans,
			n, j+1, 1, v, m+1,
			y, 1,
			0, x, 1);
		
		printf("x = \n");
		print_vec(x, n);
	}

	free(y);
	free(h);
	free(w);
	free(v);
}
