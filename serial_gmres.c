#include "serial_gmres.h"

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
	
	double* r = malloc(n * sizeof(double));
	for (int i=0; i<n; i++){
		r[i] = b[i];
	}
		
	// writes r = b - Ax
	cblas_dgemv(CblasRowMajor, CblasNoTrans,
		n, n, 1, a, n,
		x, 1, 1,
		r, 1);
	
	// writes beta = ||r||_2
	double beta = cblas_dnrm2(n, r, 1);
	
	// v[:, 0] = r / beta	
	double* v = malloc(n*(m+1) * sizeof(double));
	for (int i=0; i<n; i++){
		v[(m+1)*i + 0] = r[i] / beta;
	}			

	free(r);
			
	double* h = calloc((m+1)*m, sizeof(double));
	double* w = malloc(n * sizeof(double));
	
		
	double* v_j = malloc(n * sizeof(double));
	double* v_i = malloc(n * sizeof(double));
	for (int j=0; j<m; j++){ //cant paralellise this loop

		// writes w = A @ v[:, j]
		cblas_dgemv(CblasRowMajor, CblasNoTrans,
			n, n, 1, a, n,  
			v_j, 1, 1,		
			w, 1);			 
		
		for (int i=0; i<j; i++){
			for (int k=0; k<n; k++){
				v_i[k] = v[k*n + i];
			} 

			h[i*m + j] = cblas_ddot(n, w, 1, v_i, 1); // h[i, j] = w.T @ v[:, i] 
			cblas_daxpy(n, -h[i*m + j], v_i, 1, w, 1); // w = w - h[i, j] * v[:, i]
		}
		
		h[(j+1)*m + j] = cblas_dnrm2(n, w, 1);
		if (h[(j+1)*m + j] == 0){
			break;	
		} 
	
		// parallelise this?
		for (int i=0; i<n; i++){
			v[n*i + (j+1)] = w[i] / h[(j+1)*m + j];
		}
	}

	free(v_j);
	free(w);
	
	double* be_1 = calloc(m+1, sizeof(double));
	be_1[0] = beta;

	LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', m+1, m, 1, h, m, be_1, 1);
	// now y = be_1	

	//free(work);
	free(h);	

	// sol = x + v[:, :m] @ y_m	
	cblas_dgemv(CblasRowMajor, CblasNoTrans,
		n, m, 1, v, m+1,
		be_1, 1, 1,
		x, 1);

	free(be_1);
	free(v);
}
