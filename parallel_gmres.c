#include "parallel_gmres.h"

/**
 * @brief  
 *
 * @param a  Pointer to array of doubles for matrix of size n x n.
 * @param n  Dimension of matrix.
 * @param b  Pointer to array of doubles for vector b of lenght n.
 * @param m  Number of iterations.
 * @param x  Pointer to array of doubles for vector x for the solution to linear system Ax = b.
*/	
void parallel_gmres(double* a, int n, double* b, int m, double* x){
	
	// writes r = b - Ax
	double* r = calloc(n , sizeof(double));
	//memcpy(r, b, n);
	
	for (int i=0; i<n; i++){
		r[i] = b[i];
	}

	cblas_dgemv(CblasRowMajor, CblasNoTrans, 
		n, n, 
		-1, a, n, 
		x, 1, 
		1, r, 1);
	
	// writes beta = ||r||_2
	double beta = cblas_dnrm2(n, r, 1);
		
	double* v = calloc(n*(m+1) , sizeof(double));
	// v[:, 0] = r / beta	
	for (int i=0; i<n; i++){
		v[i*(m+1) + 0] = r[i] / beta;
	}			
	
	free(r);
		
	double* h = calloc((m+1)*m, sizeof(double));
	double* w = calloc(n, sizeof(double));
	double* v_j = calloc(n , sizeof(double));
	double* v_i = calloc(n , sizeof(double));

	//int m_ = m;
	for (int j=0; j<m; j++){ //cant paralellise this loop
		
		// writes w = A @ v[:, j]
		for (int k=0; k<n; k++){
			v_j[k] = v[k*(m+1) + j];
		} 
		cblas_dgemv(CblasRowMajor, CblasNoTrans,
			n, n, 1, a, n,  
			v_j, 1, 
			0, w, 1);			 
		
		for (int i=0; i<=j; i++){ // parallelise this?
			for (int k=0; k<n; k++){
				v_i[k] = v[k*(m+1) + i];
			} 
			h[i*m + j] = cblas_ddot(n, w, 1, v_i, 1); // h[i, j] = w.T @ v[:, i] 
			cblas_daxpy(n, -h[i*m + j], v_i, 1, w, 1); // w = w - h[i, j] * v[:, i]
		}
		
		h[(j+1)*m + j] = cblas_dnrm2(n, w, 1);
		if (fabs(h[(j+1)*m + j]) < 1e-14){
			m = j;
			break;	
		} 
	
		for (int i=0; i<n; i++){
			v[i*(m+1) + (j+1)] = w[i] / h[(j+1)*m + j];
		}
	}


	free(v_j);
	free(v_i);
	free(w);

	double* y_m = malloc((m+1) * sizeof(double));
	y_m[0] = beta;
	
	LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', m+1, m, 1, h, m, y_m, 1);
	
	free(h);
	
	// x = x + v[:, :m] @ y_m
	cblas_dgemv(CblasRowMajor, CblasNoTrans,
		n, m, 1, v, m+1,
		y_m, 1,
		1, x, 1);

	free(y_m);
	free(v);
}
