#include "serial_gmres.h"


void back_sub(double* R, double* b, int n, double* x){		
	for (int i=n-1; i>=0; i--){
		x[i] = (b[i] - cblas_ddot(n, &(R[i*n]), 1, b, 1)) / R[i*n + i];
	}
}



/**
 * @brief A function which combines two of LAPACKE's qr factorisation
 *        function to obain
 *
 * @param m Number of rows of matrix to decompose
 * @param n Number of columns of matrix to decompose
 * @param a Pointer to array of doubles representing matrix to decompose
 * @param q Pointer to array of doubles representing matrix to put Q
 *          from QR = A into.
 * @param r Pointer to array of doubles representing matrix to put R
 *          from QR = A into.
*/
void qr_decomp(int m, int n, double* a, double* q, double* r){
    int lda = n;
    double* tau = malloc(fmin(m, n) * sizeof(double));

    memcpy(q, a, n*m * sizeof(double));

    // Double precision GEneral QR Factorisation
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, q, lda, tau);

    // write to R to r
    for (int i=0; i<n; i++) {
        for (int j=i; j<n; j++){
            r[i*n + j] = q[i*n + j];
        }
    }

    int k = fmin(m, n);
    // Double precision ORthogonal Generate? QR Factorisation
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, k, q, lda, tau);
    free(tau);
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
	
	double* r = malloc(n * sizeof(double));
	// r = b
	for (int i=0; i<n; i++){
		r[i] = b[i];
	}
		
	// writes r = b - Ax
	cblas_dgemv(CblasRowMajor, CblasNoTrans,
		n, n, 
		-1, a, n,
		x, 1, 
		1, r, 1);
	
	// writes beta = ||r||_2
	double beta = cblas_dnrm2(n, r, 1);
	
	// v[:, 0] = r / beta	
	double* v = malloc(n*(m+1) * sizeof(double));
	for (int i=0; i<n; i++){
		v[i*(m+1)] = r[i] / beta;
	}			

	free(r);
			
	double* h = calloc((m+1)*m, sizeof(double));
	double* w = calloc(n, sizeof(double));
		
	double* v_j = malloc(n * sizeof(double));
	double* v_i = malloc(n * sizeof(double));
	
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
		
		
		for (int i=0; i<j; i++){ // parallelise this?
			for (int k=0; k<n; k++){
				v_i[k] = v[k*(m+1) + i];
			} 
			h[i*m + j] = cblas_ddot(n, w, 1, v_i, 1); // h[i, j] = w.T @ v[:, i] 
			cblas_daxpy(n, -h[i*m + j], v_i, 1, w, 1); // w = w - h[i, j] * v[:, i]
		}
		
		h[(j+1)*m + j] = cblas_dnrm2(n, w, 1);
		if (h[(j+1)*m + j] == 0){
			//m_ = j;
			m = j;
			break;	
		} 
	
		// parallelise this?
		for (int i=0; i<n; i++){
			v[i*(m+1) + (j+1)] = w[i] / h[(j+1)*m + j];
		}
	}
	
	//m = m_;	
	
	free(v_j);
	free(v_i);
	free(w);

	double* be_1 = calloc(m+1, sizeof(double));
	be_1[0] = beta;
	
	
	double* Q = malloc((m+1)*m * sizeof(double));
	double* R = malloc(m*m * sizeof(double));
	double* rhs = malloc((m+1) * sizeof(double));
	qr_decomp(m+1, m, h, Q, R);
	cblas_dgemv(CblasRowMajor, CblasTrans,
		m+1, m, 1, Q, m,
		be_1, 1,
		0, rhs, 1);
	back_sub(R, rhs, n, be_1); 
	// now y_m = rhs	

	free(Q);
	free(R);
	

	//LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', m+1, m, 1, h, m, be_1, 1);
	// now y_m = be_1	

	free(h);	

	// sol = x + v[:, :m] @ y_m	
	double* V = malloc(n * m * sizeof(double));
	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			V[i*m + j] = v[i*m + j];
		}
	}
	cblas_dgemv(CblasRowMajor, CblasNoTrans,
		n, m, 1, V, m,
		rhs, 1,
		1, x, 1);


	free(be_1);
	free(V);
	free(v);
}
