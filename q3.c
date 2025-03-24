#include "serial_gmres.h"
#include <stdio.h>
//#include "parallel_gmres.h"

int main(){
	
	int m = 10; //number of itarations
	int n = 10;
	double* A = calloc(n*n , sizeof(double));
	double* b = malloc(n * sizeof(double));
		
	for (int i=0; i<n; i++){
		A[i*n + i] = -4; 	
	}
	for (int i=0; i<n-1; i++){
		A[(i+1)*n + i] = 1; 	
		A[(i)*n + i+1] = 1; 	
	}
	
	for (int i=0; i<n; i++){
		b[i] = (i+1)/n; 
	}	

	double* x = calloc(n, sizeof(double));
		
	serial_gmres(A, n, b, m, x);
			

	for (int i=0; i<n; i++)
		printf("%lf, ", x[i]);
	printf("\n");

	free(A);
	free(b);
	free(x);
		
	return 0;
}
