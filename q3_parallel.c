#include "serial_gmres.h"
#include "parallel_gmres.h"
#include <stdio.h>

int main(){
	
	int m = 1000; //number of itarations
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
		b[i] = (i+1)/(double)n; 
	}	

	double* parallel_x = calloc(n, sizeof(double));

	printf("parallel_gmres\n");
	parallel_gmres(A, n, b, m, parallel_x);
	for (int i=0; i<n; i++)
		printf("%lf, ", parallel_x[i]);
	printf("\n");

	free(A);
	free(b);
	free(parallel_x);
		
	return 0;
}
