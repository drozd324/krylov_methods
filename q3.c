#include "serial_gmres.h"
#include "parallel_gmres.h"
#include <stdio.h>

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
		b[i] = (i+1)/(double)n; 
	}	

	double* serial_x  = calloc(n, sizeof(double));
		
	printf("serial_gmres\n");
	serial_gmres(A, n, b, m, serial_x);
	for (int i=0; i<n; i++)
		printf("%lf, ", serial_x[i]);
	printf("\n");

	free(A);
	free(b);
	free(serial_x);
		
	return 0;
}
