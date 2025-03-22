#include <stdio.h>

/* @brief An implementation of the Arnoldi ieration algorithm for krylov
 * 		  subspaces that performs the Arnoldi iteration for a regular
 * 		  matrix A with vector u to build the Krylov suspace of degree m.
 *
 * @param A Regular matrix A
 * @param u 
 * @param m
 * @param Q
 * @param H
*/
void arnoldi(double* A, double* u, int m, double* Q, double* H){

	//Q[i*m + j];
	
	for (int i=0; i<m; i++){
		Q[i*m + 0] = u/


}



int main(){

	



	return 0;
}
