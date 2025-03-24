import functions as fn
import numpy as np

m = 100
n = 10
A = np.zeros((n, n))
for i in range(n):
	A[i, i] = -4
for i in range(n-1):
	A[i+1, i  ] = 1
	A[i  , i+1] = 1

b = np.array([(1+i)/n for i in range(n)])

print("A =")
print(A)
print("")

print("b =")
print(b)
print("")

print("x =")
print(fn.gmres(A, b, m)[0])


print("SCIPY")
import scipy
print("x =")
print(scipy.sparse.linalg.gmres(A, b)[0])
