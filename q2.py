import numpy as np
import matplotlib.pyplot as plt
import functions as fn

n_list = [2**i for i in range(3, 8)]
rn = []

for n in n_list:
	
	A = np.zeros((n, n))
	for i in range(n):
		A[i, i] = -4
	for i in range(n-1):
		A[i+1, i  ] = 1
		A[i  , i+1] = 1
		
	b = np.array([1/n for _ in range(n)])

	_, r_hist = fn.gmres(A, b, n//2)	
	rn.append(r_hist[-1]);
	
plt.semilogx(n_list, np.array(rn)/np.linalg.norm(b), label=f"n = {n}")
plt.show() 
	

