import numpy as np
import matplotlib.pyplot as plt
import functions as fn

n_list = [2**i for i in range(3, 9)]

for n in reversed(n_list):
	
	m = n//2

	A = np.zeros((n, n))
	for i in range(n):
		A[i, i] = -4
	for i in range(n-1):
		A[i+1, i  ] = 1
		A[i  , i+1] = 1
		
	b = np.array([(i+1)/n for i in range(n)])

	_, r = fn.gmres(A, b, m)
	r_iter = len(r)
	
	k = np.arange(1, r_iter+1, 1)
	y = [np.linalg.norm(r[i]) / np.linalg.norm(b) for i in range(r_iter)]
	#y = [np.linalg.norm(r[i]) for i in range(r_iter)]
	
	plt.semilogy(k, y, label=f"n={n}")
	#plt.loglog(x, y, "o-", label=f"n={n}")

plt.legend()
plt.xlabel("k (iteration)")
plt.ylabel(r"$||r_k||_2 \, / \, ||b||_2 $")
#plt.ylim(1e-1, 1e+1)
plt.savefig("writeup/q2_fig", dpi=300)
plt.show() 

