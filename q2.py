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
	
	k = np.arange(1, len(r)+1, 1)
	y = [np.linalg.norm(r_i) / np.linalg.norm(b) for r_i in r]
	
	plt.semilogy(k, y, label=f"n={n}")

#k = np.arange(1, (n_list[-1]//2) + 1, 1)
#plt.semilogy(k, 1/(k**10), 'o-')
plt.legend()
plt.xlabel("k (iteration)")
plt.ylabel(r"$||r_k||_2 \, / \, ||b||_2 $")
plt.savefig("writeup/q2_fig", dpi=300)
plt.show()
