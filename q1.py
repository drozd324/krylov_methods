import numpy as np

def arnoldi(A, u, m):
	
	n, _ = A.shape
	
	Q = np.zeros((n, m+1))
	#q = [Q[:, i] for i in range(m+1)]
	
	H = np.zeros((m+1, m))
	#h = [H[:, i] for i in range(m)]

	q[0] = u/np.linalg.norm(u)

	for n in range(m):
		v = A @ q[n]
		for j in range(n):
			h[j][n] = q[j].T @ v
			v = v - (h[j][n] * q[j])
			
		h[n+1][n] = np.linalg.norm(v)
		q[n+1] = v / h[n+1][n]

	Q = np.reshape(np.array(q), (m+1, m+1))
	H = np.reshape(np.array(r), (m+1, m  ))

	return Q, H

A =[3, 8, 7, 3, 3, 7, 2, 3, 4, 8,
	5, 4, 1, 6, 9, 8, 3, 7, 1, 9,
	3, 6, 9, 4, 8, 6, 5, 6, 6, 6,
	5, 3, 4, 7, 4, 9, 2, 3, 5, 1,
	4, 4, 2, 1, 7, 4, 2, 2, 4, 5,
	4, 2, 8, 6, 6, 5, 2, 1, 1, 2,
	2, 8, 9, 5, 2, 9, 4, 7, 3, 3,
	9, 3, 2, 2, 7, 3, 4, 8, 7, 7,
	9, 1, 9, 3, 3, 1, 2, 7, 7, 1,
	9, 3, 2, 2, 6, 4, 4, 7, 3, 5]

A = np.reshape(np.array(A), (10, 10))

b =[+0.757516242460009,
	+2.734057963614329,
	-0.555605907443403,
	+1.144284746786790,
	+0.645280108318073,
	-0.085488474462339,
	-0.623679022063185,
	-0.465240896342741,
	+2.382909057772335,
	-0.120465395885881]

Q, H = arnoldi(A, b, 9)


print(Q)
print(H)
