import numpy as np

def arnoldi(A, u, m):
	rows, cols = A.shape
	
	Q = np.zeros((rows, m+1))
	H = np.zeros((m+1, m))

	Q[:, 0] = u/np.linalg.norm(u)

	for n in range(m):
		v = A @ Q[:, n]
		for j in range(n+1):
			H[j, n] = Q[:, j].T @ v
			v = v - (H[j, n] * Q[:, j])
			
		H[n+1, n] = np.linalg.norm(v)
		Q[:, n+1] = v / H[n+1, n]

	return Q, H

def gmres(A, b, m):
	
	n = A.shape[1]
	x = np.zeros(n)
	r_0 = b - A @ x
	beta = np.linalg.norm(r_0)  

	v = np.zeros((n, m+1))
	v[:, 0] = r_0 / beta
	
	h = np.zeros((m+1, m)) 

	rhs = np.zeros(m+1)
	rhs[0] = beta

	r = [r_0]
		
	for j in range(m): # w
		w = A @ v[:, j]
		for i in range(j+1):
			h[i, j] = np.dot(w, v[:, i])
			w = w - h[i, j] * v[:, i]
		
		h[j+1, j] = np.linalg.norm(w)
		if abs(h[j+1, j]) < 1e-14:
			m = j
			break

		v[:, j+1] = w / h[j+1, j]
		
		y = np.linalg.lstsq(h[:j+2, :j+1], rhs[:j+2], rcond=None)[0]

		x = v[:, :j+1] @ y
		r.append(b - A @ x)	

	#y = np.linalg.lstsq(h, rhs, rcond=None)[0]
	#print(f"shape = {y.shape}")
	#print(f"v = {v.shape}")
	#x = v[:, :m] @ y

	return x, r
