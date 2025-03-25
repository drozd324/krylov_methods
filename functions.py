import numpy as np
import scipy

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

def back_sub(R, b):
	"""solves Rx = b"""
	n = len(b)
	x = np.zeros(n)

	for i in range(n - 1, -1, -1): 
	
		x[i] = (b[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
	
	return x


def gmres_noiter(A, b, m): # no iter 
	
	n = A.shape[1]
	x = np.zeros(n).T	 
	r_0 = b - A @ x
	beta = np.linalg.norm(r_0)  

	print(f"beta = {beta}")

	v = np.zeros((n, m+1))
	v[:, 0] = r_0 / beta
	
	print("v = ")
	print(v[0])
	
	h = np.zeros((m+1, m)) 
	w = np.zeros(n)

	rhs = beta * np.eye(m+1, 1) 
	
	for j in range(m):
		w = A @ v[:, j]
		for i in range(j):
			h[i, j] = np.dot(w, v[:, i])
			w = w - h[i, j] * v[:, i]
		
		h[j+1, j] = np.linalg.norm(w)
		if h[j+1, j] == 0:
			m = j;
			break
		v[:, j+1] = w / h[j+1, j]
	
	print("h = ")
	print(h[1, :])
		
	#Q, R = np.linalg.qr(h)
	#y_m = back_sub(R, Q.T @ rhs)
	y_m = np.linalg.lstsq(h, rhs, rcond=None)[0]
	
	print("y_m = ")	
	print(y_m.T)

	r = np.linalg.norm(h @ y_m - rhs)	
	x = x + v[:, :m] @ y_m
	
	return x[:, -1], r

def gmres(A, b, m):
	
	final_x = 0
	r_hist = []	
		
	for k in range(1, m+1):
		x, r = gmres_noiter(A, b, k)
		final_x = x
		r_hist.append(r)
	
	return final_x, r_hist
