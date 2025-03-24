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


def gramschmidt(A):
	"""Calculates QR decomposition of a matrix A using the modified Gram Schmidt algorithm.

	Args:
		A (numpy array): numpy matrix

	Returns:
		tuple: the pair Q, R
	"""
	m, n = A.shape
	
	a = [A[:, i] for i in range(n)]
	
	Q = np.zeros((m, n))
	q = [Q[:, i] for i in range(n)]
	
	V = np.zeros((m, n))
	v = [V[:, i] for i in range(n)]
	
	R = np.zeros((n, n))
	r = [R[:, i] for i in range(n)]
	
	for i in range(n):
		v[i] = a[i]
	for i in range(n):
		r[i][i] = np.linalg.norm(v[i])
		q[i] = v[i] / r[i][i]
		for j in range(i, n):
			r[i][j] = q[i].T @ v[j]
			v[j] = v[j] - (r[i][j]*q[i])
	
	Q = np.reshape(np.array(q), (m,n))
	R = np.reshape(np.array(r), (n,n))
	
	return Q, R


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

	v = np.zeros((n, m+1))
	v[:, 0] = r_0 / beta
	
	h = np.zeros((m+1, m)) 
	w = np.zeros(n)

	rhs = beta * np.eye(m+1, 1) 
	y_m = 0
	
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
		
	Q, R = np.linalg.qr(h)
	y_m = back_sub(R, Q.T @ rhs)
	r = (np.linalg.norm(h @ y_m - rhs))
	
	x = x + v[:, :m] @ y_m 
	
	return x, r

def gmres(A, b, m):
	
	final_x = 0
	r_hist = []	
		
	for k in range(1, m+1):
		x, r = gmres_noiter(A, b, k)
		final_x = x
		r_hist.append(r)
	
	return final_x, r_hist
