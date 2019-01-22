import numpy as np

def problem1 (A, B):
	return A + B

def problem2 (A, B, C):
	return np.dot(A,B) - C

def problem3 (A, B, C):
	return A * B + np.transpose(C)

def problem4 (x, y):
	return np.inner(x,y)

def problem5 (A):
	return np.zeros(A.shape)

def problem6 (A):
	return np.ones(np.size(A,0))

def problem7 (A, x):
	return np.linalg.solve(A,x)

def problem8 (A, x):
	return np.transpose(np.linalg.solve(np.transpose(A),np.transpose(x)))

def problem9 (A, alpha):
	return A + alpha * np.eye(np.size(A,0)) 

def problem10 (A, i, j):
	return A[i][j]

def problem11 (A, i):
	return np.sum(A[i])

def problem12 (A, c, d):
	A = A[np.nonzero(A >= c)]
	A = A[np.nonzero(A <= d)]
	return np.mean(A)

def problem13 (A, k):
	w , v = np.linalg.eig(A)
	b = (-w).argsort()[:k]
	return v[:,b]

def problem14 (x, k, m, s):
	a = np.random.multivariate_normal(x + m *np.ones(x.shape), s* np.eye(x.shape[0]), (k,)).T
	return a