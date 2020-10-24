import numpy as np


def OC(nelx, nely, nele, a, b, vf, p, alpha, dc):
	"""
	OPTIMALITY CRITERIA METHOD
	Parameters
	----------
	'nelx' and 'nely' are the number of elements along the two axis
	'nele' is a total number of elements
	'a' and 'b' are phases of subscript
	'vf' is a volume fraction vector
	'p' is a penalization factor
	'alpha' is a nely-by-nelx matrix representing the density field on the plate
	'dc' is a nely-by-nelx matrix returned by the sensitivity analysis
	----------
	Returns
	-------
	'alpha' is the updated density distribution
	-------
	"""
	# update lower and upper bounds of design variables
	move = 0.2
	r = np.ones((1, nele), dtype=float)
	for k in range(p):
		if k != a and k != b:
			r = r - alpha[:, k]			# remaining volume fraction field
	l = np.maximum(0, alpha[:, a] - move)
	u = np.minimum(r, alpha[:, a] + move)
	# optimality criteria update of design variables
	l1 = 0					# lower limit of Lagrange multiplier
	l2 = 1000000000.0		# upper limit of Lagrange multiplier
	while (l2-l1)/(l1+l2) > 0.001:
		lmid = 0.5*(l2+l1)
		alpha_a = np.maximum(l, np.minimum(u, alpha[:, a]*np.sqrt(-dc/lmid)))
		if np.sum(alpha_a) > nelx*nely*vf[a]:
			l1 = lmid
		else:
			l2 = lmid
	alpha[:, a] = alpha_a
	alpha[:, b] = r - alpha_a
	return alpha
