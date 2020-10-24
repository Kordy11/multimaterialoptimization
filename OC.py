from __future__ import division
import numpy as np


def OC(nelx, nely, nele, a, b, vf, p, alpha, df):
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
	'df' is a sensitivity of frequency
	----------
	Returns
	-------
	'alpha' is the updated density distribution
	-------
	"""
	df = -df		# to maximize eigen frequency
	# update lower and upper bounds of design variables
	move = 0.2
	eta = 0.15		# stabilization factor
	r = np.ones((1, nele), dtype=float)
	for k in range(p):
		if k != a and k != b:
			r = r - alpha[:, k]		# remaining volume fraction field
	l = np.maximum(0.001, alpha[:, a] - move)
	u = np.minimum(r, alpha[:, a] + move)
	# optimality criteria update of design variables
	l1 = 0			# lower limit of Lagrange multiplier
	l2 = 1e10		# upper limit of Lagrange multiplier
	while (l2-l1) > 1e-9:
		lmid = 0.5*(l2+l1)
		alpha_a = np.maximum(l, np.minimum(u, alpha[:, a]*(-df.flatten(order='F')/lmid)**eta))
		if (np.sum(alpha_a) - nelx*nely*vf[a]) > 0:
			l1 = lmid
		else:
			l2 = lmid
	alpha[:, a] = alpha_a
	alpha[:, b] = r - alpha_a
	return alpha
