import numpy as np


def OC(nelx, nely, x, df, vf):
	"""
	OPTIMALITY CRITERIA METHOD
	Parameters
	----------
	'nelx' and 'nely' are the number of elements along the two axis
	'x' is a nely-by-nelx matrix representing the density field
	'df' is a sensitivity of frequency
	'dv' is a sensitivity of the volume constraint
	'vf' is a volume fraction
	----------
	Returns
	-------
	'xnew' is the updated density distribution
	-------
	"""
	df = -df			# to maximize eigen frequency
	l1 = 0				# lower limit to the volume Lagrange multiplier
	l2 = 10e6			# upper limit to the volume Lagrange multiplier
	move = 0.5			# the limit to the change of 'x' (stabilization)
	eta = 0.15			# damping numerical coefficient
	mu = np.max(df)		# shift to the Lagrange multiplier
	xnew = np.zeros((nelx*nely), dtype=float)
	# find of the volume multiplier using a bisection method
	while (l2-l1) > 1e-9:
		lmid = 0.5*(l2+l1)		# Lagrange multiplier
		xnew[:] = np.maximum(0.001, np.maximum(x-move, np.minimum(1.0,\
				np.minimum(x+move, x*((mu-df.flatten(order='F'))/lmid)**eta))))
		if (np.sum(xnew) - vf*nelx*nely) > 0:
			l1 = lmid
		else:
			l2 = lmid
	return xnew
