import numpy as np


def OC(nelx, nely, x, dc, dv, g):
	"""
	OPTIMALITY CRITERIA METHOD
	Parameters
	----------
	'nelx' and 'nely' are the number of elements along the two axis
	'x' is a nely-by-nelx matrix representing the density field
	'dc' is a nely-by-nelx matrix returned by the sensitivity analysis
	'dv' is a sensitivity of the volume constraint
	----------
	Returns
	-------
	'xnew' is the updated density distribution
	'gt' represents the OC condition
	-------
	"""
	l1 = 0			# lower limit to the volume Lagrange multiplier
	l2 = 10e6		# upper limit to the volume Lagrange multiplier
	move = 0.2		# the limit to the change of 'x' (stabilization)
	xnew = np.zeros(nelx*nely)
	# find of the volume multiplier using a bisection method
	while (l2-l1)/(l1+l2) > 1e-3:
		lmid = 0.5*(l2+l1)		# Lagrange multiplier
		xnew[:] = np.maximum(0.0, np.maximum(x-move, np.minimum(1.0,\
					np.minimum(x+move, x*np.sqrt(-dc/dv/lmid)))))
		gt = g + np.sum((dv*(xnew-x)))
		if gt > 0:
			l1 = lmid
		else:
			l2 = lmid
	return xnew, gt
