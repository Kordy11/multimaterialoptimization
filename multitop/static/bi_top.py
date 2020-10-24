from multitop.static.FEMM import *
from multitop.static.OC import *
import numpy as np


def bi_top(a, b, nelx, nely, p, q, vf, e, alpha_old, H, Hs, iter_max_in):
	"""
	BINARY PHASE TOPOLOGY OPTIMIZATION
	Parameters
	----------
	'a' and 'b' are phases (subscripts) of binary phase algorithm
	'nelx' and 'nely' are the number of elements along the two axis
	'p' and 'q' are penalization factors
	'vf' is a volume fraction vector
	'e' is an elastic modulus vector
	'alpha_old' is a nely-by-nelx matrix representing the density field from previous iteration
	'H' and 'Hs' are filter parameters
	'iter_max_in' is maximum number of inner iterations
	----------
	Returns
	-------
	'o' is the objective function
	'alpha' is a nely by nelx matrix representing the density field
	-------
	"""
	alpha = alpha_old.copy()
	# __ set loop counter
	iter_in = 0
	# __ inner iterations
	while iter_in < iter_max_in:
		iter_in = iter_in + 1
		# __ elasticity tensor
		E = e[0]*alpha[:, 0]**q
		for phase in range(1, p):
			E = E + e[phase]*alpha[:, phase]**q
		(U, edofMat, KE, nele) = FEM(nelx, nely, E)
		# __ objective function and sensitivity analysis
		ce = (np.dot(U[edofMat].reshape(nele, 8), KE)*U[edofMat].reshape(nele, 8)).sum(1)
		o = (E*ce).sum()
		dc = -(q*(e[a] - e[b])*alpha[:, a]**(q-1))*ce
		# __ filtering of sensitivities
		Hs = Hs.flatten()
		dc = np.asarray(H*(alpha[:, a]*dc)/Hs/np.maximum(0.001, alpha[:, a]))
		dc = np.minimum(dc, 0)		# for stabilization
		# __ optimality criteria
		alpha = OC(nelx, nely, nele, a, b, vf, p, alpha, dc)
	return o, alpha
