from multitop.modal.FEMM import *
from multitop.modal.OC import *
from multitop.modal.getsensitivity import *
import numpy as np


def bi_top(a, b, nelx, nely, p, qK, qM, v, e, rho, alpha_old, H, Hs, iter_max_in, optf):
	"""
	BINARY PHASE TOPOLOGY OPTIMIZATION
	Parameters
	----------
	'a' and 'b' are phases (subscripts) of binary phase algorithm
	'nelx' and 'nely' are the number of elements along the two axis
	'p' and 'q' are penalization factors
	'v' is a volume fraction matrix
	'e' is an elastic modulus matrix
	'alpha_old' is a nely-by-nelx matrix representing the density field from previous iteration
	'H' and 'Hs' are filter parameter
	'iter_max_in' is maximum number of inner iterations
	----------
	Returns
	-------
	'o' is the objective function
	'alpha' is a nely by nelx matrix representing the density field
	-------
	"""
	alpha = alpha_old.copy()
	# __ filtering allocation
	dv = np.ones(nelx * nely, dtype=float)		# sensitivity of the volume constraint
	# __ set loop counter
	iter_in = 0
	# __ inner iterations
	while iter_in < iter_max_in:
		iter_in = iter_in + 1
		# __ elasticity and mass tensor
		EK = e[0]*alpha[:, 0]**qK
		EM = e[0]*alpha[:, 0]**qM
		for phase in range(1, p):
			EK = EK + e[phase]*alpha[:, phase]**qK
			EM = EM + e[phase]*alpha[:, phase]**qM
		(eigenF, eigenM, KE, ME, edofMat, nele) = FEM(nelx, nely, rho, EK, EM, a, optf)
		# ----------SENSITIVITY ANALYSIS----------
		df = getsensitivity(nelx, nely, alpha, eigenF, eigenM, qK, qM, edofMat, a, KE, ME)
		dv[:] = np.ones(nele)  # sensitivity of the volume constraint
		# ----------APPLICATION OF SENSITIVITY OR DENSITY FILTER----------
		df = np.asarray((H*(alpha[:, a].flatten(order='F') * df))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, alpha[:, a])
		# ----------OPTIMALITY CRITERIA----------
		# __ optimality criteria
		alpha = OC(nelx, nely, nele, a, b, v, p, alpha, df)
	return alpha, eigenF, iter_in
