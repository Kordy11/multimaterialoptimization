import numpy as np
from numpy.linalg import multi_dot


def getsensitivity(nelx, nely, alpha, eigenF, eigenM, qK, qM, edofMat, a, KE, ME):
	"""
	RETURN THE SENSITIVITY OF THE EIGENFREQUENCY WITH RESPECT TO DENSITIES
	Parameters
	----------
	'nelx' and 'nely' are the number of elements along the two dimensions
	'alpha' is a nely-by-nelx matrix representing the density field
	'eigenF' is the eigen value
	'eigenM' is the eigen vector
	'qK' and 'qM' are the penalization coefficients for stiffness and mass respectively
	'edofMat' is matrix of degrees of freedom for each element
	'a' is active phase of subscript
	'KE' is element stiffness matrix
	'ME' is element mass matrix
	----------
	Returns
	-------
	'df' is a sensitivity of frequency
	-------
	"""
	df = np.zeros((nely, nelx))
	omega = eigenF
	Mode = eigenM[:, 0]				# get the corresponding eigenmode
	for elx in range(nelx):
		for ely in range(nely):
			edofindex = edofMat[ely + elx*nely, :]
			Modee = Mode[edofindex][np.newaxis]			# corresponding eigenvector
			dKE = (qK*(alpha[ely + elx*nely, a]**(qK-1)))*KE
			dME = (qM*(alpha[ely + elx*nely, a]**(qM-1)))*ME
			df[ely, elx] = multi_dot([Modee, (dKE - omega*dME), Modee.T])
	df = df.flatten(order='F')
	return df
