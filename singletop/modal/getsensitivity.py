import numpy as np
from numpy.linalg import multi_dot


def getsensitivity(nelx, nely, x, eigenF, eigenM, pK, pM, KE, ME, edofMat):
	"""
	RETURN THE SENSITIVITY OF THE EIGENFREQUENCY WITH RESPECT TO DENSITIES
	Parameters
	----------
	'nelx' and 'nely' are the number of elements along the two dimensions
	'x' is a nely-by-nelx matrix representing the density field
	'eigenF' is the eigen value
	'eigenM' is the eigen vector
	'pK' and 'pM' are the penalization coefficients for stiffness and mass respectively
	'KE' is element stiffness matrix
	'ME' is element mass matrix
	'edofMat' is matrix of degrees of freedom for each element
	----------
	Returns
	-------
	'df' is a sensitivity of frequency
	-------
	"""
	df = np.zeros((nely, nelx))
	omega = eigenF
	Mode = eigenM[:, 0]				# get the corresponding eigenmode
	x = np.reshape(x, (nely, nelx), order='F')
	for elx in range(nelx):
		for ely in range(nely):
			edofindex = edofMat[ely + elx*nely, :]
			Modee = Mode[edofindex][np.newaxis]		# corresponding eigenvector
			dKE = (pK*(x[ely, elx]**(pK-1)))*KE
			dME = (pM*(x[ely, elx]**(pM-1)))*ME
			df[ely, elx] = multi_dot([Modee, (dKE - omega*dME), Modee.T])
	df = df.flatten(order='F')
	return df
