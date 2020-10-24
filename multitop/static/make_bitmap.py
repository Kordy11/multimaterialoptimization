import numpy as np
import cv2


def make_bitmap(p, nelx, nely, alpha):
	"""
	MAKE BITMAP IMAGE OF MULTIPHASE TOPOLOGY
	Parameters
	----------
	'p' is a penalization factor used for SIMP model
	'nelx' and 'nely' are the number of elements along the two axis
	'alpha' is a nely-by-nelx matrix representing the density field
	----------
	Returns
	-------
	'I' bitmap image values
	-------
	"""
	color = np.array([[0, 0, 0], [0.392, 0.584, 0.929], [1, 1, 1]], dtype=float)		# steel + aluminium; black, blue, white
	# color = np.array([[0, 0, 0], [0.4627, 0.933, 0.776], [1, 1, 1]], dtype=float)		# steel + titanium; black, green, white
	I = np.zeros((nelx*nely, 3), dtype=float)
	for j in range(p):
		I[:, 0:3] = I[:, 0:3] + np.reshape(alpha[:, j], (nelx*nely, 1))*color[j, 0:3]
	# __ bilinear interpolation
	Ire = np.reshape(I, (nely, nelx, 3), order='F')
	dim = (10*nelx, 10*nely)
	I = cv2.resize(Ire, dim, interpolation=cv2.INTER_LINEAR)
	return I
