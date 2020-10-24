import numpy as np
import numpy.matlib
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def FEM(nelx, nely, x, p, Emin, Emax):
	"""
	FINITE ELEMENT METHOD
	Parameters
	----------
	'nelx' and 'nely' are the number of elements along the two dimensions
	'x' is a nely-by-nelx matrix representing the density field
	'p' is a penalization factor used for SIMP model
	'Emin' is elasticity modulus assigned to void
	'Emax' is elasticity modulus assigned to material
	----------
	Returns
	-------
	'U' displacement vector
	'edofMat' is matrix of degrees of freedom corresponding to each element
	'KE' is an element stiffness matrix
	-------
	"""
	nele = int(nelx*nely)				# number of elements
	ndof = int(2*(nelx+1)*(nely+1))		# number of degrees of freedom
	# ----------NODE IDS------------
	xn = np.linspace(0, nelx, nelx+1, dtype=int)
	yn = np.linspace(0, nely, nely+1, dtype=int)
	inn, jn = np.meshgrid(xn, yn)
	nodeids = inn*nely+jn+inn
	edofVec = np.reshape(2*nodeids[:-1, :-1]+2, (nele, 1), order='F')
	aedofMat = np.array([0, 1, 2*nely+2, 2*nely+3, 2*nely, 2*nely+1, -2, -1], dtype=int)
	edofMat = np.matlib.repmat(edofVec, 1, 8)+np.matlib.repmat(aedofMat, nele, 1)
	xf = int(0)
	yf = np.linspace(0, nely, nely+1, dtype=int)
	iff, jf = np.meshgrid(xf, yf)
	fixednid = iff*nely+jf+iff
	fixeddofs = np.concatenate((fixednid, fixednid+nely+1), axis=0)
	alldofs = np.arange(2*(nelx+1)*(nely+1))
	freedofs = np.setdiff1d(alldofs, fixeddofs)
	# ----------DEFINE LOADS AND SUPPORTS----------
	xl = int(nelx)
	yl = int(nely)
	il, jl = np.meshgrid(xl, yl)
	loadnid = il * nely + jl + il
	loaddofs = int(2*loadnid + 1)
	F = np.zeros((len(alldofs), 1), dtype=int)
	F[loaddofs] = -1
	# ----------ELEMENT STIFFNESS MATRIX----------
	E = 1
	nu = 0.3
	k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8, -1/4 + nu/12,\
			-1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
	KE = E/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
								[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
								[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
								[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
								[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
								[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
								[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
								[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
	# construction of index pointers for the sparse coo matrix format
	iK = np.kron(edofMat, np.ones((8, 1), dtype=int)).flatten()
	jK = np.kron(edofMat, np.ones((1, 8), dtype=int)).flatten()
	sK = ((KE.flatten()[np.newaxis]).T*(Emin + x**p*(Emax - Emin))).flatten(order='F')
	K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()		# global stiffness matrix
	# remove constrained DOFs from stiffness matrix
	K = K[freedofs, :][:, freedofs]
	# ----------SOLVE SYSTEM----------
	U = np.zeros((ndof, 1))
	U[freedofs, 0] = spsolve(K, F[freedofs, 0])
	return U, edofMat, KE
