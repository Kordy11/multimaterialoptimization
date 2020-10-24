import numpy as np
import numpy.matlib
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs


def FEM(nelx, nely, rho, EK, EM, a, optf):
	"""
	FINITE ELEMENT METHOD
	Parameters
	----------
	'nelx' and 'nely' are the number of elements along the two axis
	'rho' is density
	'EK' is elasticity tensor
	'EM' is mass tensor
	'a' is active phase
	'optf' is index of eigen frequency to optimize
	----------
	Returns
	-------
	'eigenF' is the eigen value
	'eigenM' is the eigen vector
	'edofMat' is matrix of degrees of freedom for each element
	'KE' is element stiffness matrix
	-------
	"""
	nele = int(nelx*nely)					# number of elements
	ndof = int(2*(nelx + 1)*(nely + 1))		# number of degrees of freedom
	# ----------NODE IDS------------
	xn = np.linspace(0, nelx, nelx + 1, dtype=int)		# number of nodes = nelx + 1
	yn = np.linspace(0, nely, nely + 1, dtype=int)
	inn, jn = np.meshgrid(xn, yn)
	nodeids = inn*nely + jn + inn
	edofVec = np.reshape(2*nodeids[:-1, :-1] + 2, (nele, 1), order='F')
	aedofMat = np.array([0, 1, 2*nely + 2, 2*nely + 3, 2*nely, 2*nely + 1, -2, -1], dtype=int)
	edofMat = np.matlib.repmat(edofVec, 1, 8) + np.matlib.repmat(aedofMat, nele, 1)
	# __ rotation constraint on left edge and support on the right edge
	fixednidl = nely/2
	fixednidr = (nely+1)*nelx+nely/2
	fixednid = np.append(fixednidl, fixednidr)
	fixeddofs = np.append(fixednid*2, fixednid*2 + 1).reshape(4, 1)
	fixeddofs = np.delete(fixeddofs, 1)
	alldofs = np.arange(2*(nelx+1)*(nely+1))
	freedofs = np.setdiff1d(alldofs, fixeddofs)
	# ----------ELEMENT STIFFNESS MATRIX----------
	nu = 0.3
	k = np.array([1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8, -1/4 + nu/12,\
				-1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
	KE = (1/(1-nu**2))*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
								[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
								[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
								[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
								[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
								[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
								[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
								[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
	# ----------ELEMENT MASS MATRIX----------
	ME = ((rho[a])/36)*np.array([[4, 0, 2, 0, 1, 0, 2, 0],
							[0, 4, 0, 2, 0, 1, 0, 2],
							[2, 0, 4, 0, 2, 0, 1, 0],
							[0, 2, 0, 4, 0, 2, 0, 1],
							[1, 0, 2, 0, 4, 0, 2, 0],
							[0, 1, 0, 2, 0, 4, 0, 2],
							[2, 0, 1, 0, 2, 0, 4, 0],
							[0, 2, 0, 1, 0, 2, 0, 4]])
	# construct the index pointers for the coo format
	iKM = np.kron(edofMat, np.ones((8, 1), dtype=int)).flatten()
	jKM = np.kron(edofMat, np.ones((1, 8), dtype=int)).flatten()
	sK = ((KE.flatten()[np.newaxis]).T*EK).flatten(order='F')
	K = coo_matrix((sK, (iKM, jKM)), shape=(ndof, ndof)).tocsc()
	mK = ((ME.flatten()[np.newaxis]).T*EM).flatten(order='F')
	M = coo_matrix((mK, (iKM, jKM)), shape=(ndof, ndof)).tocsc()
	# remove constrained dofs from matrix
	M = M[freedofs, :][:, freedofs]
	K = K[freedofs, :][:, freedofs]
	# ----------SOLVE SYSTEM----------
	eigenF, eigM = eigs(K, k=optf+1, M=M, sigma=0)
	# __ return sorted eigenvectors and eigenvalues
	eigenF = np.sort(eigenF)		# return sorted eigen values
	eigenF = eigenF[optf]
	eigenF = (np.real(eigenF))**0.5/(2*np.pi)		# frequency in Hz
	eigenM = np.zeros((ndof, 1), dtype=float)
	eigenM[freedofs, :] = eigM[:, optf][np.newaxis].T		# addition of fixed dofs
	eigenM = eigenM[:, eigenF.argsort()]					# return sorted eigen vectors
	return eigenF, eigenM, KE, ME, edofMat, nele
