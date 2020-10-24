import numpy as np
from scipy.sparse import coo_matrix


def filter(nelx, nely, rf):
	"""
	FILTER
	Parameters
	----------
	'nelx' and 'nely' are the number of elements along the two dimensions
	'rf' is a filter radius
	----------
	Returns
	-------
	'H' is a weighted factor
	'Hs' is a summation of H
	'nele' is a total number of elements
	-------
	"""
	nele = int(nelx*nely)		# number of elements
	nfilter = nele*((2*(np.ceil(rf)-1)+1)**2)
	iH = np.zeros(int(nfilter))
	jH = np.zeros(int(nfilter))
	sH = np.zeros(int(nfilter))
	cc = 0
	for i in range(nelx):
		for j in range(nely):
			row = i*nely+j
			kk1 = int(np.maximum(i - (np.ceil(rf) - 1), 0))
			kk2 = int(np.minimum(i + np.ceil(rf), nelx))
			ll1 = int(np.maximum(j - (np.ceil(rf) - 1), 0))
			ll2 = int(np.minimum(j + np.ceil(rf), nely))
			for k in range(kk1, kk2):
				for l in range(ll1, ll2):
					col = k * nely + l
					fac = rf - np.sqrt(((i-k)*(i-k) + (j-l)*(j-l)))
					iH[cc] = row
					jH[cc] = col
					sH[cc] = np.maximum(0.0, fac)
					cc = cc+1
	# assembly of weighted factor + summation
	H = coo_matrix((sH, (iH, jH)), shape=(nele, nele)).tocsc()
	Hs = H.sum(1)
	return H, Hs, nele
