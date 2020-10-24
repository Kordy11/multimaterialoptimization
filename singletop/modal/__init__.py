from __future__ import division
from singletop.modal.filterr import *
from singletop.modal.FEMM import *
from singletop.modal.getsensitivity import *
from singletop.modal.OC import *
import numpy as np
import matplotlib.pyplot as plt
# full print
import sys
np.set_printoptions(threshold=sys.maxsize)


# ----------TIC TOC TIMER----------
def tic():
	import time
	global startTime_for_tictoc
	startTime_for_tictoc = time.time()


def toc():
	import time
	if 'startTime_for_tictoc' in globals():
		print()
		print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
	else:
		print()
		print("Toc: start time not set")


# ----------INPUT PARAMETERS----------
tic()					# start tic-toc timer
nelx = 250				# number of elements along x axis
nely = 60				# number of elements along y axis
Emin = 1e-9				# Young's elastic modulus assigned to void
Emax = 210e3			# Young's elastic modulus assigned to material [MPa]; Esteel = 210e3 MPa
nu = 0.3				# Poisson's ratio [-]
rho = 7.8e-9			# density [t/mm3]
vf = 0.5				# volume fraction
rf = 3					# filter radius [mm]
pK = 3					# penalization factor - stiffness
pM = 3					# penalization factor - mass
iter_max = 200			# maximum number of iterations
optf = 0				# index of eigen frequency to optimize, 1st eigenfreq is 0 (Python counting from 0)
# ----------PRINT----------
print()
print("SINGLE-MATERIAL TOPOLOGY OPTIMIZATION")
print("Maximization of eigenfrequency problem")
print("elements: " + str(nelx) + " x " + str(nely))
print("volume fraction: " + str(vf) + ", rf: " + str(rf) + ", p: " + str(pK))
print()
# ----------FILTER----------
(H, Hs, nele) = filter(nelx, nely, rf)
# ----------ALLOCATION OF VARIABLES----------
x = vf*np.ones(nelx*nely, dtype=float)		# allocation of design variable
xold = x.copy()
g = 0					# must be initialized for OC
# __ optimization cycle setting
iter = 0
change = 1
# __ allocation and initialization for plot
plt.figure(1)
plt.ion()						# ensure that redrawing is possible
plt.style.use('grayscale')
changes = np.array([])			# history of the density change (plot)
eigenFs = np.array([])			# history of the eigen frequency (plot)
iters = np.array([])			# history of the iterations (plot)
while change > 0.01 and iter < iter_max:
	iter = iter + 1
	# ----------FINITE ELEMENT METHOD----------
	(eigenF, eigenM, KE, ME, edofMat, nele) = FEM(nelx, nely, x, pK, pM, Emin, Emax, rho, optf)
	# ----------SENSITIVITY ANALYSIS----------
	df = getsensitivity(nelx, nely, x, eigenF, eigenM, pK, pM, KE, ME, edofMat)
	# ----------APPLICATION OF SENSITIVITY OR DENSITY FILTER----------
	df = np.asarray((H*(x.flatten(order='F')*df))[np.newaxis].T/Hs)[:, 0]/np.maximum(0.001, x)
	# ----------OPTIMALITY CRITERIA----------
	xold[:] = x
	x[:] = OC(nelx, nely, x, df, vf)
	# __ compute the change of density by the infinity norm
	change = np.linalg.norm(x-xold, np.inf)
	# ----------PLOTTING----------
	# __ plot convergence in terms of eigen frequency
	plt.figure(1)
	eigenFs = np.append(eigenFs, eigenF)
	iters = np.append(iters, iter)
	plt.plot(iters, eigenFs, color='black')
	plt.xlabel('Iterations [-]')
	plt.ylabel('Frequency [Hz]')
	plt.title("Maximization of eigen frequency")
	plt.pause(0.001)		# draw now
	plt.show()
	# __ plot convergence in terms of density change
	plt.figure(2)
	changes = np.append(changes, change)
	plt.plot(iters, changes, color='k')
	plt.xlabel('Iterations [-]')
	plt.ylabel('Change of design variable [-]')
	plt.title("Convergence of the design variable")
	plt.show()
	plt.pause(0.001)		# draw now
	# __ plot optimized shape
	plt.figure(4)
	plt.imshow(-x.reshape((nely, nelx), order='F'))
	plt.show()
	plt.pause(0.001)		# draw now
	# __ print iteration history
	print("iter: {0}, change: {1:.4f}, f: {2:.3f}".format(iter, change, eigenF) + ' Hz')
plt.savefig('modal.png')		# save density field from last iteration
toc()			# end tic-toc timer
