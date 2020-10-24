from singletop.static.FEMM import *
from singletop.static.filterr import *
from singletop.static.OC import *
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
tic()				# start tic-toc timer
nelx = 210			# number of elements along x axis
nely = 40			# number of elements along y axis
vf = 0.5			# volume fraction
rf = 3				# filter radius [mm]
p = 3				# penalization factor
iter_max = 200		# maximum number of iterations
# __ max and min stiffness - based on the SIMP modified approach
Emin = 1e-9			# Young's elastic modulus assigned to void [MPa]
Emax = 1			# Young's elastic modulus assigned to material [MPa]
# ----------PRINT----------
print()
print("SINGLE-MATERIAL TOPOLOGY OPTIMIZATION")
print("Minimum compliance design")
print("elements: " + str(nelx) + " x " + str(nely))
print("volume fraction: " + str(vf) + ", rf: " + str(rf) + ", p: " + str(p))
print()
# ----------FILTER----------
(H, Hs, nele) = filter(nelx, nely, rf)
# ----------ALLOCATION OF VARIABLES----------
x = vf*np.ones(nele, dtype=float)		# allocation of design variable
xold = x.copy()
x = x.copy()
g = 0					# must be initialized for OC
# __ initialize plot and plot the initial design
plt.figure(1)
plt.ion()				# ensure that redrawing is possible
plt.style.use('grayscale')
# __ allocation for plot
changes = np.array([])		# history of the density change (plot)
cs = np.array([])			# history of the compliance (plot)
iters = np.array([])		# history of the iterations (plot)
# __ filtering allocation
dv = np.ones(nele, dtype=float)		# sensitivity of the volume constraint
dc = np.ones(nele, dtype=float)		# sensitivity of compliance
ce = np.ones(nele, dtype=float)		# substitution for UT*KE*U calculation
# __ set iter counter and gradient vectors
iter = 0
change = 1
while change > 0.01 and iter < iter_max:
	iter = iter + 1
	# ----------FINITE ELEMENT METHOD----------
	(U, edofMat, KE) = FEM(nelx, nely, x, p, Emin, Emax)
	# ----------SENSITIVITY ANALYSIS----------
	ce[:] = (np.dot(U[edofMat].reshape(nele, 8), KE)*U[edofMat].reshape(nele, 8)).sum(1)
	c = ((Emin + x**p*(Emax - Emin))*ce).sum()				# total compliance
	dc[:] = (-p*x**(p-1)*(Emax-Emin))*ce					# sensitivity
	dv[:] = np.ones(nele)										# sensitivity of the volume constraint
	# ----------APPLICATION OF SENSITIVITY FILTER----------
	dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:, 0]/np.maximum(0.001, x)
	# ----------OPTIMALITY CRITERIA----------
	xold[:] = x
	(x[:], g) = OC(nelx, nely, x, dc, dv, g)
	# __ compute the change of density by the inf. norm
	change = np.linalg.norm(x.reshape(nelx*nely, 1) - xold.reshape(nelx*nely, 1), np.inf)
	# ----------PLOTTING----------
	# __ plot convergence in terms of compliance
	plt.figure(1)
	cs = np.append(cs, c)
	iters = np.append(iters, iter)
	plt.plot(iters, cs, color='k')
	plt.title('Minimization of compliance')
	plt.xlabel('Iterations [-]')
	plt.ylabel('Compliance [mm/N]')
	plt.show()
	plt.pause(0.001)		# draw now
	# __ plot convergence in terms of density change
	plt.figure(2)
	changes = np.append(changes, change)
	plt.plot(iters, changes, color='k')
	plt.title("Convergence of the design variable")
	plt.xlabel('Iterations [-]')
	plt.ylabel('Change of design variable [-]')
	plt.show()
	plt.pause(0.001)		# draw now
	# __ plot optimized shape
	plt.figure(3)
	plt.imshow(-x.reshape((nelx, nely)).T)
	plt.show()
	plt.pause(0.001)		# draw now
	# __ print iteration history
	print("it.: {0}, change: {1:.4f}, c: {2:.3f}".format(iter, change, c))
plt.savefig('static.png')		# save the optimized shape from last iteration
toc()			# end tic-toc timer
