from multitop.modal.filterr import *
from multitop.modal.bi_top import *
from multitop.modal.make_bitmap import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
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
nelx = 300				# number of elements along x axis
nely = 50				# number of elements along y axis
tol_out = 0.02			# tolerance of outer cycle
tol_f = 0.05			# filter adjusting tolerance
iter_max_in = 2			# maximum number of iterations of inner cycle
iter_max_out = 500		# maximum number of iterations of outer cycle
rf = 9					# filter radius
p = 2					# number of phases/materials
qK = 3					# stiffness penalization factor used in elasticity tensor calculation
qM = 3					# mass penalization factor used in elasticity tensor calculation
optf = 0				# eigenfrequency to optimize index - 1st (py 0) eigenfrequency
e = np.array([210e3, 70e3, 1e-9])				# elasticity modulus
e = np.reshape(e, (len(e), 1))					# transposition
v = np.array([0.4, 0.6, 0.4])					# volume fraction
v = np.reshape(v, (len(v), 1))					# transposition
rho = np.array([7.8e-9, 2.7e-9, 1e-12])		# density
rho = np.reshape(rho, (len(rho), 1))			# transposition
# ----------PRINT----------
print()
print("MULTI-MATERIAL TOPOLOGY OPTIMIZATION")
print("Maximization of eigen frequency problem")
print("elements: " + str(nelx) + " x " + str(nely))
print()
print("Parameters for material nr. 1:")
print("Elastic modulus = " + str(e[0, 0]) + " MPa, " + "volume fraction = " + str(v[0, 0]) + ", density = " + str(rho[0, 0]) + " t/mm3")
print()
print("Parameters for material nr. 2:")
print("Elastic modulus = " + str(e[1, 0]) + " MPa, " + "volume fraction = " + str(v[1, 0]) + ", density = " + str(rho[1, 0]) + " t/mm3")
print()
# ----------FILTER----------
(H, Hs, nele) = filter(nelx, nely, rf)
# ----------ALLOCATION OF VARIABLES----------
# __ creation of design variables matrix as function of volume fraction
alpha = np.zeros((nelx*nely, p), dtype=float)
for i in range(p):
	alpha[:, i] = v[i]
# __ set iter counter and gradient vectors
change_out = 2*tol_out
iter_out = 0
# __ allocation for plot
changes_out = np.array([])		# history of the density change (plot)
iters_out = np.array([])		# history of the iterations (plot)
itersc_out = np.array([])		# history of the iterations (plot)
eigenF = np.array([])
eigenFs = np.array([])			# history of the eigen frequency (plot)
# __ allocation and initialization for plot
plt.figure(1)
plt.ion()						# ensure that redrawing is possible
while iter_out < iter_max_out and change_out > tol_out:
	alpha_old = alpha.copy()
	for a in range(p):				# a is one of phases (subscripts) of binary phase algorithm
		for b in range(a+1, p):		# b is another phase (background) of binary phase algorithm
			(alpha, eigenF, iter_in) = bi_top(a, b, nelx, nely, p, qK, qM, v, e, rho, alpha_old, H, Hs, iter_max_in, optf)
			if iter_out == 0 and iter_in == 2 and a == 0 and b == 1:
				eigenFs = np.append(eigenFs, eigenF)
				iters_out = np.append(iters_out, iter_out)
	iter_out = iter_out + 1
	change_out = LA.norm(alpha - alpha_old, np.inf)
	print("iter: {0}, change: {1:.4f}, f: {2:.3f}".format(iter_out, change_out, eigenF) + ' Hz')
	# __ update filter - to avoid local minimums
	if change_out < tol_f and rf > 3:
		tol_f = 0.99*tol_f
		rf = 0.99*rf
		(H, Hs, nele) = filter(nelx, nely, rf)
	# ----------PLOTTING----------
	# __ plot convergence in terms of density change
	plt.figure(1)
	itersc_out = np.append(itersc_out, iter_out)
	changes_out = np.append(changes_out, change_out)
	plt.plot(itersc_out, changes_out, color='k')
	plt.xlabel('Iterations [-]')
	plt.ylabel('Change of design variable [-]')
	plt.title("Convergence of the design variable")
	plt.show()
	plt.pause(0.001)		# draw now
	# __ plot convergence in terms of density change
	plt.figure(2)
	iters_out = np.append(iters_out, iter_out)
	eigenFs = np.append(eigenFs, eigenF)
	plt.plot(iters_out, eigenFs, color='k')
	plt.xlabel('Iterations [-]')
	plt.ylabel('Frequency [Hz]')
	plt.title("Maximization of eigenfrequency")
	plt.show()
	plt.pause(0.001)		# draw now
	# __ plot bitmap
	I = make_bitmap(p, nelx, nely, alpha)
	plt.figure(3)
	plt.imshow(I)
	plt.axis('off')			# turn off the axis
	plt.show()
	plt.pause(0.001)		# draw now
plt.savefig('bitmap.png')		# save density field from last iteration
toc()			# end tic-toc timer
