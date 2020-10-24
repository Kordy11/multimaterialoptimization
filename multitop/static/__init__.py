from multitop.static.filterr import *
from multitop.static.bi_top import *
from multitop.static.make_bitmap import *
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
nelx = 200				# number of elements along x axis
nely = 40				# number of elements along y axis
tol_out = 0.025			# tolerance of outer cycle
tol_f = 0.05			# filter adjusting tolerance
iter_max_in = 2			# maximum number of iterations of inner cycle
iter_max_out = 500		# maximum number of iterations of outer cycle
rf = 8					# filter radius
p = 3					# penalization factor, p = nr. of materials
q = 3					# penalization factor used in elasticity tensor calculation
e = np.array([3, 1, 1e-9])				# elasticity modulus
e = np.reshape(e, (len(e), 1))			# transposition
vf = np.array([0.4, 0.2, 0.4])			# volume fraction
vf = np.reshape(vf, (len(vf), 1))		# transposition
# ----------PRINT----------
print()
print("MULTI-MATERIAL TOPOLOGY OPTIMIZATION")
print("Minimum compliance problem")
print("elements: " + str(nelx) + " x " + str(nely))
print()
print("Parameters for material nr. 1:")
print("Elastic modulus = " + str(e[0, 0]) + " MPa, " + "volume fraction = " + str(vf[0, 0]))
print()
print("Parameters for material nr. 2:")
print("Elastic modulus = " + str(e[1, 0]) + " MPa, " + "volume fraction = " + str(vf[1, 0]))
print()
# ----------FILTER----------
(H, Hs, nele) = filter(nelx, nely, rf)
# ----------ALLOCATION OF VARIABLES----------
# __ creation of design variables matrix as function of volume fraction
alpha = np.zeros((nelx*nely, p), dtype=float)
for i in range(p):
	alpha[:, i] = vf[i]
# __ set iter counter and gradient vectors
change_out = 2*tol_out
iter_out = 0
# __ allocation for plot
changes_out = np.array([])  # history of the density change (plot)
objs = np.array([])  # history of the compliance (plot)
iters_out = np.array([])  # history of the iterations (plot)
# __ initialize plot and plot the initial design
plt.figure(1)
plt.ion()		# ensure that redrawing is possible
while iter_out < iter_max_out and change_out > tol_out:
	alpha_old = alpha.copy()
	for a in range(p):				# a is one of phases (subscripts) of binary phase algorithm
		for b in range(a+1, p):		# b is another phase (background) of binary phase algorithm
			(obj, alpha) = bi_top(a, b, nelx, nely, p, q, vf, e, alpha, H, Hs, iter_max_in)
	iter_out = iter_out + 1
	change_out = LA.norm(alpha - alpha_old, np.inf)		# returns a vector infinity norm, max(abs(alpha - alpha_old))
	print("it.: {0}, change: {1:.4f}, c: {2:.3f}".format(iter_out, change_out, obj) + ' mm/N')
	# __ update filter - to avoid local minimums
	if change_out < tol_f and rf > 3:
		tol_f = 0.99*tol_f
		rf = 0.99*rf
		(H, Hs, nele) = filter(nelx, nely, rf)
	# ----------PLOTTING----------
	# __ plot convergence in terms of compliance
	plt.figure(1)
	objs = np.append(objs, obj)
	iters_out = np.append(iters_out, iter_out)
	plt.plot(iters_out, objs, color='k')
	plt.title('Minimization of compliance')
	plt.xlabel('Iterations [-]')
	plt.ylabel('Compliance [mm/N]')
	plt.pause(0.001)		# draw now
	# __ plot convergence in terms of density change
	plt.figure(2)
	changes_out = np.append(changes_out, change_out)
	plt.plot(iters_out, changes_out, color='k')
	plt.title("Convergence of the design variable")
	plt.xlabel('Iterations [-]')
	plt.ylabel('Change of design variable [-]')
	plt.show()
	plt.pause(0.001)		# draw now
	# __ plot bitmap
	I = make_bitmap(p, nelx, nely, alpha)
	plt.figure(3)
	plt.imshow(I)
	plt.axis('off')			# turn off the axis
	plt.show()
	plt.pause(0.001)		# draw now
plt.savefig('bitmap_static.png')		# save density field from last iteration
toc()			# end tic-toc timer
