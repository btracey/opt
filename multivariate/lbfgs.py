import numpy as np
import math
import scipy.optimize as optimize

def rosen(x):
	"""The Rosenbrock function"""
	s =  sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
	#print "rosen(x)=", s
	return s

def rosen_der(x):
	xm = x[1:-1]
	xm_m1 = x[:-2]
	xm_p1 = x[2:]
	der = np.zeros_like(x)
	der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
	der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
	der[-1] = 200*(x[-1]-x[-2]**2)
	print
	print "x=", x
	#print "der = ",der
	return der


m = 10
n = 3
pi = 3.14159265357938462
A = np.zeros((m,n))
B = np.zeros((m,1))
for i in range(m):
	for j in range(n):
		A[i,j] =  i*m +j
	B[i,0] = i

def myfun(x):
	ans = np.dot(A,x)
	loss = 0
	for i in range(m):
		diff = ans[i] - B[i,0]
		loss += diff * diff
	return loss

#def mygrad(x):



x0 =np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = optimize.minimize(rosen, x0, method='bfgs', jac=rosen_der, options={'disp': True}, tol=1e-2)

#x0 = np.zeros((n,1))
#x0[0,0] = 9.2
#x0[1,0] = 2.7
#x0[1,0] = 6.3
#res = optimize.minimize(myfun, x0, method='bfgs', options={'disp': True})
print res

#optx, optf, d = optimize.fmin_l_bfgs_b(rosen,x0, fprime=rosen_der, iprint=0)
#print optx
#print optf
#print d['funcalls']