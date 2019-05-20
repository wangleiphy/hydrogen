import autograd.numpy as np 
from autograd import value_and_grad
from scipy.optimize import minimize
import matplotlib.pyplot as plt 

def generalize_eigh(H, S):
    '''
    H v = S v w
    where U^{T} S U = I
    '''
    s, u = np.linalg.eigh(S)
    U = np.dot(u, np.diag(1/np.sqrt(s)))
    w, v = np.linalg.eigh(np.dot(np.dot(U.transpose(), H), U))
    return w, v 

def energy(params):
    '''
    H = (-0.5 laplacian -1/r)
    basis exp(-alpha * r^2)
    '''
    T = 3*params[:, None] * params * np.pi**1.5/(params[:, None] + params)**2.5
    A = -2*np.pi/(params[:, None] + params)
    H = T + A
    S = (np.pi/(params[:, None]+ params))**1.5
    w, v = generalize_eigh(H, S) 
    return w[0]

energy_and_gradient = value_and_grad(energy)

def closure(x):
    alpha = np.exp(x) # make sure the exponents are positive
    e, g = energy_and_gradient(alpha)
    return e, g

def plotpsi(x):
    alpha = np.exp(x)
    rmesh = np.linspace(0, 10, 101)
    psi = np.exp(-alpha[:, None]*rmesh)
    plt.cla()
    for i in range(len(x)): # plot basis functions, its coefs are in the generalize eigh solver
        plt.plot(rmesh, psi[i])
    plt.xlabel('$r$')
    plt.ylabel('$\psi$')
    plt.draw()
    plt.pause(0.05)

if __name__=='__main__':
    np.random.seed(42)

    N = 4 # number of basis
    x0 = np.random.randn(N)
    res = minimize(closure, x0, method='L-BFGS-B', jac=True, callback=plotpsi, options={'gtol': 1e-07, 'disp':True})
