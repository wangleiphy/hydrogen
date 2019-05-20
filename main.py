import autograd.numpy as np 
from autograd import value_and_grad
from scipy.optimize import minimize
import matplotlib.pyplot as plt 

def generalize_eigh(H, S):
    '''
    H v = w S v 
    '''
    s, u = np.linalg.eigh(S)
    V = np.dot(u, np.diag(1/np.sqrt(s)))
    w, v = np.linalg.eigh(np.dot(np.dot(V.transpose(), H), V))
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
    psi = np.sum(np.exp(-alpha[:, None]*rmesh), axis=0)
    plt.cla()
    plt.plot(rmesh, psi)
    plt.draw()
    plt.pause(0.05)

if __name__=='__main__':
    N = 4 # number of basis
    x0 = np.random.rand(N)
    res = minimize(closure, x0, method='L-BFGS-B', jac=True, callback=plotpsi, options={'gtol': 1e-07, 'disp':True})