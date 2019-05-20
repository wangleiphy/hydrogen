import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

def energy(x):
    '''
    H = (-0.5 laplacian -1/r)
    psi = coef * exp(-params * r^2)
    '''
    params = np.exp(x[:len(x)//2])
    coefs = x[len(x)//2:]

    T = 3*params[:, None] * params * np.pi**1.5/(params[:, None] + params)**2.5
    A = -2*np.pi/(params[:, None] + params)
    H = T + A
    S = (np.pi/(params[:, None]+ params))**1.5

    up = np.dot(np.dot(coefs, H), coefs)
    down = np.dot(np.dot(coefs, S), coefs)
    return up/down

energy_and_gradient = value_and_grad(energy)

def closure(x):
    e, g = energy_and_gradient(x)
    return e, g

if __name__=='__main__':
    np.random.seed(42)

    N = 4 # number of basis
    x0 = np.random.randn(2*N)
    res = minimize(closure, x0, method='L-BFGS-B', jac=True, options={'gtol': 1e-07, 'disp':True})
