import numpy as np
from numpy.linalg import eigh

from scipy.linalg import expm
from scipy.sparse import bmat
from scipy.linalg import ishermitian

# Hamiltonian parameters
params=dict(C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6)

# Set of Pauli matrices for spin
sigma0 = np.array([[1, 0], [0, 1]])
sigmaX = np.array([[0, 1], [1, 0]])
sigmaY = np.array([[0, -1j], [1j, 0]])
sigmaZ = np.array([[1, 0], [0, -1]])


# Set of Pauli matrices for pseudo-spin
lambda0 = np.array([[1, 0], [0, 1]])
lambdaX = np.array([[0, 1], [1, 0]])
lambdaY = np.array([[0, -1j], [1j, 0]])
lambdaZ = np.array([[1, 0], [0, -1]])





#####################################################################
##################### ANALYTICAL EXPRESSION #########################
#####################################################################


# hyperbolic cosecant function
def csch(x):

    return 1. / np.sinh(x)


# function which determines the coefficients lambda_i and gamma_i
def parameters(irow, icol, L, omega, C0 = -0.0068, D1 = 1.3, A1 = 2.2, M0 = 0.28, B1 = 10, hbar = 1.):

    # switch over non-zero components
    match (irow,icol):
            
        # spin-up components
        case (0,0) | (0,2) | (2,0) | (2,2): p = +1

        # spin-down components
        case (1,1) | (1,3) | (3,1) | (3,3): p = -1
                                   
        # trivial components
        case _: p = 2

    
    # E0 plus/minus
    E0p = C0 + M0 + p*L - hbar*omega; E0m = C0 - M0 + p*L - hbar*omega
            
    # parameters a,b,c,d
    a = -E0p/(B1-D1); b = 1j*p*A1/(B1-D1); c = E0m/(B1+D1); d = -1j*p*A1/(B1+D1)

    # parameters A,B,C,D
    A = a+c+b*d; B = a*c; C = 1./(b*c); D = (a+b*d)/(b*c); Delta = np.sqrt(A**2-4*B)

    # lambda parameters
    lambda1 = np.sqrt((A-Delta)/2); lambda2 = np.sqrt((A+Delta)/2)
    # gamma parameters
    gamma1 = lambda1*( C*(lambda1**2)-D ); gamma2 = lambda2*( C*(lambda2**2)-D ) 
   
    return lambda1, lambda2, gamma1, gamma2



# function which define the coefficients alpha(sigma,lambda,lambda')
def coefficients(irow, icol, d, lambda1, lambda2, gamma1, gamma2, alpha, beta):

    # switch over non-zero components
    match (irow,icol):
            
        # (lambda, lambda') = (+,+)
        case (0,0) | (1,1):
            # coefficient alpha_1
            a1 = alpha*gamma2*csch(lambda1*d)/(gamma1*lambda2-gamma2*lambda1)
            # coefficient alpha_2
            a2 = -alpha*gamma1*csch(lambda2*d)/(gamma1*lambda2-gamma2*lambda1)
        
        # (lambda, lambda') = (-,+)
        case (2,0) | (3,1): 
            # coefficient alpha_1
            a1 = alpha*gamma1*gamma2*csch(lambda1*d)/(gamma1*lambda2-gamma2*lambda1)
            # coefficient alpha_2
            a2 = -alpha*gamma1*gamma2*csch(lambda2*d)/(gamma1*lambda2-gamma2*lambda1)

        # (lambda, lambda') = (+,-)
        case (0,2) | (1,3):
            # coefficient alpha_1
            a1 = beta*csch(lambda1*d)/(gamma1*lambda1-gamma2*lambda2)
            # coefficient alpha_2
            a2 = -beta*csch(lambda2*d)/(gamma1*lambda1-gamma2*lambda2)
                    

        # (lambda, lambda') = (-,-)
        case (2,2) | (3,3):
            # coefficient alpha_1
            a1 = beta*gamma1*csch(lambda1*d)/(gamma1*lambda1-gamma2*lambda2)
            # coefficient alpha_2
            a2 = -beta*gamma2*csch(lambda2*d)/(gamma1*lambda1-gamma2*lambda2)
                    
        # trivial components
        case _:
    
            a1 = 0.; a2 = 0.
                
    return a1, a2


# function evaluating the Green's function analytically for kx=ky=0
def GMTI_analytical(d, z, Z, L, omega, C0 = -0.0068, D1 = 1.3, A1 = 2.2, M0 = 0.28, B1 = 10, hbar = 1.):

    # non-homogeneous terms
    alpha = hbar/(B1-D1); beta = -hbar/(B1+D1)
    
    # define empty Green function
    GF = np.zeros((4,4), dtype='complex')

    # loop over rows
    for irow in range(4):
        
        # loop over columns
        for icol in range(4):

            # parameters depending on Hamiltonian 
            l1, l2, g1, g2 = parameters(irow, icol, L, omega, C0=C0, D1=D1, A1=A1, M0=M0, B1=B1, hbar=hbar)
            
            # coefficients for the Green's function components
            a1, a2 = coefficients(irow, icol, d, l1, l2, g1, g2, alpha, beta)

            # switch over non-zero components
            match (irow,icol):
            
                # g11 and g22
                case (0,0) | (1,1): 
                    
                    gij = a1*np.sinh(l1*z)*np.sinh(l1*(Z-d)) + a2*np.sinh(l2*z)*np.sinh(l2*(Z-d)) if z <= Z else a1*np.sinh(l1*Z)*np.sinh(l1*(z-d)) + a2*np.sinh(l2*Z)*np.sinh(l2*(z-d))

                # g31 and g42
                case (2,0) | (3,1): 
                    
                    gij = a1*np.cosh(l1*z)*np.sinh(l1*(Z-d)) + a2*np.cosh(l2*z)*np.sinh(l2*(Z-d)) if z <= Z else a1*np.cosh(l1*(z-d))*np.sinh(l1*Z) + a2*np.cosh(l2*(z-d))*np.sinh(l2*Z)

                # g13 and g24
                case (0,2) | (1,3): 
                    
                    gij = a1*np.sinh(l1*z)*np.cosh(l1*(Z-d)) + a2*np.sinh(l2*z)*np.cosh(l2*(Z-d)) if z <= Z else a1*np.sinh(l1*(z-d))*np.cosh(l1*Z) + a2*np.sinh(l2*(z-d))*np.cosh(l2*Z)

                # g33 and g44
                case (2,2) | (3,3): 
                    
                    gij = a1*np.cosh(l1*z)*np.cosh(l1*(Z-d)) + a2*np.cosh(l2*z)*np.cosh(l2*(Z-d)) if z <= Z else a1*np.cosh(l1*Z)*np.cosh(l1*(z-d)) + a2*np.cosh(l2*Z)*np.cosh(l2*(z-d))

                # trivial components
                case _:
                    
                    gij = 0.

            # fill the matrix for the Green's function
            GF[irow][icol] = gij

    return GF
    
    
    
