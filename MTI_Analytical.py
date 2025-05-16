import numpy as np

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
###################### ANALYTICAL SOLUTION G0 #######################
#####################################################################



# function which determines the coefficients lambda_i and gamma_i
def parameters(spin, L, omega, all=False, C0 = -0.0068, D1 = 1.3, A1 = 2.2, M0 = 0.28, B1 = 10, hbar = 1.):

    # switch over non-zero components
    match spin:
            
        # spin-up components
        case 'up': p = +1

        # spin-down components
        case 'down': p = -1
                                   
        # trivial components
        case _: p = 0

    
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

    if all==False:
        return lambda1, lambda2, gamma1, gamma2
    else:
        return lambda1, lambda2, gamma1, gamma2, E0p, E0m, A,B,C,D, a,b,c,d



# define non-homogeneity alpha and a_i coefficients
def coefficients(spin, L, omega, C0 = -0.0068, D1 = 1.3, A1 = 2.2, M0 = 0.28, B1 = 10, hbar = 1.):

    # get coefficients lambda_i, gamma_i
    l1, l2, g1, g2 = parameters(spin=spin, L=L, omega=omega, C0=C0, D1=D1, A1=A1, M0=M0, B1=B1, hbar=hbar)

    # compute non-homogeneity
    aplus = hbar/(B1-D1); aminus = -hbar/(B1+D1)

    # initialize matrix for coefficients a_{1,2}
    a_coeff = np.empty((2,2), dtype='object')
    
    
    ### coefficients ++ ###
    # coefficient a1
    a1 = aplus * (g1*l1-g2*l2) / ( l1*l2*(g1-g2) )
    # coefficient a2
    a2 = -1j*aplus * (g1*l1-g2*l2)*(g2*l1+g1*l2) / ( l1*l2*(g1-g2)*(g2*l1-g1*l2) )
    # add to matrix
    a_coeff[0][0] = (a1, a2)

    ### coefficients +- ###
    # coefficient a1
    a1 = aminus * (l1-l2) / ( l1*l2*(g1-g2) )
    # coefficient a2
    a2 = -1j*aminus * (l1+l2) / ( l1*l2*(g1-g2) )
    # add to matrix
    a_coeff[0][1] = (a1, a2)

    ### coefficients -+ ###
    # coefficient a1
    a1 = -aplus * g1*g2*(l1-l2) / ( l1*l2*(g1-g2) )
    # coefficient a2
    a2 = 1j*aplus * g1*g2*(l1+l2) / ( l1*l2*(g1-g2) )
    # add to matrix
    a_coeff[1][0] = (a1, a2)

    ### coefficients -- ###
    # coefficient a1
    a1 = aminus * (g1*l2-g2*l1) / ( l1*l2*(g1-g2) )
    # coefficient a2
    a2 = -1j*aminus * (g1*l2-g2*l1)*(g1*l1+g2*l2) / ( l1*l2*(g1-g2)*(g1*l1-g2*l2) )
    # add to matrix
    a_coeff[1][1] = (a1, a2)

    return  a_coeff, l1.real, l1.imag



# function for change of basis in MTI Green's function
# from parity block-basis to spin-block basis
def Change_Basis(gf):

    g_new = [[gf[0][0], gf[0][2], gf[0][1], gf[0][3]],
             [gf[2][0], gf[2][2], gf[2][1], gf[2][3]],
             [gf[1][0], gf[1][2], gf[1][1], gf[1][3]],
             [gf[3][0], gf[3][2], gf[3][1], gf[3][3]]]
    
    return np.array(g_new)



# evaluate GF at z=0 in the semi-infinite case
def GMTI_SemiInfinite(d, Z, L, omega, C0 = -0.0068, D1 = 1.3, A1 = 2.2, M0 = 0.28, B1 = 10, hbar = 1.):

    g = []
    for spin in ['up', 'down']:

        # initialize matrix
        g_sigma = np.empty((2,2), dtype='complex')
        # get coefficients
        a_coeff, l, k = coefficients(spin=spin, L=L, omega=omega, C0=C0, D1=D1, A1=A1, M0=M0, B1=B1, hbar=hbar)

        
        for idx in range(2):
            for jdx in range(2):

                a1 = a_coeff[idx][jdx][0]; a2 = a_coeff[idx][jdx][1]
                g_sigma[idx][jdx] = math.exp(-l*Z) * ( a1*np.cos(k*Z) + a2*np.sin(k*Z) )

        g.append(g_sigma)
                
    return np.block([[g[0], np.zeros((2, 2))], [np.zeros((2, 2)), g[1]]])





#####################################################################
###################### ANALYTICAL SOLUTION F2 #######################
#####################################################################


# Function evaluating the F2-MTI using the G0 obtained analytically in the semi-infinite limit
def FMTI2_SemiInfinite(d, z, Z, L, mu, Delta, omega, Gamma, C = -0.0068, D1 = 1.3, A1 = 2.2, M = 0.28, B1 = 10, hbar=1., t=1.):

    # GMTI(z0,z)
    GMTIz = GMTI_SemiInfinite(d=d, Z=z, L=L, omega=-omega, C0=C, D1=D1, A1=A1, M0=M, B1=B1, hbar=hbar)
    
    # GMTI(z0,z')
    GMTIZ = GMTI_SemiInfinite(d=d, Z=Z, L=L, omega=omega, C0=C, D1=D1, A1=A1, M0=M, B1=B1, hbar=hbar)
    
    # FSC(0)
    FSC = FSC_matrix(z=0., kx=0., ky=0., mu=mu, Delta=Delta, w=omega, t=t, hbar=hbar)
    
    return (np.transpose(GMTIz) @ np.conj(Gamma) @ FSC @ Gamma.H @ GMTIZ).A
