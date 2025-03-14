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
###################### DIRICHLET B.C. SOLUTION ######################
#####################################################################


# Function defining the Matrix M
def Mmat(kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar = 1.):

    # abbreviations
    k = np.sqrt(kx**2+ky**2); e0 = C + D2*k**2; m0 = M - B2*k**2

    # denominators
    a = 1./(B1-D1); b = 1./(B1+D1)

    # Zero Matrix
    zero = np.zeros((4,4))

    # Unitary Matrix
    id = np.identity(4)

    # Matrix C (Constant)
    Cmat = np.array(
        [[-a*(e0+m0+L-hbar*w), 0, 0, -a*A2*(kx-1j*ky)],
         [0, -a*(e0+m0-L-hbar*w), -a*A2*(kx+1j*ky),0],
         [0, b*A2*(kx-1j*ky), b*(e0-m0+L-hbar*w), 0],
         [b*A2*(kx+1j*ky), 0, 0, b*(e0-m0-L-hbar*w)]])

    # Matrix D (first derivatives)
    Dmat = np.array(
        [[0, 0, 1j*a*A1, 0],
         [0, 0, 0, -1j*a*A1],
         [-1j*b*A1, 0, 0, 0],
         [0, 1j*b*A1, 0, 0]])
    
    # Compone the matrix M
    return np.block([[zero, id], [Cmat, Dmat]])



# Function computing the exponetial matrix exp(Mz)
def expMz(z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar = 1.):

    return expm( Mmat(kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)*z )



# Function defining the matrix A (Z=z', d=thickness)
def Amat(d, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar = 1.):

    # exponential at z=0
    eM0 = expMz(0, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # exponential at z=d
    eMd = expMz(d, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # exponential at z=d
    eMZ = expMz(Z, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # initialize an empty matrix
    A = np.empty([16, 16], dtype=complex)

    # fill in the matrix A
    for i in range(4):
        for j in range(8):

            # boundary conditions z=0
            A[i][j] = eM0[i][j]
            A[i][j+8] = 0

            # boundary conditions z=d
            A[i+4][j] = 0
            A[i+4][j+8] = eMd[i][j]

            # continuity condition z=z'
            A[i+8][j] = eMZ[i][j]
            A[i+8][j+8] = -eMZ[i][j]

            # derivative jump at z=z'
            A[i+12][j] = eMZ[i+4][j]
            A[i+12][j+8] = -eMZ[i+4][j]

    return A



# Function defining the non-homogeneous term y
def Yvec(icol, D1 = 1.3, B1 = 10, hbar = 1.):

    # initialize an empty vector
    y = np.zeros([16], dtype=complex)

    # fill in the nonzero value
    match icol:
    
        # column 1,2
        case 1 | 2:
            y[11+icol] = hbar/(B1-D1)
            
        # column 3,4
        case 3 | 4:
            y[11+icol] = -hbar/(B1+D1)

    return(y)



# Function for solving the system and finding the particular solution
def psolution_DirichletBC(icol, d, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar=1.):

    # vetcor y 
    y = Yvec(icol, B1=B1, D1=D1, hbar=hbar)
    
    # matrix A
    A = Amat(d, Z, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    return np.linalg.solve(A, y)



# Function evaluating the Green's function (Z=z', d=thickness)
def GMTI_DirichletBC(d, z, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar = 1.):

    # empty matrix for Green's function
    G = np.empty([4, 4], dtype='complex')

    # general solution M_ij at z
    eMz = expMz(z=z, kx=kx, ky=ky, L=L, w=w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # loop over columns (sigma',lambda')
    for icol in range(4):
        
        # particular solutions c_j
        cj = psolution_DirichletBC(icol=icol+1, d=d, Z=Z, kx=kx, ky=ky, L=L, w=w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
        
        # select left or right depending on z,z'
        cj = cj[:8] if z<Z else cj[8:]

        # loop over the 4 sigma,lambda states i=1,2,3,4
        for irow in range(4):
            
            # evaluate Green's function components 
            G[irow][icol] = np.sum( [eMz[irow][j]*cj[j] for j in range(8)] )
        
    return G
    
    
    

####### DIAGONAL SOLUTION (CROSS-CHECK) #######


# Analytic result for diagonal part in the trivial case (Z=z', d=thickness)
def GMTI_diagonal(d, z, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, M = 0.28, B1 = 10, B2 = 56.6, hbar = 1.):

    # abbreviations
    k = np.sqrt(kx**2+ky**2); e0 = C + D2*k**2; m0 = M - B2*k**2

    # empty array for diagonal elements
    gfdiag = []
    
    # loop over the four diagonal elements
    for idx in range(1,5):

        match idx:     
            case 1:
                hp = e0 + m0 + L; hz = B1-D1
            case 2:
                hp = e0 + m0 - L; hz = B1-D1
            case 3:
                hp = e0 - m0 + L; hz = -B1-D1
            case 4:
                hp = e0 - m0 - L; hz = -B1-D1

        # parameters
        alpha = complex(-hbar/hz,0); beta = complex(-(hbar*w-hp)/hz,0); k = np.sqrt(beta); gamma = np.tan(k*d)

        # left solution
        if z <= Z:
            gii = alpha/(k*gamma) * ( np.sin(k*Z) - gamma*np.cos(k*Z) ) * np.sin(k*z)
        # right solution
        else:
            gii = alpha/(k*gamma) * np.sin(k*Z) * ( np.sin(k*z) - gamma*np.cos(k*z) )

        gfdiag.append(gii)

    return np.array(gfdiag)





#####################################################################
###################### NEUMANN B.C. SOLUTION ########################
#####################################################################


# Function defining the matrix A_12 for the columns 1,2
# (B.C. vanishing x' and y' at the boundaries - Neumann boundary conditions)
def Amat_Neumann(d, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar = 1.):

    # exponential at z=0
    eM0 = expMz(0, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # exponential at z=d
    eMd = expMz(d, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # exponential at z=Z
    eMZ = expMz(Z, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # initialize an empty matrix
    A = np.empty([16, 16], dtype=complex)

    # fill in the matrix A
    for i in range(4):
        for j in range(8):

            # boundary conditions z=0
            # (vanishing of first-derivative for x-like components)
            if i==0 or i==1:
                A[i][j] = eM0[i+4][j]
                A[i][j+8] = 0
            # (vanishing of first-derivative for y-like components)
            if i==2 or i==3:
                A[i][j] = eM0[i+4][j]
                A[i][j+8] = 0
                
            
            # boundary conditions z=d
            # (vanishing of first-derivative for x-like components)
            if i==0 or i==1:
                A[i+4][j] = 0
                A[i+4][j+8] = eMd[i+4][j]
            # (vanishing of function for y-like components)
            if i==2 or i==3:
                A[i+4][j] = 0
                A[i+4][j+8] = eMd[i][j]


            # continuity condition z=z'
            A[i+8][j] = eMZ[i][j]
            A[i+8][j+8] = -eMZ[i][j]

            # derivative jump at z=z'
            A[i+12][j] = eMZ[i+4][j]
            A[i+12][j+8] = -eMZ[i+4][j]

    return A



# Function for solving the system and finding the particular solution
def psolution_Neumann(icol, d, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar=1.):

    # initialize an empty vector
    y = np.zeros([16], dtype=complex)
        
    # select matrix A and vector y
    match icol:
    
        # column 1,2
        case 1 | 2:
            # non-homogeneity vector 
            y[11+icol] = hbar/(B1-D1)

        # column 3,4
        case 3 | 4:
            # non-homogeneity vector 
            y[11+icol] = -hbar/(B1+D1)
            
    # matrix of equations
    A = Amat_Neumann(d, Z, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
            
    return np.linalg.solve(A, y)



# Function evaluating the Green's function (Z=z', d=thickness)
def GMTI_NeumannBC(d, z, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar = 1.):

    # empty matrix for Green's function
    G = np.empty([4, 4], dtype='complex')

    # general solution M_ij at z
    eMz = expMz(z=z, kx=kx, ky=ky, L=L, w=w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # loop over columns (sigma',lambda')
    for icol in range(4):
        
        # particular solutions c_j
        cj = psolution_Neumann(icol=icol+1, d=d, Z=Z, kx=kx, ky=ky, L=L, w=w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
        
        # select left or right depending on z,z'
        cj = cj[:8] if z<Z else cj[8:]

        # loop over the 4 sigma,lambda states i=1,2,3,4
        for irow in range(4):
            
            # evaluate Green's function components 
            G[irow][icol] = np.sum( [eMz[irow][j]*cj[j] for j in range(8)] )
        
    return G





#####################################################################
####################### MIXED B.C. SOLUTION #########################
#####################################################################


# Function defining the matrix A_12 for the columns 1,2
# (B.C. vanishing x' and y at the boundaries)
def Amat_12(d, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar = 1.):

    # exponential at z=0
    eM0 = expMz(0, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # exponential at z=d
    eMd = expMz(d, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # exponential at z=Z
    eMZ = expMz(Z, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # initialize an empty matrix
    A = np.empty([16, 16], dtype=complex)

    # fill in the matrix A
    for i in range(4):
        for j in range(8):

            # boundary conditions z=0
            # (vanishing of first-derivative for x-like components)
            if i==0 or i==1:
                A[i][j] = eM0[i+4][j]
                A[i][j+8] = 0
            # (vanishing of function for y-like components)
            if i==2 or i==3:
                A[i][j] = eM0[i][j]
                A[i][j+8] = 0
            
            # boundary conditions z=d
            # (vanishing of first-derivative for x-like components)
            if i==0 or i==1:
                A[i+4][j] = 0
                A[i+4][j+8] = eMd[i+4][j]
            # (vanishing of function for y-like components)
            if i==2 or i==3:
                A[i+4][j] = 0
                A[i+4][j+8] = eMd[i][j]

            # continuity condition z=z'
            A[i+8][j] = eMZ[i][j]
            A[i+8][j+8] = -eMZ[i][j]

            # derivative jump at z=z'
            A[i+12][j] = eMZ[i+4][j]
            A[i+12][j+8] = -eMZ[i+4][j]

    return A



# Function defining the matrix A_34 for the columns 3, 4
# (B.C. vanishing x and y' at the boundaries)
def Amat_34(d, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar = 1.):

    # exponential at z=0
    eM0 = expMz(0, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # exponential at z=d
    eMd = expMz(d, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # exponential at z=Z
    eMZ = expMz(Z, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # initialize an empty matrix
    A = np.empty([16, 16], dtype=complex)

    # fill in the matrix A
    for i in range(4):
        for j in range(8):

            # boundary conditions z=0
            # (vanishing of function for x-like components)
            if i==0 or i==1:
                A[i][j] = eM0[i][j]
                A[i][j+8] = 0
            # (vanishing of first-derivative for y-like components)
            if i==2 or i==3:
                A[i][j] = eM0[i+4][j]
                A[i][j+8] = 0

            # boundary conditions z=d
            # (vanishing of function for x-like components)
            if i==0 or i==1:
                A[i+4][j] = 0
                A[i+4][j+8] = eMd[i][j]
            # (vanishing of first-derivative for y-like components)
            if i==2 or i==3:
                A[i+4][j] = 0
                A[i+4][j+8] = eMd[i+4][j]

            # continuity condition z=z'
            A[i+8][j] = eMZ[i][j]
            A[i+8][j+8] = -eMZ[i][j]

            # derivative jump at z=z'
            A[i+12][j] = eMZ[i+4][j]
            A[i+12][j+8] = -eMZ[i+4][j]

    return A



# Function for solving the system and finding the particular solution
def psolution_mixedBC(icol, d, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar=1.):

    # initialize an empty vector
    y = np.zeros([16], dtype=complex)
        
    # select matrix A and vector y
    match icol:
    
        # column 1,2
        case 1 | 2:
            # non-homogeneity vector 
            y[11+icol] = hbar/(B1-D1)
            # matrix of equations
            A = Amat_34(d, Z, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
            
        # column 3,4
        case 3 | 4:
            # non-homogeneity vector 
            y[11+icol] = -hbar/(B1+D1)
            # matrix of equations
            A = Amat_34(d, Z, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    return np.linalg.solve(A, y)



# Function evaluating the Green's function (Z=z', d=thickness)
def GMTI_mixedBC(d, z, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar = 1.):

    # empty matrix for Green's function
    G = np.empty([4, 4], dtype='complex')

    # general solution M_ij at z
    eMz = expMz(z=z, kx=kx, ky=ky, L=L, w=w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # loop over columns (sigma',lambda')
    for icol in range(4):
        
        # particular solutions c_j
        cj = psolution_mixedBC(icol=icol+1, d=d, Z=Z, kx=kx, ky=ky, L=L, w=w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
        
        # select left or right depending on z,z'
        cj = cj[:8] if z<Z else cj[8:]

        # loop over the 4 sigma,lambda states i=1,2,3,4
        for irow in range(4):
            
            # evaluate Green's function components 
            G[irow][icol] = np.sum( [eMz[irow][j]*cj[j] for j in range(8)] )
        
    return G





