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
####################### NORMAL B.C. SOLUTION ########################
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
def psolution_normalBC(icol, d, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar=1.):

    # vetcor y 
    y = Yvec(icol, B1=B1, D1=D1, hbar=hbar)
    
    # matrix A
    A = Amat(d, Z, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    return np.linalg.solve(A, y)



# Function evaluating the Green's function (Z=z', d=thickness)
def GMTI_normalBC(d, z, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar = 1.):

    # empty matrix for Green's function
    G = np.empty([4, 4], dtype='complex')

    # general solution M_ij at z
    eMz = expMz(z=z, kx=kx, ky=ky, L=L, w=w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # loop over columns (sigma',lambda')
    for icol in range(4):
        
        # particular solutions c_j
        cj = psolution_normalBC(icol=icol+1, d=d, Z=Z, kx=kx, ky=ky, L=L, w=w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
        
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
##################### MODIFIED B.C. SOLUTION ########################
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
def psolution_modifiedBC(icol, d, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar=1.):

    # initialize an empty vector
    y = np.zeros([16], dtype=complex)
        
    # select matrix A and vector y
    match icol:
    
        # column 1,2
        case 1 | 2:
            # non-homogeneity vector 
            y[11+icol] = hbar/(B1-D1)
            # matrix of equations
            A = Amat_12(d, Z, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
            
        # column 3,4
        case 3 | 4:
            # non-homogeneity vector 
            y[11+icol] = -hbar/(B1+D1)
            # matrix of equations
            A = Amat_34(d, Z, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    return np.linalg.solve(A, y)



# Function evaluating the Green's function (Z=z', d=thickness)
def GMTI_modifiedBC(d, z, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar = 1.):

    # empty matrix for Green's function
    G = np.empty([4, 4], dtype='complex')

    # general solution M_ij at z
    eMz = expMz(z=z, kx=kx, ky=ky, L=L, w=w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # loop over columns (sigma',lambda')
    for icol in range(4):
        
        # particular solutions c_j
        cj = psolution_modifiedBC(icol=icol+1, d=d, Z=Z, kx=kx, ky=ky, L=L, w=w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
        
        # select left or right depending on z,z'
        cj = cj[:8] if z<Z else cj[8:]

        # loop over the 4 sigma,lambda states i=1,2,3,4
        for irow in range(4):
            
            # evaluate Green's function components 
            G[irow][icol] = np.sum( [eMz[irow][j]*cj[j] for j in range(8)] )
        
    return G






#####################################################################
####################### DISCRETIZED SOLUTION ########################
#####################################################################


# Function defining the Matrices A,B,C
def ABCmat(kx, ky, L, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

    k = np.sqrt(kx**2+ky**2); e0 = C + D2*k**2; m0 = M - B2*k**2

    #matrix A
    Amat = B1*np.kron(lambdaZ, sigma0) - D1*np.kron(lambda0, sigma0)

    # matrix B
    Bmat = -1j*A1*np.kron(lambdaX, sigmaZ)

    # matrix C
    Cmat = e0*np.kron(lambda0, sigma0) + L*np.kron(lambda0, sigmaZ) + A2*(kx*np.kron(lambdaX, sigmaX) + ky*np.kron(lambdaX, sigmaY)) + m0*np.kron(lambdaZ, sigma0)

    return [Amat,Bmat,Cmat]


# Function building the 3D MTI Hamiltonian from A,B,C
def hMTI(kx, ky, kz, L, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

    # build A,B,C matrices
    [Amat, Bmat, Cmat] = ABCmat(kx, ky, L, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2)

    return -np.power(kz,2)*Amat + 1j*kz*Bmat + Cmat
    
    
# Build the tigh-binding hamiltonian (Nlat=lattice points, dZ=lattice spacing)
def TBham(Nlat, dZ, kx, ky, L, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

    # build A,B,C matrices
    [Amat, Bmat, Cmat] = ABCmat(kx, ky, L, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2)

    # on-site energy
    onsite = Cmat - 2./np.power(dZ,2)*Amat;

    # hopping energy
    hopping_plus = 1./np.power(dZ,2)*Amat + 1./(2.*dZ)*Bmat

    # hopping minus
    hopping_minus = 1./np.power(dZ,2)*Amat - 1./(2.*dZ)*Bmat

    # define a zero tight-binding matrix
    TBmat = [ [None for _ in range(Nlat) ] for _ in range(Nlat)]
		
	# populate the tight-binding matrix
    for i in range(Nlat):
		
        # onsite diagonal energy 
        TBmat[i][i] = onsite
        # hopping energy (n+1)
        if i+1 < Nlat: TBmat[i][i+1] = hopping_plus
        # hopping energy (n-1)
        if i-1 > -1: TBmat[i][i-1] = hopping_minus

    return bmat(TBmat)


# Compute energy and wavefunctions in the MTI slab
def eigenstates(Nlat, dZ, kx, ky, L, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

    # thickness 
    d = (Nlat-1)*dZ
    # lattice
    lattice = np.linspace(0., d, num=Nlat)
    
    # build the tight-binding matrix
    tb = TBham(Nlat, dZ, kx, ky, L, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2)

    # solve the tight-binding problem
    if ishermitian(tb.toarray()) is True:
        egval, egvec = eigh(tb.toarray())
    
    # indices sorted by absolute value
    idx_sort = np.argsort(np.abs(egval)) 
    # reorder egval 
    egval = egval[idx_sort]
    # reorder egvec 
    egvec = [egvec[:, idx] for idx in idx_sort]

    # wavefunctions as spinors
    spinors = np.array([[ egv[4*ilat:4*ilat+4] for ilat in range(Nlat)] for egv in egvec])
    
    # loop over eigenstates 
    for iegv in range(len(egval)):

        # probability density for each lattice point
        pd = np.array([np.vdot(s, s) for s in spinors[iegv]]).real
        # normalization
        norm = np.trapz(pd, x=lattice)
        # normalize spinors over lattice
        spinors[iegv] = np.divide(spinors[iegv], np.sqrt(norm))

    return lattice, egval, spinors


# Function evaluating the Green's function from wavefunctions (Z=z', d=thickness)
def GMTI_discretized(Nstates, egval, spinors, z, Z, kx, ky, L, w, eta = 1E-9, EF = 0., hbar = 1.):
    
    # empty matrix for Green's function
    gf = np.zeros([4, 4], dtype='complex')

    # loop over columns of GF
    for icol in range(4):
        # loop over rows of GF
        for irow in range(4):

            # sum over Nstates
            for istate in range(Nstates):
                # psi
                psi = spinors[istate][z][irow]
                # psi star
                psistar = np.conjugate(spinors[istate][Z][icol])
                # energy
                en = egval[istate]
                # perform sum over states
                gf[irow][icol] += psi*psistar/(w-en/hbar+1j*eta*np.sign(en-EF))

    return gf





#####################################################################
###################### ANALYTICAL SOLUTION ##########################
#####################################################################


import numpy as np

from GreenFunctions_MTI import GMTI_normalBC, GMTI_modifiedBC
from GreenFunctions_SC import GSC, GSC_matrix, FSC, FSC_matrix

# Hamiltonian parameters
params=dict(C0 = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M0 = 0.28, B1 = 10, B2 = 56.6)


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
def coefficients(d, irow, icol, lambda1, lambda2, gamma1, gamma2, alpha, beta):

    # switch over non-zero components
    match (irow,icol):
            
        # (lambda, lambda') = (+,+)
        case (0,0) | (1,1):
            # coefficient alpha_1
            a1 = alpha*gamma2*csch(lambda1*d)/(gamma2*lambda1-gamma1*lambda2)
            # coefficient alpha_2
            a2 = -alpha*gamma1*csch(lambda2*d)/(gamma2*lambda1-gamma1*lambda2)
        
        # (lambda, lambda') = (-,+)
        case (2,0) | (3,1): 
            # coefficient alpha_1
            a1 = alpha*gamma1*gamma2*csch(lambda1*d)/(gamma2*lambda1-gamma1*lambda2)
            # coefficient alpha_2
            a2 = -alpha*gamma1*gamma2*csch(lambda2*d)/(gamma2*lambda1-gamma1*lambda2)

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
            a1, a2 = coefficients(d, irow, icol, l1, l2, g1, g2, alpha, beta)

            # diagonal components
            if irow == icol:

                if z <= Z:
                    gij = a1*np.cosh(l1*z)*np.cosh(l1*(Z-d)) + a2*np.cosh(l2*z)*np.cosh(l2*(Z-d))
                else:
                    gij = a1*np.cosh(l1*Z)*np.cosh(l1*(z-d)) + a2*np.cosh(l2*Z)*np.cosh(l2*(z-d))

            # off-diagonal components
            if irow != icol:

                if z <= Z:
                    gij = a1*np.sinh(l1*z)*np.cosh(l1*(Z-d)) + a2*np.sinh(l2*z)*np.cosh(l2*(Z-d))
                else:
                    gij = a1*np.cosh(l1*Z)*np.sinh(l1*(z-d)) + a2*np.cosh(l2*Z)*np.sinh(l2*(z-d))
            

            # fill the matrix for the Green's function
            GF[irow][icol] = gij

    return GF


