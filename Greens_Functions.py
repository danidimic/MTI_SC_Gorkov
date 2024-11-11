import numpy as np
from scipy.linalg import expm
from scipy.sparse import coo_matrix, bmat
from scipy.sparse.linalg import eigs, eigsh

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
########################## EXACT SOLUTION ###########################
#####################################################################


# Function defining the Matrix M
def Mmat(kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

    k = np.sqrt(kx**2+ky**2); e0 = C + D2*k**2; m0 = M - B2*k**2

    # Zero Matrix
    zero = np.zeros((4,4))

    # Unitary Matrix
    id = np.identity(4)

    # Matrix C (Constant)
    Cmat = np.array(
        [[-(e0+m0+L-1j*w)/(B1-D1), 0, 0, -A2*(kx-1j*ky)/(B1-D1)],
         [0, -(e0+m0-L-1j*w)/(B1-D1), -A2*(kx+1j*ky)/(B1-D1),0],
         [0, A2*(kx-1j*ky)/(B1+D1), (e0-m0+L-1j*w)/(B1+D1), 0],
         [A2*(kx+1j*ky)/(B1+D1), 0, 0, (e0-m0-L-1j*w)/(B1+D1)]])

    # Matrix D (first derivatives)
    Dmat = np.array(
        [[0, 0, 1j*A1/(B1-D1), 0],
         [0, 0, 0, -1j*A1/(B1-D1)],
         [-1j*A1/(B1+D1), 0, 0, 0],
         [0, 1j*A1/(B1+D1), 0, 0]])
    
    # Compone the matrix M
    return np.block([[zero, id], [Cmat, Dmat]])


# Function computing the exponetial matrix exp(Mz)
def expMz(z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

    return expm( Mmat(kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2)*z )


# Function defining the matrix A (Z=z', d=thickness)
def Amat(d, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

    # exponential at z=0
    eM0 = expMz(0, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2)
    
    # exponential at z=d
    eMd = expMz(d, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2)

    # exponential at z=d
    eMZ = expMz(Z, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2)

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
def Yvec(icol, D1 = 1.3, B1 = 10):

    # initialize an empty vector
    y = np.zeros([16], dtype=complex)

    # fill in the nonzero value
    match icol:
    
        # column 1,2
        case 1 | 2:
            y[11+icol] = -1j/(B1-D1)
            
        # column 3,4
        case 3 | 4:
            y[11+icol] = -1j/(B1+D1)

    return(y)


# Function for solving the system and finding the particular solution
def psolution(icol, d, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

    # vetcor y 
    y = Yvec(icol, B1=B1, D1=D1)
    
    # matrix A
    A = Amat(d, Z, kx, ky, L, w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2)

    return np.linalg.solve(A, y)


# Function evaluating the Green's function (Z=z', d=thickness)
def GFexact(d, z, Z, kx, ky, L, w, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

    # empty matrix for Green's function
    G = np.empty([4, 4], dtype='complex')

    # general solution M_ij at z
    eMz = expMz(z=z, kx=kx, ky=ky, L=L, w=w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2)

    # loop over columns (sigma',lambda')
    for icol in range(4):
        
        # particular solutions c_j
        cj = psolution(icol=icol+1, d=d, Z=Z, kx=kx, ky=ky, L=L, w=w)
        
        # select left or right depending on z,z'
        cj = cj[:8] if z<Z else cj[8:]

        # loop over the 4 sigma,lambda states i=1,2,3,4
        for irow in range(4):
            
            # evaluate Green's function components 
            G[irow][icol] = np.sum( [eMz[irow][j]*cj[j] for j in range(8)] )
        
    return G



    

#####################################################################
######################## NUMERICAL SOLUTION #########################
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
    hopping_plus = 1./np.power(dZ,2)*Amat + 1./(2.*np.power(dZ,2))*Bmat

    # hopping minus
    hopping_minus = np.conjugate(np.transpose(hopping_plus))

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
def eigenstates(Nlat, dZ, kx, ky, L, Neig=10, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

    # thickness 
    d = (Nlat-1)*dZ
    # lattice
    lattice = np.linspace(0., d, num=Nlat)
    
    # build the tight-binding matrix
    tb = TBham(Nlat, dZ, kx, ky, L, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2)
    
    # solve the tight-binding problem
    egval, egvec = eigsh(tb, k=Neig, sigma=0)

    # spinors for wavefunctions
    spinors = np.array([[ egvec[:, iegv][4*ilat:4*ilat+4] for ilat in range(Nlat)] for iegv in range(Neig)])

    return [lattice, egval, np.array(spinors)]


# Function evaluating the Green's function from wavefunctions (Z=z', d=thickness)
def GFapprox(Nstates, z, Z, kx, ky, L, w, egval = None, spinors = None, Nlat = None, dZ = None, eta = 1E-6, EF = 0., hbar = 1., C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

    # empty matrix for Green's function
    gf = np.zeros([4, 4], dtype='complex')

    # compute eigenstates if not provided
    if egval is None or spinors is None:
        # eigenstates
        lattice, egval, spinors = eigensates(Nlat=Nlat, dZ=dZ, kx=kx, ky=ky, L=L, Neig=Nstates)

    # check on number of states
    if Nstates <= spinors.shape[0]:

        # loop over columns of GF
        for icol in range(4):
            # loop over rows of GF
            for irow in range(4):

                # perform sum over states
                gf[irow][icol] = np.sum( [spinors[istate][z][irow]*np.conjugate(spinors[istate][Z][icol])/(w-egval[istate]/hbar+1j*eta*np.sign(egval[istate]-EF)) for istate in range(Nstates)] )

    return gf
