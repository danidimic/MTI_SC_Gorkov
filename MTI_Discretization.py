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
################### DIRICHLET B.C. TIGHT-BINDING ####################
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
def TBham_Dirichlet(Nlat, dZ, kx, ky, L, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

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
def eigenstates_Dirichlet(Nlat, dZ, kx, ky, L, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

    # thickness 
    d = (Nlat-1)*dZ
    # lattice
    lattice = np.linspace(0., d, num=Nlat)
    
    # build the tight-binding matrix
    tb = TBham_Dirichlet(Nlat, dZ, kx, ky, L, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2)

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




#####################################################################
################## FULL HAMILTONIAN TIGHT-BINDING ###################
#####################################################################


# Function defining the Matrices a,b,c for the MTI
def abcMTI(kx, ky, L, C0 = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6):

    k = np.sqrt(kx**2+ky**2); e0 = C0 + D2*k**2; m0 = M - B2*k**2

    #matrix A
    amat = B1*np.kron(lambdaZ, sigma0) - D1*np.kron(lambda0, sigma0)

    # matrix B
    bmat = -1j*A1*np.kron(lambdaX, sigmaZ)

    # matrix C
    cmat = e0*np.kron(lambda0, sigma0) + L*np.kron(lambda0, sigmaZ) + A2*( kx*np.kron(lambdaX, sigmaX) + ky*np.kron(lambdaX, sigmaY) ) + m0*np.kron(lambdaZ, sigma0)

    return [amat, bmat, cmat]



# Function defining the Matrices alpha and gamma for the SC
def abcSC(kx, ky, mu, Delta, t=1.):
        
    E0 = t*(kx**2+ky**2)

    # make Delta complex if real
    if isinstance(Delta, float):
        Delta = complex(Delta)

    # matrix alpha
    alpha = -t*np.kron(tauZ, sigma0)

    # matrix beta
    beta = np.zeros((4,4))

    # matrix gamma 
    gamma = (E0-mu)*np.kron(tauZ, sigma0) - Delta.real*np.kron(tauY, sigmaY) - Delta.imag*np.kron(tauX, sigmaY)

    return [alpha, beta, gamma]



# Function defining the Matrices A,B,C in the different cases
def ABCmatrices(kx, ky, mu, Delta, L, dZ, G, C0 = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1.):

    # MTI a,b,c matrices
    a,b,c = abcMTI(kx=kx, ky=ky, L=L, C0=C0, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2)

    # SC alpha and gamma matrices
    alpha, beta, gamma = abcSC(kx=kx, ky=ky, mu=mu, Delta=Delta, t=1.) 

    # Gamma matrix in chosen basis
    Gamma = np.block([G, np.zeros((4,2))]) 

    # matrices for the full MTI-SC heterostructure 
    A = np.block([[a, np.zeros((4,4))],[np.zeros((4,4)), alpha]])
    B = np.block([[b, np.zeros((4,4))],[np.zeros((4,4)), beta]])
    C = np.block([[c, np.zeros((4,4))],[np.zeros((4,4)), gamma]])

    # matrices for the SC part only 
    Asc = np.block([[np.zeros((4,4)), np.zeros((4,4))],[np.zeros((4,4)), alpha]])
    Bsc = np.block([[np.zeros((4,4)), np.zeros((4,4))],[np.zeros((4,4)), beta]])
    Csc = np.block([[np.zeros((4,4)), np.zeros((4,4))],[np.zeros((4,4)), gamma]])

   
    # matrix for tunneling interface
    Tun = np.block([[np.zeros((4,4)), Gamma],[np.conj(np.transpose(Gamma)), np.zeros((4,4))]])
    
    return A, B, C, Asc, Csc, Tun



# Build the tigh-binding hamiltonian 
# (Nlat=lattice points, dZ=lattice spacing, zR = right interface, zL = left interface, z0 = tunneling interface)
def TBham_FullHamiltonian(kx, ky, mu, Delta, L, Nlat, dZ, zL, zR, z0, G, C0 = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1.):

    # get A,B,C matrices
    A,B,C, Asc,Csc, Tun = ABCmatrices(kx=kx, ky=ky, mu=mu, Delta=Delta, L=L, dZ=dZ, G=G, C0=C0, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, t=t)

    # define a zero tight-binding matrix
    TBmat = [ [None for _ in range(Nlat) ] for _ in range(Nlat)]
		
	# loop over lattice sites
    for ilat in range(Nlat):
        
        # SC only 
        if ilat < zL or ilat > zR:
            
            # on-site energy
            onsite = Csc - 2./np.power(dZ,2)*Asc;
            
            # hopping plus
            hopping_plus = 1./np.power(dZ,2)*Asc
            
            # hopping minus
            hopping_minus = 1./np.power(dZ,2)*Asc

        
        # MTI-SC heterostructure 
        if ilat >= zL and ilat <= zR:
            
            # on-site energy
            onsite = C - 2./np.power(dZ,2)*A;  
            
            # hopping plus
            hopping_plus = 1./np.power(dZ,2)*A + 1./(2.*dZ)*B if ilat != zR else 1./np.power(dZ,2)*Asc
            
            # hopping minus
            hopping_minus = 1./np.power(dZ,2)*A - 1./(2.*dZ)*B if ilat != zL else 1./np.power(dZ,2)*Asc

        
        # tunneling at site z0
        if ilat == z0:
            onsite += Tun
        
        # onsite diagonal energy 
        TBmat[ilat][ilat] = onsite
        # hopping energy (n+1)
        if ilat+1 < Nlat: TBmat[ilat][ilat+1] = hopping_plus
        # hopping energy (n-1)
        if ilat-1 > -1: TBmat[ilat][ilat-1] = hopping_minus

    return bmat(TBmat)



# Compute energy and wavefunctions in the full system
def eigenstates_FullHamiltonian(kx, ky, mu, Delta, L, Nlat, dZ, zL, zR, z0, G, C0 = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1.):

    # number of components
    Nc = 8
    # thickness 
    d = (Nlat-1)*dZ
    # lattice
    lattice = np.linspace(0., d, num=Nlat) 
    
    # build the tight-binding matrix
    tb = TBham_FullHamiltonian(kx, ky, mu, Delta, L, Nlat, dZ, zL, zR, z0, G, C0=C0, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, t=t)

    # get rank of TB matrix
    rank = matrix_rank(tb.toarray())
    
    # solve the tight-binding problem
    if ishermitian(tb.toarray()) is True: 
        # compute eigenstates
        egval, egvec = eigh(tb.toarray())
    
    # indices sorted by absolute value
    idx_sort = np.argsort(np.abs(egval)) 
    # reorder egval 
    egval = egval[idx_sort]
    # reorder egvec 
    egvec = [egvec[:, idx] for idx in idx_sort]

    return rank, lattice, np.array(egval), np.array(egvec)



# get the spinors corresponding to e/h in MTI/SC
def getSpinors(egvec, Nlat, type='all'):

    # number of components
    Nc = 8
    
    match type:
        # electrons in MTI
        case 'eMTI':
            spinors = np.array([[ egv[Nc*ilat+0:Nc*ilat+4] for ilat in range(Nlat)] for egv in egvec])

        # electrons in SC
        case 'eSC':
            spinors = np.array([[ egv[Nc*ilat+4:Nc*ilat+6] for ilat in range(Nlat)] for egv in egvec])
        # holes in SC
        case 'hSC':
            spinors = np.array([[ egv[Nc*ilat+6:Nc*ilat+8] for ilat in range(Nlat)] for egv in egvec])
        # full spinor in heterostructure
        case _:
            spinors = np.array([[ egv[Nc*ilat:Nc*ilat+Nc] for ilat in range(Nlat)] for egv in egvec])

    return spinors




# get the spinors corresponding to e/h in MTI/SC
def getComponents(egvec, Nlat, type='all'):

    # number of components
    Nc = 8; components = []
    
    match type:
        # electrons in MTI
        case 'eMTI':
            for icomp in range(4):
                # get the 4 MTI components
                components.append( np.array([[ egv[Nc*ilat+icomp] for ilat in range(Nlat)] for egv in egvec]) )
        # electrons in SC
        case 'eSC':
            for icomp in range(4,6):
                # get the 2 SC electron components
                components.append( np.array([[ egv[Nc*ilat+icomp] for ilat in range(Nlat)] for egv in egvec]) )
        # holes in SC
        case 'hSC':
            for icomp in range(6,8):
                # get the 2 SC hole components
                components.append( np.array([[ egv[Nc*ilat+icomp] for ilat in range(Nlat)] for egv in egvec]) )

    return components




#####################################################################
################### DISCRETIZED GREEN'S FUNCTION ####################
#####################################################################


# Function evaluating the Green's function from wavefunctions (Z=z', d=thickness)
def GMTI_discretized(Nstates, egval, spinors, z, Z, w, n0 = 0, eta = 1E-9, EF = 0., hbar = 1.):
    
    # empty matrix for Green's function
    gf = np.zeros([4, 4], dtype='complex')

    # loop over columns of GF
    for icol in range(4):
        # loop over rows of GF
        for irow in range(4):

            # sum over Nstates
            for istate in range(Nstates):
                # psi
                psi = spinors[n0+istate][z][irow]
                # psi star
                psistar = np.conjugate(spinors[n0+istate][Z][icol])
                # energy
                en = egval[n0+istate]
                # perform sum over states
                gf[irow][icol] += psi*psistar/(w-en/hbar+1j*eta*np.sign(en-EF))
                

    return gf





