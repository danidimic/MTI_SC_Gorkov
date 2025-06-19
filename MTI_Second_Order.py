import numpy as np

from MTI_Differential_Equation import GMTI_NeumannBC
from SC_Gorkov_Equation import GSC_matrix, FSC_matrix


# Hamiltonian parameters
params=dict(C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6)


# function for change of basis in MTI Green's function
# from parity block-basis to spin-block basis
def Change_Basis(gf):

    g_new = [[gf[0][0], gf[0][2], gf[0][1], gf[0][3]],
             [gf[2][0], gf[2][2], gf[2][1], gf[2][3]],
             [gf[1][0], gf[1][2], gf[1][1], gf[1][3]],
             [gf[3][0], gf[3][2], gf[3][1], gf[3][3]]]
    
    return np.array(g_new)


# function that extract the four 2x2 blocks that make up the pairing matrix
def Block_Decomposition(f):

	f00 = f[0:2, 0:2]
	f01 = f[0:2, 2:4]
	f10 = f[2:4, 0:2]
	f11 = f[2:4, 2:4]
	
	return f00, f01, f10, f11



# function that exchange the off-diagonal blocks
def Block_Reverse(f):

	# extract the 4 blocks 
	f00, f01, f10, f11 = Block_Decomposition(f)
	
	return np.block([[f00, f10],[f01, f11]])



#######################################################
############# SOLUTION WITH NEUMANN B.C. ##############
#######################################################



# Function evaluating the G2-MTI using the G0 with Neumann BC
def GMTI2_NeumannBC(d, z, Z, z0, kx, ky, L, mu, Delta, omega, Gamma, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar=1., t=1.):

    # GMTI(z,z0)
    GMTIz = GMTI_NeumannBC(d=d, z=z, Z=z0, kx=kx, ky=ky, L=L, w=omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # GMTI(z0,z')
    GMTIZ = GMTI_NeumannBC(d=d, z=z0, Z=Z, kx=kx, ky=ky, L=L, w=omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # GSC(0)
    GSC = GSC_matrix(z=0., kx=kx, ky=ky, mu=mu, Delta=Delta, w=omega, t=t, hbar=hbar)
    
    return (GMTIz @ Gamma @ GSC @ Gamma.H @ GMTIZ).A



# Function evaluating the F2-MTI using the G0 with Neumann BC
def FMTI2_NeumannBC(d, z, Z, z0, kx, ky, L, mu, Delta, omega, Gamma, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar=1., t=1.):

    # GMTI(z0,z) hole part
    GMTIz = GMTI_NeumannBC(d=d, z=z0, Z=z, kx=-kx, ky=-ky, L=L, w=-omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # GMTI(z0,z') electron part
    GMTIZ = GMTI_NeumannBC(d=d, z=z0, Z=Z, kx=kx, ky=ky, L=L, w=omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # FSC(0) Cooper pair part
    FSC = FSC_matrix(z=0., kx=kx, ky=ky, mu=mu, Delta=Delta, w=omega, t=t, hbar=hbar)
    
    return (np.transpose(GMTIz) @ np.conj(Gamma) @ FSC @ Gamma.H @ GMTIZ).A



#  Function that evaluates the F2-MTI in relative coordinates Z0 and zrel
def FMTI2_Relative_Coordinates(d, Z0, zrel, kx, ky, L, mu, Delta, omega, Gamma, z0=0, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar=1., t=1.):

    # separate coordinates z1, z2
    z1 = Z0 + 1/2*zrel; z2 = Z0 - 1/2*zrel

    return FMTI2_NeumannBC(d=d, z=z1, Z=z2, z0=z0, kx=kx, ky=ky, L=L, mu=mu, Delta=Delta, omega=omega, Gamma=Gamma, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)





#####################################################################
######################### WIGNER TRANSFORM ##########################
#####################################################################


# Evaluate discrete Fourier transform in relative coordinates
# kZ0=center of mass of Cooper pair, Nzrel=number of discrete lattice points for z relative
def FMTI2_Wigner_Transform(d, Z0, k, kx, ky, L, mu, Delta, omega, Gamma, N=199, z0=0, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar=1., t=1.):
    
	# boundaries for |z1-z2|
	z_min = max(-2*Z0, 2*(Z0-d)); z_max = min(2*Z0, 2*(d-Z0) )
	
	# discrete lattice for relative coordinates
	if z_min != z_max:
		zrelative = np.linspace(z_min, z_max, N)
	else:
		zrelative = np.array([0])
    
	# zero matrix for Wigner transform
	F2_k = np.zeros((4,4), dtype='complex')
    
	# loop over relative coordinate z
	for z in zrelative:
    
		# evaluate F2 in relative coordinates
		F2_rc = FMTI2_Relative_Coordinates(d=d, Z0=Z0, zrel=z, kx=kx, ky=ky,  L=L, mu=mu, Delta=Delta, omega=omega, Gamma=Gamma, z0=z0, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar, t=t)

		# exponential factor
		e = np.exp(-1j*k*z)
		
		# loop over components
		for i in range(4):
			for j in range(4):
        		
				# sum to build Wigner transform
				F2_k[i,j] += e * F2_rc[i,j]
	
	return F2_k





#######################################################
############### MONTE-CARLO SOLUTION ##################
#######################################################



# function for tunneling amplitude 
def spatial_tunneling(z, d, lT):

    # Gaussian tunneling amplitude
    return np.exp( -(z-d)**2/(2*lT**2) )



###### Normal Green's Function #########

# define the integrand functions
def GMTI2_integrand(z1, z2, d, z, Z, kx, ky, L, mu, Delta, omega, Gamma, lT, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1., hbar = 1.):

    # spatial functions
    fz1 = spatial_tunneling(z1, d=d, lT=lT); fz2 = spatial_tunneling(z2, d=d, lT=lT)

    # GMTI(z,w)
    GMTIz = GMTI_normalBC(d=d, z=z, Z=z1, kx=kx, ky=ky, L=L, w=omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # GMTI(v,z')
    GMTIZ = GMTI_normalBC(d=d, z=z2, Z=Z, kx=kx, ky=ky, L=L, w=omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # GSC(v-w)
    GSC = GSC_matrix(z=z2-z1, kx=kx, ky=ky, mu=mu, Delta=Delta, w=omega, t=t, hbar=hbar)
    
    return fz1*fz2 * (GMTIz @ Gamma @ GSC @ Gamma.H @ GMTIZ).A



# Second order correction to the normal GF
def GMTI2_montecarlo(d, z, Z, kx, ky, L, mu, Delta, omega, Gamma, lT, Nsamples = 10000, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1., hbar = 1.):

    # normal sampling
    #z1_samples = np.random.uniform(0., d, Nsamples); z2_samples = np.random.uniform(0., d, Nsamples)

    # importance sampling
    z1 = np.random.normal(loc=d, scale=lT, size=Nsamples); z2 = np.random.normal(loc=d, scale=lT, size=Nsamples);
    # restrict to values in [0,d]
    z1_samples = np.where(z1 > d, 2*d-z1, z1); z2_samples = np.where(z2 > d, 2*d-z2, z2)
    
    # importance sampling function 
    qi = lambda x1,x2: spatial_tunneling(x1,d,lT)*spatial_tunneling(x2,d,lT)
    
    # matrix for mean values
    fsum = np.zeros((4,4), dtype='complex')

    # loop over samples
    for (z1,z2) in zip(z1_samples, z2_samples):

        # compute mean of fwv function
        fsum += GMTI2_integrand(z1, z2, d=d, z=z, Z=Z, kx=kx, ky=ky, L=L, mu=mu, Delta=Delta, omega=omega, Gamma=Gamma, lT=lT, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)/qi(z1,z2)

    return pow(d,2)*fsum/float(Nsamples)



###### Anomalous Green's Function #########

# define the integrand functions
def FMTI2_integrand(z1, z2, d, z, Z, kx, ky, L, mu, Delta, omega, Gamma, lT, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1., hbar = 1.):

    # spatial functions
    fz1 = spatial_tunneling(z1, d=d, lT=lT); fz2 = spatial_tunneling(z2, d=d, lT=lT)

    # GMTI(z,z1; -omega)
    GMTIz = np.transpose(GMTI_normalBC(d=d, z=z, Z=z1, kx=kx, ky=ky, L=L, w=-omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar))
    
    # GMTI(z2,z'; omega)
    GMTIZ = GMTI_normalBC(d=d, z=z2, Z=Z, kx=kx, ky=ky, L=L, w=omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # FSC(z2-z1; omega)
    FSC = FSC_matrix(z=z2-z1, kx=kx, ky=ky, mu=mu, Delta=Delta, w=omega, t=t, hbar=hbar)
    
    return fz1*fz2 * (GMTIz @ np.conjugate(Gamma) @ FSC @ Gamma.H @ GMTIZ).A



# Second order correction to the normal GF
def FMTI2_montecarlo(d, z, Z, kx, ky, L, mu, Delta, omega, Gamma, lT, Nsamples = 10000, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1., hbar = 1.):

    # z1,z2 random samples
    #z1_samples = np.random.uniform(0., d, Nsamples); z2_samples = np.random.uniform(0., d, Nsamples)

    # importance sampling
    z1 = np.random.normal(loc=d, scale=lT, size=Nsamples); z2 = np.random.normal(loc=d, scale=lT, size=Nsamples);
    # restrict to values in [0,d]
    z1_samples = np.where(z1 > d, 2*d-z1, z1); z2_samples = np.where(z2 > d, 2*d-z2, z2)
    
    # importance sampling function 
    qi = lambda x1,x2: spatial_tunneling(x1,d,lT)*spatial_tunneling(x2,d,lT)
    
    # matrix for mean values
    fsum = np.zeros((4,4), dtype='complex')

    # loop over samples
    for (z1,z2) in zip(z1_samples, z2_samples):
        
        # compute mean of fwv function
        fsum += FMTI2_integrand(z1, z2, d=d, z=z, Z=Z, kx=kx, ky=ky, L=L, mu=mu, Delta=Delta, omega=omega, Gamma=Gamma, lT=lT, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)/qi(z1,z2)

    return pow(d,2)*fsum/float(Nsamples)
    
    
