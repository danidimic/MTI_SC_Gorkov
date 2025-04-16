import numpy as np

from MTI_Differential_Equation import GMTI_NeumannBC
from SC_Gorkov_Equation import GSC_matrix, FSC_matrix


# Hamiltonian parameters
params=dict(C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6)





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
    GSC = GSC_matrix(z=0., kx=0., ky=0., mu=mu, Delta=Delta, w=omega, t=t, hbar=hbar)
    
    return (GMTIz @ Gamma @ GSC @ Gamma.H @ GMTIZ).A



# Function evaluating the F2-MTI using the G0 with Neumann BC
def FMTI2_NeumannBC(d, z, Z, z0, kx, ky, L, mu, Delta, omega, Gamma, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar=1., t=1.):

    # GMTI(z0,z)
    GMTIz = GMTI_NeumannBC(d=d, z=z0, Z=z, kx=kx, ky=ky, L=L, w=-omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # GMTI(z0,z')
    GMTIZ = GMTI_NeumannBC(d=d, z=z0, Z=Z, kx=kx, ky=ky, L=L, w=omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # FSC(0)
    FSC = FSC_matrix(z=0., kx=0., ky=0., mu=mu, Delta=Delta, w=omega, t=t, hbar=hbar)
    
    return (np.transpose(GMTIz) @ np.conj(Gamma) @ FSC @ Gamma.H @ GMTIZ).A






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
    
    
