import numpy as np
from scipy.integrate import quad_vec

from GreenFunctions_MTI import GFexact
from GreenFunctions_SC import GFnormal, GFnormalMat, GFanomalous, GFanomalousMat

# Hamiltonian parameters
params=dict(C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6)




##################################
########## TUNNELING #############
##################################


# function for tunneling amplitude 
def spatial_tunneling(z, d, lT):

    # Gaussian tunneling amplitude
    return np.exp( -(z-d)**2/(2*lT**2) )




##################################
############ NORMAL ##############
##################################


# define the integrand functions
def G2_integrand(z1, z2, d, z, Z, kx, ky, L, mu, Delta, omega, Gamma, lT, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1., hbar = 1.):

    # spatial functions
    fz1 = spatial_tunneling(z1, d=d, lT=lT); fz2 = spatial_tunneling(z2, d=d, lT=lT)

    # GMTI(z,w)
    GMTIz = GFexact(d=d, z=z, Z=z1, kx=kx, ky=ky, L=L, w=omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # GMTI(v,z')
    GMTIZ = GFexact(d=d, z=z2, Z=Z, kx=kx, ky=ky, L=L, w=omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # GSC(v-w)
    GSC = GFnormalMat(z=z2-z1, kx=kx, ky=ky, mu=mu, Delta=Delta, w=omega, t=t, hbar=hbar)
    
    return fz1*fz2 * (GMTIz @ Gamma @ GSC @ Gamma.H @ GMTIZ).A



# Second order correction to the normal GF
def G2_montecarlo(d, z, Z, kx, ky, L, mu, Delta, omega, Gamma, lT, Nsamples = 10000, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1., hbar = 1.):

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
        fsum += G2_integrand(z1, z2, d=d, z=z, Z=Z, kx=kx, ky=ky, L=L, mu=mu, Delta=Delta, omega=omega, Gamma=Gamma, lT=lT, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)/qi(z1,z2)

    return pow(d,2)*fsum/float(Nsamples)



# Second order correction to the normal GF with scipy quad_vec
def G2_quad_vec(d, z, Z, kx, ky, L, mu, Delta, omega, Gamma, lT, Nint=100, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1., hbar = 1.):

    # define integrand function 
    fintegrand = lambda z1, z2: G2_integrand(z1, z2, d=d, z=z, Z=Z, kx=kx, ky=ky, L=L, mu=mu, Delta=Delta, omega=omega, Gamma=Gamma, lT=lT, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # compute double integral
    return quad_vec(lambda z1 : quad_vec(lambda z2: fintegrand(z1, z2), 0., d)[0], 0., d)[0]




##################################
########### ANOMALOUS ############
##################################



# define the integrand functions
def F2_integrand(z1, z2, d, z, Z, kx, ky, L, mu, Delta, omega, Gamma, lT, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1., hbar = 1.):

    # spatial functions
    fz1 = spatial_tunneling(z1, d=d, lT=lT); fz2 = spatial_tunneling(z2, d=d, lT=lT)

    # GMTI(z,z1; -omega)
    GMTIz = np.transpose(GFexact(d=d, z=z, Z=z1, kx=kx, ky=ky, L=L, w=-omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar))
    
    # GMTI(z2,z'; omega)
    GMTIZ = GFexact(d=d, z=z2, Z=Z, kx=kx, ky=ky, L=L, w=omega, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # FSC(z2-z1; omega)
    FSC = GFanomalousMat(z=z2-z1, kx=kx, ky=ky, mu=mu, Delta=Delta, w=omega, t=t, hbar=hbar)
    
    return fz1*fz2 * (GMTIz @ np.conjugate(Gamma) @ FSC @ Gamma.H @ GMTIZ).A



# Second order correction to the normal GF
def F2_montecarlo(d, z, Z, kx, ky, L, mu, Delta, omega, Gamma, lT, Nsamples = 10000, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1., hbar = 1.):

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
        fsum += F2_integrand(z1, z2, d=d, z=z, Z=Z, kx=kx, ky=ky, L=L, mu=mu, Delta=Delta, omega=omega, Gamma=Gamma, lT=lT, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)/qi(z1,z2)

    return pow(d,2)*fsum/float(Nsamples)
    
    

# Second order correction to the normal GF with scipy quad_vec
def F2_quad_vec(d, z, Z, kx, ky, L, mu, Delta, omega, Gamma, lT, Nint=100, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1., hbar = 1.):

    # define integrand function 
    fintegrand = lambda z1, z2: F2_integrand(z1, z2, d=d, z=z, Z=Z, kx=kx, ky=ky, L=L, mu=mu, Delta=Delta, omega=omega, Gamma=Gamma, lT=lT, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)

    # compute double integral
    return quad_vec(lambda z1 : quad_vec(lambda z2: fintegrand(z1, z2), 0., d)[0], 0., d)[0]

