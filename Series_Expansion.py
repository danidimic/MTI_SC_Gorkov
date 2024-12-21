import numpy as np

from GreenFunctions_MTI import GFexact
from GreenFunctions_SC import GFnormal, GFnormalMat, GFanomalous, GFanomalousMat

# Hamiltonian parameters
params=dict(C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6)



#####################################################################
##################### SECOND-ORDER CORRECTION #######################
#####################################################################


# Second order correction to the normal GF
def G_second_order(d, z, Z, kx, ky, L, mu, Delta, w, Gamma, lT, Nint=100, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, t = 1., hbar = 1.):

    # function for tunneling amplitude
    ftun = lambda v1,v2 : spatial_tunneling(v1, d=d, lT=lT)*spatial_tunneling(v2, d=d, lT=lT)
    
    # function for G_MTI(z,v)
    GMTIz = lambda v : GFexact(d=d, z=z, Z=v, kx=kx, ky=ky, L=L, w=w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # function for G_MTI(v',z')
    GMTIZ = lambda v : GFexact(d=d, z=v, Z=Z, kx=kx, ky=ky, L=L, w=w, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
    
    # function for G_SC
    GSC = lambda v1,v2 : GFnormalMat(z=-v1, Z=v2, kx=kx, ky=ky, mu=mu, Delta=Delta, w=w, t=t, hbar=hbar)

    # define integrand function 
    fintegrand = lambda v1, v2: (ftun(v1,v2) * GMTIz(v1) @ Gamma @ GSC(v1,v2) @ Gamma.H @ GMTIZ(v2))

    # compute double integral
    return quad_vec(lambda v1 : quad_vec(lambda v2: fintegrand(v1, v2), 0., d)[0], 0., d)[0]



