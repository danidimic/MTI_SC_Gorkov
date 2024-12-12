import numpy as np



#####################################################################
######################### IN-GAP SOLUTION ###########################
#####################################################################



# function which computes the normal GF in the SC for energy below the gap
def GFnormal(z, Z, kx, ky, mu, Delta, w, t = 1., hbar = 1.):
    
    # energy e0
    e0 = t*(kx**2+ky**2)-mu
    # frequency w0
    w0 = np.sqrt( complex((hbar*w)**2 - np.abs(Delta)**2) )
    
    # Omega plus/minus
    Wp = hbar*w+w0; Wm = hbar*w-w0
    # poles
    kp = 1j*np.sqrt(1./t*(e0+w0)); km = 1j*np.sqrt(1./t*(e0-w0))

    return 1j*hbar/(4*t*w0) * ( 1./kp*Wm * np.exp(1j*kp*np.abs(z-Z)) - 1./km*Wp * np.exp(1j*km*np.abs(z-Z)))



# function which computes the anomalous GF in the SC for energy below the gap
def GFanomalous(z, Z, kx, ky, mu, Delta, w, t = 1., hbar = 1.):
    
    # energy e0
    e0 = t*(kx**2+ky**2)-mu
    # frequency w0
    w0 = np.sqrt( complex((hbar*w)**2 - np.abs(Delta)**2) )
    # poles
    kp = 1j*np.sqrt(1./t*(e0+w0)); km = 1j*np.sqrt(1./t*(e0-w0))

    return -hbar*np.conj(Delta)/(4*t*w0) * ( 1./kp*np.exp(1j*kp*np.abs(z-Z)) - 1./km*np.exp(1j*km*np.abs(z-Z)) )    




