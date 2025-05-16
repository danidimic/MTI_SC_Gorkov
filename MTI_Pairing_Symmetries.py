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



# function for change of basis in MTI Green's function
# from parity block-basis to spin-block basis and reverse
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
	

	


