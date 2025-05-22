import numpy as np
from MTI_Second_Order import FMTI2_NeumannBC

from IPython.display import Math, display
from sympy import Symbol, Matrix, I, init_printing, simplify, factor_terms, trace, latex, kronecker_product, sqrt




#####################################################################
######################### WIGNER TRANSFORM ##########################
#####################################################################


# function for change of basis in MTI Green's function
# from parity block-basis to spin-block basis
def Change_Basis(gf):

    g_new = [[gf[0][0], gf[0][2], gf[0][1], gf[0][3]],
             [gf[2][0], gf[2][2], gf[2][1], gf[2][3]],
             [gf[1][0], gf[1][2], gf[1][1], gf[1][3]],
             [gf[3][0], gf[3][2], gf[3][1], gf[3][3]]]
    
    return np.array(g_new)



# Evaluate discrete Fourier transform in relative coordinates
# kZ0=center of mass of Cooper pair, Nzrel=number of discrete lattice points for z relative
def DiscreteFT(d, Z0, kx, ky, L, mu, Delta, omega, Gamma, Nzrel=200, z0=0, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar=1., t=1.):
    

    # boundaries for |z1-z2|
    max_zrel = max(d, d-Z0)
    # discrete lattice for relative coordinates
    zrelative = np.linspace(-max_zrel, max_zrel, Nzrel)    
    
    # lattice spacing for relative coordinates
    a = abs( zrelative[1]-zrelative[2] ); N = len(zrelative)

    # F2 in relative coordinates
    F2_rc = []; 
    # loop over relative coordinate z
    for z in zrelative:
    
        # separate coordinates z1, z2
        z1 = Z0 + 1/2*z; z2 = Z0 - 1/2*z
    
        # evaluate F2 as function of relative position z for fixed center of mass Z
        F2_rc.append( FMTI2_NeumannBC(d=d, z=z1, Z=z2, z0=z0, kx=kx, ky=ky, L=L, mu=mu, Delta=Delta, omega=omega, Gamma=Gamma, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar) )
    
    # array for F2 in relative coordinates (real space)
    F2_rc = np.array(F2_rc)

    # get array of k values
    k = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=a))
    # evaluate the Wigner transform
    F2_k = np.fft.fftshift(np.fft.fft(F2_rc))
    
    # change basis to the spin x orbital one
    F2_k = np.array([Change_Basis(f) for f in F2_k])
    
    return k, F2_k






def DiscreteFT_v2(d, Z0, kx, ky, kz, L, mu, Delta, omega, Gamma, Nzrel=200, z0=0, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar=1., t=1.):
    

    # boundaries for |z1-z2|
    max_zrel = max(d, d-Z0)
    # discrete lattice for relative coordinates
    zrelative = np.linspace(-max_zrel, max_zrel, Nzrel); N = Nzrel    
    
    # F2 in relative coordinates
    F2_k = np.zeros((4,4), dtype='complex'); n = 0
    
    # loop over relative coordinate z
    for z in zrelative:
    
        # separate coordinates z1, z2
        z1 = Z0 + 1/2*z; z2 = Z0 - 1/2*z
    
        # evaluate F2 as function of relative position z for fixed center of mass Z
        F2_rc = FMTI2_NeumannBC(d=d, z=z1, Z=z2, z0=z0, kx=kx, ky=ky, L=L, mu=mu, Delta=Delta, omega=omega, Gamma=Gamma, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar)
        
        # sum and multiply by complex exponential
        F2_k += np.exp( -2*np.pi*1j * kz * float(n/N) ) * F2_rc
        
        # increase n
        n += 1
        
    return F2_k








    
#####################################################################
################## PROJECTION OVER SINGLET/TRIPLET ##################
#####################################################################


# define physical basis for pairing
basis = [r"\uparrow +", r"\uparrow -", r"\downarrow +", r"\downarrow -"]

# build a 4×4 Matrix of Symbols f_{α,β}
M = Matrix([[ Symbol(f"f_{{{i},{j}}}") 
              for j in basis ] 
            for i in basis ])


# enable LaTeX rendering
init_printing(use_latex='mathjax')


# define Pauli matrices
s0 = Matrix([[1,0],[0,1]])
sx = Matrix([[0,1],[1,0]])
sy = Matrix([[0,-I],[I,0]])
sz = Matrix([[1,0],[0,-1]])


# define the basis for the projection
basis = {
    '11':   (s0 + sz)/2,
    '22':   (s0 - sz)/2,
    'sym':  sx/sqrt(2),
    'asym': I*sy/sqrt(2)
}


# convert basis into dictionary of numpy arrays
basis_np = {
    '11': np.matrix( basis['11'].tolist(), dtype=complex ),
    '22': np.matrix( basis['22'].tolist(), dtype=complex ),
    'sym': np.matrix( basis['sym'].tolist(), dtype=complex ),
    'asym': np.matrix( basis['asym'].tolist(), dtype=complex ),
}


# define LaTeX for spin matrices
basis_sigma = {
    '11':   r"\sigma_{\uparrow\uparrow}",
    '22':   r"\sigma_{\downarrow\downarrow}",
    'sym':  r"\sigma_{(\uparrow\downarrow+\downarrow\uparrow)}",
    'asym': r"\sigma_{(\downarrow\uparrow-\uparrow\downarrow)}"
}


# labels for projection
label_sigma = {
    '11':   r"\uparrow\uparrow",
    '22':   r"\downarrow\downarrow",
    'sym':  r"(\uparrow\downarrow+\downarrow\uparrow)",
    'asym': r"(\downarrow\uparrow-\uparrow\downarrow)"
}


# define LaTeX for orbital matrices
basis_lambda = {
    '11':   r"\lambda_{+ +}",
    '22':   r"\lambda_{- -}",
    'sym':  r"\lambda_{sym}",
    'asym': r"\lambda_{asym}"
}

# labels for projection
label_lambda = {
    '11':   r"++",
    '22':   r"--",
    'sym':  r"sym",
    'asym': r"asym"
}



# function which render the projection over spin singlet and triplet basis in Latex
def render_projection(M: Matrix, spin: str, orbital: str):

	# form the two-qubit basis element
    B_ab = kronecker_product(basis[spin], basis[orbital])
    # project
    c = simplify( trace(B_ab.H * M)/trace(B_ab.H*B_ab) )
    
    # build & display MathJax
    eq = rf"""
    f_{{{label_sigma[spin]}, {label_lambda[spin]}}}(\mathbf{{k}},\omega)
    \;=\;
    \mathrm{{Tr}}\!\Bigl[
      ({basis_sigma[spin]} \otimes {basis_lambda[orbital]})\, \Delta_{{\mathrm{{ind}}}}(\mathbf{{k}},\omega)
    \Bigr]
    \;=\;
    {latex(c)}
    """
    
    display(Math(eq))



# function which render the matrix corresponding to different channels in Latex
def render_channel(spin: str, orbital: str):

    # get spin matrix
    A = basis[spin]
    # get orbital matrix
    B = basis[orbital]
    # evaluate tensor product
    T = kronecker_product(A, B)
    
    eq = rf"""
    {basis_sigma[spin]} \otimes {basis_lambda[orbital]} 
    \;=\;
    {latex(T)} \,,
    \qquad
    \left( {basis_sigma[spin]} \otimes {basis_lambda[orbital]} \right)^\dagger 
    \;=\;
    {latex(T.H)}
    """
    display(Math(eq))



# project the pairing F over one selected channel
def projection(Delta, spin: str, orbital: str):

    # transform F into np.matrix
    Delta = np.matrix(Delta)
    # get the matrix to use for the projection
    Lambda_A = np.matrix( np.kron(basis_np[spin], basis_np[orbital]) )

    return np.trace(Lambda_A.H @ Delta) / np.trace(Lambda_A.H @ Lambda_A)



# project the pairing F over one selected channel
def channel(spin: str, orbital: str):

    Lambda_A = np.kron(basis_np[spin], basis_np[orbital])
    return Lambda_A








'''
#####################################################################
################## PROJECTION OVER PAULI MATRICES  ##################
#####################################################################


# enable LaTeX rendering
init_printing(use_latex='mathjax')


# define the Pauli matrices using sympy.Matrix
paulis = {
    '0': Matrix([[1, 0],
                 [0, 1]]),
    'x': Matrix([[0, 1],
                 [1, 0]]),
    'y': Matrix([[0, -I],
                 [I,  0]]),
    'z': Matrix([[1,  0],
                 [0, -1]])
}


# helper to get the LaTeX name of each spin Pauli matrix
sigma_name = {
    '0': r"\sigma_0",
    'x': r"\sigma_x",
    'y': r"\sigma_y",
    'z': r"\sigma_z"
}

# helper to get the LaTeX name of each spin Pauli matrix
lambda_name = {
    '0': r"\lambda_0",
    'x': r"\lambda_x",
    'y': r"\lambda_y",
    'z': r"\lambda_z"
}



# compute and display the outer product
def render_outer_product(p: str, q: str):

    # matrix in spin space
    A = paulis[p]
    # matrix in orbital space
    B = paulis[q]
    # use kronecker_product
    T = kronecker_product(A, B)
    
    # build and display the LaTeX equation
    eq = (
        rf"{sigma_name[p]} \,\otimes\, {lambda_name[q]}"
        rf" \;=\; {latex(T)}"
    )
    
    return Math(eq)


# spin singlet matrices
def spin_singlet_channels():

	# get y matrix in spin space
	A = paulis['y']

	singlet = []
	# loop over orbital Pauli matrices
	for b in ['0', 'x', 'y', 'z']:
    
		# get matrix in orbital space
		B = paulis[b]
		# evaluate outer product
		T = kronecker_product(A, B)

		# build and display the LaTeX equation
		eq = (
			rf" \Lambda^{{\mathrm{{singlet}}}}_{{{b}}} \;=\;"
			rf"{sigma_name['y']} \,\otimes\, {lambda_name[b]}"
			rf" \;=\; {latex(T)}"
		)
		
		# append to singlet
		singlet.append(Math(eq))		

	return singlet



# spin triplet matrices
def spin_triplet_channels(a: str):

	# get symmetric matrix in spin space
	A = paulis[a]

	triplet = []
	# loop over orbital Pauli matrices
	for b in ['0', 'x', 'y', 'z']:
    
		# get matrix in orbital space
		B = paulis[b]
		# evaluate outer product
		T = kronecker_product(A, B)

		# build and display the LaTeX equation
		eq = (
			rf" \Lambda^{{\mathrm{{triplet}}}}_{{{b}}} \;=\;"
			rf"{sigma_name[a]} \,\otimes\, {lambda_name[b]}"
			rf" \;=\; {latex(T)}"
		)
		
		# append to singlet
		triplet.append(Math(eq))		

	return triplet



# spin triplet or singlet matrices
def spin_channels(a: str):

	# get symmetric matrix in spin space
	A = paulis[a]
	
	if a == 'y':
		spin = 'singlet'
	else:
		spin = 'triplet'

	triplet = []
	# loop over orbital Pauli matrices
	for b in ['0', 'x', 'y', 'z']:
    
		# get matrix in orbital space
		B = paulis[b]
		# evaluate outer product
		T = kronecker_product(A, B)

		# build and display the LaTeX equation
		eq = (
			rf" \Lambda^{{ {spin} }}_{{{b}}} \;=\;"
			rf"{sigma_name[a]} \,\otimes\, {lambda_name[b]}"
			rf" \;=\; {latex(T)}"
		)
		
		# append to singlet
		triplet.append(Math(eq))		

	return triplet



# get and render in Latex the coefficients of the projection
def Pauli_matrices_projection(M: Matrix, spin: str, orbital: str):

	# construct matrix fro projection
    Lambda = kronecker_product(paulis[spin], paulis[orbital])
    
    # compute coefficient f_A
    f = simplify( trace(Lambda * M) / 4 ); f = factor_terms(f)
    
    eq = rf"""
    f_{{{spin}{orbital}}}(\mathbf{{k}},\omega)
    \;=\;
    \frac{{1}}{{4}}\,
    \mathrm{{Tr}}\!\Bigl[\,
      ({sigma_name[spin]}\otimes{lambda_name[orbital]})\,\Delta_{{\mathrm{{ind}}}}(\mathbf{{k}},\omega)
    \Bigr]
    \;=\;
    {latex(f)}
    """
    
    display(Math(eq))

'''






