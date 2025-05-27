import cmath
import numpy as np

from MTI_Second_Order import FMTI2_NeumannBC, FMTI2_Relative_Coordinates, FMTI2_Wigner_Transform, Change_Basis

from IPython.display import Math, display
from sympy import Symbol, Matrix, I, init_printing, simplify, factor_terms, trace, latex, kronecker_product, sqrt




#####################################################################
################### PROJECTION OVER SPIN/ORBITAL ####################
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
basis_numpy = {
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
def Render_Projection(M: Matrix, spin: str, orbital: str):

	# form the two-qubit basis element
    B_ab = kronecker_product(basis[spin], basis[orbital])
    # project
    c = simplify( trace(B_ab.H * M)/trace(B_ab.H*B_ab) )
    
    # build & display MathJax
    eq = rf"""
    f_{{{label_sigma[spin]}, {label_lambda[orbital]}}}(\mathbf{{k}},\omega)
    \;=\;
    \mathrm{{Tr}}\!\Bigl[
      ({basis_sigma[spin]} \otimes {basis_lambda[orbital]})\, \Delta_{{\mathrm{{ind}}}}(\mathbf{{k}},\omega)
    \Bigr]
    \;=\;
    {latex(c)}
    """
    
    display(Math(eq))



# function which render the matrix corresponding to different channels in Latex
def Render_Channel(spin: str, orbital: str):

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
def Pairing_Projection(Delta, spin: str, orbital: str):

    # transform F into np.matrix
    Delta = np.matrix(Delta)
    # get the matrix to use for the projection
    Lambda_A = np.matrix( np.kron(basis_numpy[spin], basis_numpy[orbital]) )

    return np.trace(Lambda_A.H @ Delta) / np.trace(Lambda_A.H @ Lambda_A)



# get the matrix corresponding to given channel
def Pairing_Channel(spin: str, orbital: str):

    Lambda_A = np.kron(basis_numpy[spin], basis_numpy[orbital])
    return Lambda_A




#####################################################################
######################## SYMMETRY ANALYSIS  #########################
#####################################################################


# Function that evaluates the 4 symmetric components in frequency and momentum
def Symmetric_Components(d, Z0, k, kx, ky, L, mu, Delta, omega, Gamma, N=199, z0=0, C = -0.0068, D1 = 1.3, D2 = 19.6, A1 = 2.2, A2 = 4.1, M = 0.28, B1 = 10, B2 = 56.6, hbar=1., t=1.):

	### Mind the change of basis to have the pairing in spin x orbital space !!! ###
	
	# evaluate the pairing at the given parameters
	F = Change_Basis( FMTI2_Wigner_Transform(d=d, Z0=Z0, k=k, kx=kx, ky=ky, L=L, mu=mu, Delta=Delta, omega=omega, Gamma=Gamma, N=N, z0=z0, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar, t=t) )

	# evaluate the pairing reverting the momentum
	F_k = Change_Basis( FMTI2_Wigner_Transform(d=d, Z0=Z0, k=-k, kx=-kx, ky=-ky, L=L, mu=mu, Delta=Delta, omega=omega, Gamma=Gamma, N=N, z0=z0, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar, t=t) )
	
	# evaluate the pairing reverting the frequency
	F_w = Change_Basis( FMTI2_Wigner_Transform(d=d, Z0=Z0, k=k, kx=kx, ky=ky, L=L, mu=mu, Delta=Delta, omega=-omega, Gamma=Gamma, N=N, z0=z0, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar, t=t) )

	# evaluate the pairing reverting frequency and momentum
	F_kw = Change_Basis( FMTI2_Wigner_Transform(d=d, Z0=Z0, k=-k, kx=-kx, ky=-ky, L=L, mu=mu, Delta=Delta, omega=-omega, Gamma=Gamma, N=N, z0=z0, C=C, D1=D1, D2=D2, A1=A1, A2=A2, M=M, B1=B1, B2=B2, hbar=hbar, t=t) )


	# even energy, even momentum
	ee = 1./4. * (F + F_k + F_w + F_kw)
	
	# even energy, odd momentum
	eo = 1./4. * (F - F_k + F_w - F_kw)
	
	# odd energy, even momentum
	oe = 1./4. * (F + F_k - F_w - F_kw)
	
	# odd energy, odd momentum
	oo = 1./4. * (F - F_k - F_w + F_kw)	


	# save components and full pairing
	symmetrized_pairing = {
		# even in energy
		'even': {
			'even': ee,
			'odd':  eo
		},
		# odd in energy
		'odd': {
			'even': oe,
			'odd':  oo
		},
		
		# full pairing matrix
		'total': F
	}
	
	return symmetrized_pairing



# Function which acces a specified component and channel in symmetrized pairing
def Get_Symmetry_Channel(symmetrized_pairing, frequency: str, momentum: str, spin: str, orbital: str):

	# extract component with given symmetry in frequency/momentum
	F_sym = symmetrized_pairing[frequency][momentum]
	
	# project over spin/orbital channel 
	f_A_sym = Pairing_Projection(Delta=F_sym, spin=spin, orbital=orbital)
	
	return f_A_sym
    


# Function which return the total component and project over spin/orbital
def Get_Total_Component(symmetrized_pairing, spin: str, orbital: str):

	# get total pairing matrix
	F_tot = symmetrized_pairing['total']
	
	# project over spin/orbital channel 
	f_A = Pairing_Projection(Delta=F_tot, spin=spin, orbital=orbital)
	
	return f_A
	


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






