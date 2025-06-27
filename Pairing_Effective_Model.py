import cmath
import numpy as np

from MTI_Second_Order import FMTI2_NeumannBC, FMTI2_Relative_Coordinates, FMTI2_Wigner_Transform, Change_Basis, Block_Decomposition, Block_Reverse

from IPython.display import Math, display
from sympy import Symbol, Matrix, I, init_printing, simplify, nsimplify, factor_terms, trace, latex, kronecker_product, sqrt





#####################################################################
##################### PROJECTION OVER CHANNELS ######################
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

# define Pauli matrices
t0 = Matrix([[1,0],[0,1]])
tx = Matrix([[0,1],[1,0]])
ty = Matrix([[0,-I],[I,0]])
tz = Matrix([[1,0],[0,-1]])


# define the basis for the projection
basis = {
    'uu': 1/2*(s0+sz),
    'sym': sx/sqrt(2),
    'asym': I*sy/sqrt(2),
    'dd': 1/2*(s0-sz),
    
    '++': 1/2*(t0+tz),
    '+-': 1/2*(tx+I*ty),
    '-+': 1/2*(tx-I*ty),
    '--': 1/2*(t0-tz)
}


# convert basis into dictionary of numpy arrays
basis_numpy = {
    'uu': np.matrix( basis['uu'].tolist(), dtype=complex ),
    'sym': np.matrix( basis['sym'].tolist(), dtype=complex ),
    'asym': np.matrix( basis['asym'].tolist(), dtype=complex ),
    'dd': np.matrix( basis['dd'].tolist(), dtype=complex ),
    
    '++': np.matrix( basis['++'].tolist(), dtype=complex ),
    '+-': np.matrix( basis['+-'].tolist(), dtype=complex ),
    '-+': np.matrix( basis['-+'].tolist(), dtype=complex ),
    '--': np.matrix( basis['--'].tolist(), dtype=complex )    
}


# labels for projection
basis_latex = {
    'uu':  r"\frac12 \left( \sigma_0 + \sigma_z \right) ",
    'sym':  r"\frac{1}{\sqrt{2}} \sigma_x ",
    'asym':  r"\frac{i}{\sqrt{2}} \sigma_y ",
    'dd':  r"\frac12 \left( \sigma_0 - \sigma_z \right) ",
    
    '++':   r"\frac12 \left( \tau_0 + \tau_z \right) ",
    '+-':   r"\frac12 \left( \tau_x + i \tau_y \right) ",
    '-+':   r"\frac12 \left( \tau_x - i \tau_y \right) ",
    '--':   r"\frac12 \left( \tau_0 - \tau_z \right) "
}



# function which render the projection over spin singlet and triplet basis in Latex
def Render_Projection(M: Matrix, spin: str, orbital: str):

    # get spin matrix
    S = basis[spin]
    # get orbital matrix
    O = basis[orbital]
    # evaluate tensor product
    Lambda_A = kronecker_product(S,O)
    
    # get trace
    tr = nsimplify(simplify( trace(Lambda_A.H*Lambda_A) ))
    # project
    f_A = nsimplify(simplify( trace(Lambda_A.H * M) / trace(Lambda_A.H*Lambda_A) ))
    
    eq = rf"""
    \Lambda_A 
    \;=\;
    {basis_latex[spin]} \otimes {basis_latex[orbital]}
    """
    display(Math(eq))
    
    
    # build & display MathJax
    eq = rf"""
    f_A(z)
    \;=\;
    \mathrm{{Tr}}\!\left[
      \Lambda_A^\dagger \, 
      \Delta_{{\mathrm{{ind}}}}(z)
    \right]
    \;=\;
    {latex(f_A)} \,,
    \qquad
    \mathrm{{Tr}}\!\left[
      \Lambda_A^\dagger \, \Lambda_A
    \right]
    \;=\;
    {latex(tr)}
    """
    display(Math(eq))
    


# function which render the matrix corresponding to different channels in Latex
def Render_Channel(spin: str, orbital: str):

    # get spin matrix
    S = basis[spin]
    # get orbital matrix
    O = basis[orbital]
    # evaluate tensor product
    Lambda_A = kronecker_product(S,O).applyfunc(nsimplify)
    
    eq = rf"""
    {basis_latex[spin]} \otimes {basis_latex[orbital]} 
    \;=\;
    {latex(Lambda_A)}
    """
    display(Math(eq))



# function that computes the projection simbolically 
def Simpy_Projection(M: Matrix, spin: str, orbital: str):

    # get spin matrix
    S = basis[spin]
    # get orbital matrix
    O = basis[orbital]
    # evaluate tensor product
    Lambda_A = kronecker_product(S,O)
    
    # project
    f_A = nsimplify(simplify( trace(Lambda_A.H * M) / trace(Lambda_A.H*Lambda_A) ))
    
    return f_A



# project the pairing F over one selected channel
def Pairing_Projection(Delta, spin: str, orbital: str):

    # get spin matrix
    S = basis_numpy[spin]
    # get orbital matrix
    O = basis_numpy[orbital]
    # evaluate tensor product
    Lambda_A = np.matrix( np.kron(S,O) )
    
    return np.trace(Lambda_A.H @ Delta) / np.trace(Lambda_A.H @ Lambda_A)



# get the matrix corresponding to given channel
def Pairing_Channel(spin: str, orbital: str):

    # get spin matrix
    S = basis_numpy[spin]
    # get orbital matrix
    O = basis_numpy[orbital]
    # evaluate tensor product
    Lambda_A = np.matrix( np.kron(S,O) )
    
    return Lambda_A



# get all coefficients from projection
def Project_All(Delta, normalize=True):

	# matrix for coefficients
	coeffs = np.zeros((4, 4), dtype=complex)
	
	# loop over spin matrices
	for idx,spin in zip( range(4), ['asym', 'sym','uu', 'dd']):
	
		# get spin matrix
		S = basis_numpy[spin]
		
		# loop over orbital matrices
		for jdx,orbital in zip( range(4), ['++', '+-', '-+', '--']):

			# get orbital matrix
			O = basis_numpy[orbital]

			# evaluate tensor product
			Lambda_A = np.matrix( np.kron(S,O) )
			
			# get the projection
			coeffs[idx,jdx] = np.trace(Lambda_A.H @ Delta) / np.trace(Lambda_A.H @ Lambda_A)
		

	# normalize coefficients
	if normalize == True: 
		coeffs /= np.linalg.norm(coeffs)
	
	return coeffs



# function that reconstruct the pairing from the coefficients
def Reconstruct(coeffs):

	pairing = np.zeros((4,4),  dtype='complex')
	
	# loop over spin matrices
	for idx,spin in zip( range(4), ['asym', 'sym', 'uu', 'dd']):
	
		# get spin matrix
		S = basis_numpy[spin]
		
		# loop over orbital matrices
		for jdx,orbital in zip( range(4), ['++', '+-', '-+', '--']):

			# get orbital matrix
			O = basis_numpy[orbital]

			# evaluate tensor product
			Lambda_A = np.matrix( np.kron(S,O) )
			
			pairing += coeffs[idx,jdx] * Lambda_A
	

	return pairing



