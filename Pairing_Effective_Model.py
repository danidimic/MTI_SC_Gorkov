import cmath
import numpy as np

from MTI_Second_Order import FMTI2_NeumannBC, FMTI2_Relative_Coordinates, FMTI2_Wigner_Transform, Change_Basis, Block_Decomposition, Block_Reverse

from IPython.display import Math, display
from sympy import Symbol, Matrix, I, init_printing, simplify, factor_terms, trace, latex, kronecker_product, sqrt





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



# define the basis for the projection
basis = {
    's0': s0,
    'sx': sx,
    'sy': sy,
    'sz': sz,
    
    't0': s0,
    'tx': sx,
    'ty': sy,
    'tz': sz
}


# convert basis into dictionary of numpy arrays
basis_numpy = {
    's0': np.matrix( basis['s0'].tolist(), dtype=complex ),
    'sx': np.matrix( basis['sx'].tolist(), dtype=complex ),
    'sy': np.matrix( basis['sy'].tolist(), dtype=complex ),
    'sz': np.matrix( basis['sz'].tolist(), dtype=complex ),
    
    't0': np.matrix( basis['t0'].tolist(), dtype=complex ),
    'tx': np.matrix( basis['tx'].tolist(), dtype=complex ),
    'ty': np.matrix( basis['ty'].tolist(), dtype=complex ),
    'tz': np.matrix( basis['tz'].tolist(), dtype=complex )    
}


# labels for projection
label_latex = {
    's0':  r"\sigma_0",
    'sx':  r"\sigma_x",
    'sy':  r"\sigma_y",
    'sz':  r"\sigma_z",
    
    't0':   r"\tau_0",
    'tx':   r"\tau_x",
    'ty':   r"\tau_y",
    'tz':   r"\tau_z"
}


'''
# define LaTeX for matrices
basis_latex = {
    's0':  r"\sigma_0",
    'sx':  r"\sigma_x",
    'sy':  r"\sigma_y",
    'sz':  r"\sigma_z",
    
    't0':   r"\tau_0",
    'tx':   r"\tau_x",
    'ty':   r"\tau_y",
    'tz':   r"\tau_z"
}
'''



# function which render the projection over spin singlet and triplet basis in Latex
def Render_Projection(M: Matrix, spin: str, orbital: str):

    # get spin matrix
    S = basis[spin]
    # get orbital matrix
    O = basis[orbital]
    # evaluate tensor product
    Lambda_A = kronecker_product(S,O)
    
    f_A = simplify( trace(Lambda_A.H * M) / trace(Lambda_A.H*Lambda_A) )
    
    # build & display MathJax
    eq = rf"""
    f_{{{label_latex[spin]}, {label_latex[orbital]}}}(z)
    \;=\;
    \mathrm{{Tr}}\!\Bigl[
      ({label_latex[spin]} \otimes {label_latex[orbital]})\, \Delta_{{\mathrm{{ind}}}}(z)
    \Bigr]
    \;=\;
    {latex(f_A)}
    """
    
    display(Math(eq))



# function which render the matrix corresponding to different channels in Latex
def Render_Channel(spin: str, orbital: str):

    # get spin matrix
    S = basis[spin]
    # get orbital matrix
    O = basis[orbital]
    # evaluate tensor product
    Lambda_A = simplify( kronecker_product(S,O) )
    
    eq = rf"""
    {label_latex[spin]} \otimes {label_latex[orbital]} 
    \;=\;
    {latex(Lambda_A)} \,,
    \qquad
    \left( {label_latex[spin]} \otimes {label_latex[orbital]} \right)^T 
    \;=\;
    {latex(Lambda_A.T)}
    """
    display(Math(eq))




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
	for idx,spin in zip( range(4), ['s0', 'sx', 'sy', 'sz']):
	
		# get spin matrix
		S = basis_numpy[spin]
		
		# loop over orbital matrices
		for jdx,orbital in zip( range(4), ['t0', 'tx', 'ty', 'tz']):

			# get orbital matrix
			O = basis_numpy[orbital]

			# evaluate tensor product
			Lambda_A = np.matrix( np.kron(S,O) )
			
			# get the projection
			coeffs[idx,jdx] = np.trace(Lambda_A.H @ Delta) / np.trace(Lambda_A.H @ Lambda_A)

	if normalize == True: coeffs /= np.linalg.norm(coeffs)
	
	return coeffs



# function that reconstruct the pairing from the coefficients
def Reconstruct(coeffs):

	pairing = np.zeros((4,4),  dtype='complex')
	
	# loop over spin matrices
	for idx,spin in zip( range(4), ['s0', 'sx', 'sy', 'sz']):
	
		# get spin matrix
		S = basis_numpy[spin]
		
		# loop over orbital matrices
		for jdx,orbital in zip( range(4), ['t0', 'tx', 'ty', 'tz']):

			# get orbital matrix
			O = basis_numpy[orbital]

			# evaluate tensor product
			Lambda_A = np.matrix( np.kron(S,O) )
			
			pairing += coeffs[idx,jdx] * Lambda_A
	

	return pairing



'''
M = np.kron(basis_numpy['sy'], basis_numpy['t0'])
print(M)
print()
print(np.transpose(M))
print()
print(M==-np.transpose(M))
'''




