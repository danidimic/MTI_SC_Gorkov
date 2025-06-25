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
    's0': s0/sqrt(2),
    'sx': sx/sqrt(2),
    'sy': sy/sqrt(2),
    'sz': sz/sqrt(2),
    
    't0': s0/sqrt(2),
    'tx': sx/sqrt(2),
    'ty': sy/sqrt(2),
    'tz': sz/sqrt(2)
}



# convert basis into dictionary of numpy arrays
basis_numpy = {
    's0': np.matrix( basis['s0'].tolist(), dtype=complex ),
    'sx': np.matrix( basis['sx'].tolist(), dtype=complex ),
    'sy': np.matrix( basis['sy'].tolist(), dtype=complex ),
    'sz': np.matrix( basis['sz'].tolist(), dtype=complex ),
    
    't0': np.matrix( basis['s0'].tolist(), dtype=complex ),
    'tx': np.matrix( basis['sx'].tolist(), dtype=complex ),
    'ty': np.matrix( basis['sy'].tolist(), dtype=complex ),
    'tz': np.matrix( basis['sz'].tolist(), dtype=complex )    
}


# define LaTeX for spin matrices
basis_sigma = {
    's0':   r"\sigma_0",
    'sx':   r"\sigma_x",
    'sy':  r"\sigma_y",
    'sz':   r"\sigma_z"
}


# labels for projection
label_sigma = {
    's0':   r"\sigma_0",
    'sx':   r"\sigma_x",
    'sy':  r"\sigma_y",
    'sz':   r"\sigma_z"
}


# define LaTeX for orbital matrices
basis_lambda = {
    't0':   r"\tau_0",
    'tx':   r"\tau_x",
    'ty':   r"\tau_y",
    'tz':   r"\tau_z"
}

# labels for projection
label_lambda = {
    't0':   r"\tau_0",
    'tx':   r"\tau_x",
    'ty':   r"\tau_y",
    'tz':   r"\tau_z"
}



# function which render the projection over spin singlet and triplet basis in Latex
def Render_Projection(M: Matrix, spin: str, orbital: str):

	# form the two-qubit basis element
    B_ab = kronecker_product(basis[spin], basis[orbital])
    # project
    c = simplify( trace(B_ab.H * M) / trace(B_ab.H*B_ab) )
    
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
    
    # check normalization
    #print( 'trace = ', str(round(np.trace(Lambda_A.H @ Lambda_A))) )

    return np.trace(Lambda_A.H @ Delta) / np.trace(Lambda_A.H @ Lambda_A)



# get the matrix corresponding to given channel
def Pairing_Channel(spin: str, orbital: str):

    Lambda_A = np.kron(basis_numpy[spin], basis_numpy[orbital])
    return Lambda_A


