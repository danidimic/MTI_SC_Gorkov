from IPython.display import Math, display
from sympy import Symbol, Matrix, I, init_printing, simplify, trace, latex, kronecker_product


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




#####################################################################
######################## PAIRING CHANNELS  ##########################
#####################################################################



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


#####################################################################
##################### PROJECTION COEFFICIENTS  ######################
#####################################################################


# define your physical basis labels (spin x orbital)
basis = [r"\uparrow+", r"\uparrow-", r"\downarrow+", r"\downarrow-"]

# build a 4×4 Matrix of Symbols f_{α,β}
M = Matrix([[ Symbol(f"f_{{{i},{j}}}") 
              for j in basis ] 
            for i in basis ])


# get and render in Latex the coefficients of the projection
def coefficient_projection(M: Matrix, spin: str, orbital: str):

	# construct matrix fro projection
    Lambda = kronecker_product(paulis[spin], paulis[orbital])
    
    # compute coefficient f_A
    f = simplify(trace(Lambda * M) / 4)
    
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


