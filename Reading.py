import cmath
import numpy as np



# read the files with the 2nd order correction to GF
# startline = offset to start rreading values
# zIdx = select z (zIdx=0) or z' (zIdx=1) as independent variable
def Read_GreenFunction(filename, startline = 11, zIdx=0):

	# path in local folder
	fn = 'MC-Integrals/Results/' + filename
	
	# list for z and GF
	zlattice = []; GF = []

	with open(fn, 'r') as file:
		
		idx = 0
		# loop over lines of file
		for line in file:
			
			# count lines
			idx += 1
			# read line
			values = map(complex, line.split())
			
			if idx >= startline:
			
				gf = []
				# loop over elements in line
				for value in values:
					# append value
					gf.append(value)
			
			
				
				# add z to list
				zlattice.append(gf[zIdx].real)
				# remove z or z'
				gf = np.delete(gf, [0, 1])
				# reshape into 4x4 array
				GF.append(np.reshape(gf, (4,4)))

	
	# convert to np.array
	zlattice = np.array(zlattice); GF = np.array(GF)

	# sorted indices
	indices = np.argsort(zlattice)
	# sorted lattice vector
	zlattice = zlattice[indices]
	# sorted GF correspondingly
	GF = GF[indices]
	
	return zlattice, GF




# get the pairing as module and phase
def Get_Pairing(filename, startline = 11, zIdx=0):

	zlattice, GF = Read_GreenFunction(filename)

	Delta = []; phi = []
	# loop over zlattice
	for idx in range(len(zlattice)):
	
		# get module of GF
		module = [[ abs(GF[idx][irow][icol]) for icol in range(4)] for irow in range(4)]
		# get phase of GF
		phase = [[ cmath.phase(GF[idx][irow][icol]) for icol in range(4)] for irow in range(4)]
		
		# add to Delta and phi
		Delta.append(module); phi.append(phase)
	
	
	return zlattice, np.array(Delta), np.array(phase)
















