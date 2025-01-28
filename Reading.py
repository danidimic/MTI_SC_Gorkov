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
			
			if idx > 10:
			
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


#z,gf = Read_GreenFunction('G2-zZ.out')

#print(z.shape, gf.shape)


