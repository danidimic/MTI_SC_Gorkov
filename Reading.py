import numpy as np



def Read_GreenFunction(filename, startline = 11):

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
				zlattice.append(gf[0].real)
				# remove z and z'
				gf = np.delete(gf, [0, 1])	
				# reshape into 4x4 array
				GF.append(np.reshape(gf, (4,4)))


	return np.array(zlattice), np.array(GF)


#z,gf = Read_GreenFunction('G2-zZ.out')

#print(z.shape, gf.shape)


