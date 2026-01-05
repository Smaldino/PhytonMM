import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ----------------------------------------------------------------------
# 1. Symbolic RREF with SymPy (unchanged logic, just comment added)
# ----------------------------------------------------------------------
AS = sym.Matrix([
    				[1, 2, 2, 3, 1, 0, 0],
					[2, 4, 5, 3, 0, 1, 0],
					[3, 6, 1, 4, 0, 0, 1]]
                )
AS_rref = AS.rref()
print("RREF and pivot columns:", AS_rref)


# ----------------------------------------------------------------------
# 2. Original echelon form function 
# ----------------------------------------------------------------------
def echelon_form(A):
	"""Original echelon form implementation with clearer variable names."""
	U = np.copy(A)
	m, n = A.shape
	j = 0
	pivot_columns = []
	
	for i in range(0,m):
		found_pivot = False
		while not found_pivot and j < n:
			# Find row with maximum absolute value in column j from row i onward
			max_row = np.argmax(abs(U[i:m, j])) + i

			# If the current entry is zero, swap with the max row
			if abs(U[i, j]) == 0:
				U[[i, max_row], :] = U[[max_row, i], :]

			# If after swapping we have a non-zero pivot, proceed
			if abs(U[i, j]) > 0:
				pivot_columns.append(j)
				# Eliminate entries below the pivot
				multipliers = U[i+1:m, j] / U[i, j]
				U[i+1:m, j+1:n] -= np.outer(multipliers, U[i, j+1:n])
				U[i+1:m, j] = 0
				j = j + 1
				found_pivot = True
			else:
				j = j + 1  # try next column
				found_pivot = False
	return (U, pivot_columns)


def echelon_form_verbose(A):
	"""Echelon form with intermediate matrix display."""
	U = np.copy(A).astype(float)
	m, n = A.shape
	j = 0
	pivot_columns = []
	
	print("Initial matrix:")
	print(U)
	print()
	
	for i in range(0, m):
		found_pivot = False
		while not found_pivot and j < n:
			max_row = np.argmax(abs(U[i:m, j])) + i
			
			if abs(U[i, j]) == 0:
				U[[i, max_row], :] = U[[max_row, i], :]
				print(f"Row swap: R{i+1} <-> R{max_row+1}")
				print(U)
				print()
			
			if abs(U[i, j]) > 1e-15:
				pivot_columns.append(j)
				print(f"Pivot found at position ({i+1}, {j+1}) = {U[i, j]:.3f}")
				
				multipliers = U[i+1:m, j] / U[i, j]
				for k, mult in enumerate(multipliers):
					if abs(mult) > 1e-15:
						print(f"R{i+k+2} = R{i+k+2} - ({mult:.3f}) * R{i+1}")
				
				U[i+1:m, j+1:n] -= np.outer(multipliers, U[i, j+1:n])
				U[i+1:m, j] = 0
				
				print("After elimination:")
				print(U)
				print()
				
				j += 1
				found_pivot = True
			else:
				j += 1
				found_pivot = False
	
	print(f"Final echelon form with pivot columns: {pivot_columns}")
	return (U, pivot_columns)


# ----------------------------------------------------------------------
# 3. Original RREF function 
# ----------------------------------------------------------------------
def RREF(A):
    U = np.copy(A).astype(float)
    (m, n) = A.shape
    j = 0
    pivot_columns = []
    for i in range(0, m):
        found_pivot = False
        while not found_pivot and j < n:
            indm = np.argmax(abs(U[i:m, j]))
            indm = indm + 1
            if abs(U[indm, j]) == 0:
                U[[i, indm], :] = U[[indm, i], :]
            if abs(U[i, j]) > 0:
                pivot_columns.append(j)
                M = U[i+1:m, j] / U[i, j]
                U[i+1:m, j+1:n] = U[i+1:m, j+1:n] - np.outer(M, U[i, j+1:n])
                U[i+1:m, j] = 0
                U[i, j:n] = U[i, j:n] / U[i, j]
                if i > 0:
                    M = U[0:i, j] / U[i, j]
                    U[0:i, j:n] = U[0:i, j:n] - np.outer(M, U[i, j:n])
                found_pivot = True
                j = j + 1
            else:
                found_pivot = False
                j = j + 1
    return (U, pivot_columns)


# ----------------------------------------------------------------------
# 4. Improved echelon form with tolerance (only variable names changed)
# ----------------------------------------------------------------------
def ef_withpivot(A): 
	U = np.copy(A) # copy the matrix A in U 
	(m,n)=A.shape
	j = 0 # index related to the column
	pivot_columns = []
	for i in range(0,m):
		found_pivot = False
		while not found_pivot & j < n:
			indm=np.argmax(abs(U[i:m,j])) # find the pivotal index, maximum element in the column j
			indm=indm+i
			if (indm != i) & (abs(U[indm,j]) > 1e-15): # perform the permutation to work well we should change != 0 with an absolute value less then a constant very small
				U[ [i, indm],:]=U[[indm,i],:]
			if (abs(U[i,j]) > 0):
				pivot_columns.append(j)
				M = U[i+1:m,j]/U[i,j] # vector because we do elimination of all the row below the pivotal one
				# in numpy array there is no difference row vectors or column vectors
				# np.outer to perform the outer product
				U[i+1:m,j+1:n]=U[i+1:m,j+1:n]-np.outer(M,U[i,j+1:n]) 
				U[i+1:m,j]=0
				found_pivot = True
				j=j+1
			else:
				found_pivot = False
				j=j+1    
	return(U,pivot_columns)

def rref_withpivot(A, r=None):
    # r is the number of columns that will be considered in the reduction phase
    U = np.copy(A).astype(float)  # ensure float for division
    m, n = A.shape
    if r is None:
        r = n  # default: consider all columns
    r = min(r, n)  # safety: don't exceed actual columns

    P = np.eye(m)
    j = 0
    for i in range(0, m): 
        ech = 1
        while (ech == 1) and (j < r):
            indm = np.argmax(abs(U[i:m, j])) + i
            if (indm != i) and (abs(U[indm, j]) > 1e-15):  # perform the permutation
                U[[i, indm], :] = U[[indm, i], :] 
                P[[i, indm], :] = P[[indm, i], :] 
            if abs(U[i, j]) > 1e-15:
                # Eliminate below
                M = U[i+1:m, j] / U[i, j]
                U[i+1:m, j+1:n] -= np.outer(M, U[i, j+1:n])
                U[i+1:m, j] = 0.0
                # Normalize pivot row
                U[i, j:n] /= U[i, j]
                # Eliminate above
                if i > 0:
                    M = U[0:i, j] / U[i, j]
                    U[0:i, j:n] -= np.outer(M, U[i, j:n])
                j += 1
                ech = 0
            else:
                j += 1
                ech = 1
    return U, P 


# ----------------------------------------------------------------------
# 5. Image compression demo (only variable names clarified)
# ----------------------------------------------------------------------
# Load image
try:
	image_data = mpimg.imread('0002.jpg')
	# Convert to grayscale
	rgb_weights = [0.2989, 0.5870, 0.1140]
	grayscale_image = np.dot(image_data, rgb_weights)

	# Apply echelon form with pivot
	echelon_result, pivot_columns = echelon_form_with_pivot(grayscale_image)

	# Show original and compressed (using only pivot columns)
	plt.figure()
	plt.subplot(1, 2, 1)
	plt.imshow(grayscale_image, cmap='gray')
	plt.title('Original Image')

	plt.subplot(1, 2, 2)
	plt.imshow(grayscale_image[:, pivot_columns], cmap='gray')
	plt.title('Compressed Image (Pivot Columns Only)')
	plt.show()

except FileNotFoundError:
	print("Image '0002.jpg' not found. Skipping image demo.")
	

# Matrix examples

M1 = np.array([
    [2, 1, 3], 
    [4, 2, 1], 
    [6, 3, 4]
    ])
M2 = np.array([
    [1, 2, 3, 4], 
    [2, 4, 6, 8],
    [3, 6, 9, 12]
    ])
M3 = np.array([
    [1, 0, 2],
    [0, 1, 3],
    [0, 0, 0]
    ])

# print("\nMatrix 1:")
# print(M1)
# result1, pivots1 = echelon_form(M1)
# print("\nEchelon form:")
# print(result1)

# print("\nMatrix 2:")
# print(M2)
# result2, pivots2 = echelon_form(M2)
# print("\nEchelon form:")
# print(result2)

# print("\nMatrix 3:")
# print(M3)
# result3, pivots3 = echelon_form(M3)
# print("\nEchelon form:")
# print(result3)


# Test verbose function
print("\n" + "="*50)
print("VERBOSE ECHELON FORM DEMONSTRATION")
print("="*50)
print("\nTesting with Matrix 1:")
echelon_form_verbose(M1)