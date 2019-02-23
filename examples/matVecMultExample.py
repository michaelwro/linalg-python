"""

EXAMPLE

matVecMult: linalg-python
Return the matrix-vector product.

By: Michael Wrona
Student, B.S. Aerospace Engineering
Iowa State University (Ames, IA)

"""


import numpy as np


def matVecMult(A, b):
    """
    MATRIX-VECTOR MULTIPLICATION
    Find the product from multiplying a matrix and vector
    **NOTE: INPUTS MUST BE NUMPY ARRAYS
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([2, 3])
    >>> matVecMult(a, b)
    >>> array([8, 18])
    """
    aRows, aCols = np.shape(A)  # Get dimensions
    bRows = len(b)

    if type(A) != np.ndarray or type(b) != np.ndarray:
        # Check for proper data type
        print("INPUT IS NOT NUMPY ARRAY IN ROUTINE 'matVecMult()'")
        return

    if int(aCols) != int(bRows):
        # Check for compatable dimensions of matrix/vector
        print("INCOMPATABLE ARRAY DIMENSIONS IN ROUTINE 'matVecMult()'")
        return

    output = np.zeros([aRows], dtype="float64")  # Create output array
    for i in range(aRows):
        output[i] = 0.0
        for j in range(aCols):
            output[i] += A[i, j] * b[j]
    return output


# Define matrix & vector
mat = np.array([[1, -1, 2], [0, -3, 1]])
vec = np.array([2, 1, 0])

# Calculate product
prod = matVecMult(mat, vec)

print("Result:")
print(prod)

