"""

EXAMPLE

LUDECOMPOSE: linalg-python
Decompose a square matrix into its LU-factors

By: Michael Wrona
Student, B.S. Aerospace Engineering
Iowa State University (Ames, IA)

"""


import numpy as np


def luDecompose(A):
    """
    LUDECOMPUSE
    Decompose a square matrix into its LU-factors
    ** NOTE: 'A' MUST be NumPy ARRAYS:
    >>> A = np.array([n, n])    # Ex: np.array([[2, -1], [4, 3]])
    """
    # Check if input matrix is square
    aRows, aCols = np.shape(A)
    if aRows != aCols:
        print("ARRAY 'A' IS NOT SQUARE IN ROUTINE 'luDecompose()'")
        return

    # Check if input is Numpy array type
    if type(A) != np.ndarray:
        print("INPUT ARRAY IS NOT NUMPY ARRAY")
        return

    n = aRows

    lower = np.zeros([n, n], dtype="float64")
    upper = np.zeros([n, n], dtype="float64")

    # DECOMPOSE MATRIX
    for i in range(n):
        # UPPER TRIANGULAR MATRIX
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += lower[i, j] * upper[j, k]
            upper[i, k] = A[i, k] - sum

        # LOWER TRIANGULAR MATRIX
        for k in range(i, n):
            if i == k:
                lower[i, i] = 1  # Diagonal element as 1
            else:
                sum = 0
                for j in range(i):
                    sum += lower[k, j] * upper[j, i]
                lower[k, i] = (A[k, i] - sum) / upper[i, i]

    return lower, upper


# Define matrix
a = np.array([[3, 1, 6], [-6, 0, -16], [0, 8, -17]])

# Perform LU-factorization
l, u = luDecompose(a)

print("Lower Factor:")
print(l)

print("Upper Factor:")
print(u)
