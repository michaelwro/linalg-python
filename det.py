"""

det: linalg-python
Use LU-factorization to calculate the determinant of a square matrix.

By: Michael Wrona
Student, B.S. Aerospace Engineering
Iowa State University (Ames, IA)

"""


import numpy as np


def det(A):
    """
    DETERMINANT OF A MATRIX
    Use LU-Decomposition to calculate the determinant of a matrix
    **NOTE: Input must be a NUMPY ARRAY

    INPUT:
    'A': Square input array of size n x n

    >>> A = np.array([[1, 2], [3, 4]])
    >>> det(A)
    >>> -2
    """
    # Check if input matrix is square
    aRows, aCols = np.shape(A)
    if aRows != aCols:
        print("ARRAY 'A' IS NOT SQUARE!")
        return

    # Check if input is Numpy array type
    if type(A) != np.ndarray:
        print("INPUT ARRAY IS NOT NUMPY ARRAY")
        return

    n = aRows  # Matrix dimension
    lu = np.zeros([n, n], dtype="float64")

    # LU Factorization
    for i in range(n):
        for j in range(i, n):
            x = 0.0
            for k in range(i):
                x += lu[i, k] * lu[k, j]
            lu[i, j] = a[i, j] - x
        for j in range(i + 1, n):
            x = 0.0
            for k in range(i):
                x += lu[j, k] * lu[k, i]
            lu[j, i] = (1 / lu[i, i]) * (a[j, i] - x)

    # Calculate determinant from diagonal elements
    det = 1.0
    for i in range(n):
        det *= lu[i, i]
    return det  # Return solution

