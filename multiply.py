"""

multiply: linalg-python
Use the 'ijk' algorithm to calculate a matrix-matrix product.

By: Michael Wrona
Student, B.S. Aerospace Engineering
Iowa State University (Ames, IA)

"""


import numpy as np


def multiply(a, b):
    """
    MULTIPLY TWO MATRICES
    INPUT
    'a', 'b': Input matrices

    NOTE: 'a' and 'b' MUST be NumPy ARRAYS:
    >>> a = np.array([p, n])
    >>> b = np.array([n, m])
    Pseudo-Code from: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
    """
    aRows, aCols = np.shape(a)  # Get dimensions of matrices
    bRows, bCols = np.shape(b)

    # Check that matrix dimensions agree for multiplication
    if aCols != bRows:
        print("MATRIX DIMENSIONS DO NOT AGREE!!")
        return

    # Check if input is Numpy array type
    if type(a) != np.ndarray or type(b) != np.ndarray:
        print("INPUT ARRAY IS NOT NUMPY ARRAY")
        return

    # Compute matrix multiplication (ijk method)
    outMat = np.zeros([aRows, bCols], dtype="float64")  # Shape of output matrix
    for i in range(aRows):
        for j in range(bCols):
            sum = 0
            for k in range(aCols):
                sum += a[i, k] * b[k, j]
            outMat[i, j] = sum

    return outMat

