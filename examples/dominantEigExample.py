"""

EXAMPLE

DOMINANTEIG: linalg-python
Use power ieration to calculate the dominant eigenvalue of a square matrix.

By: Michael Wrona
Student, B.S. Aerospace Engineering
Iowa State University (Ames, IA)

"""


import numpy as np


def dominantEig(A):
    """
    FIND THE DOMINANT EIGENVALUE OF A MATRIX
    Use power iteration to find the largest eigenvalue of a matrix
    'A': Square matrix of interest
    **NOTE: INPUT MUST BE NUMPY ARRAY
    >>> a = np.array([[1, 2], [3, 4]])
    >>> dominantEig(a)
    >>> 5.372281
    """

    def norm(v):
        """ Calculate the norm of a vector (p=2) """
        n = len(v)
        out = 0.0
        for i in range(n):
            out += pow(abs(v[i]), 2)
        return np.sqrt(out)

    def matVecMult(A, b):
        """ Calculate matrix-vector product """
        aRows, aCols = np.shape(A)  # Get dimensions
        bRows = len(b)
        output = np.zeros([aRows], dtype="float64")  # Create output array
        for i in range(aRows):
            output[i] = 0.0
            for j in range(aCols):
                output[i] += A[i, j] * b[j]
        return output  # Return answer

    if type(A) != np.ndarray:
        # Check data type
        print("INPUT IS NOT NUMPY ARRAY IN ROUTINE 'dominantEig()'")
        return

    n, aCols = np.shape(A)
    if n != aCols:
        # Check if input is a square matrix
        print("INPUT ARRAY IS NOT SQUARE IN ROUTINE 'dominantEig()'")
        return

    tol = 1e-12  # Solution tolerance. Will break if met
    maxIter = 12000  # Max. number of iterations. Will break if met
    iters = 1
    lamTol = 0.0  # For calculating convergence

    w = np.zeros([n], dtype="float64")
    v = np.random.randn(n)  # Start with random initial condition/guess with norm(v0) = 1
    v = v / norm(v)

    while True:
        w = matVecMult(A, v)
        v = w / norm(w)
        prod = matVecMult(A, v)

        lam = 0.0
        for i in range(n):
            lam += v[i] * prod[i]

        if iters >= maxIter:
            # Break if max. iters. is reached
            print("MAX ITERS. REACHED IN ROUTINE 'dominantEig()'")
            break

        if abs(lam - lamTol) <= tol:
            # Break if within tolerance
            print("TOLERANCE MET IN ROUTINE 'dominantEig()'")
            break

        iters += 1
        lamTol = lam

    return lam


a = np.array([[2, 0, 0],
            [0, 3, 4],
            [0, 4, 9]], dtype='float64')     # Define matrix

eigenvalue = dominantEig(a)     # Determine dominant eigenvalue

# Print result
# Actual answer: 11
print("Dominant eigenvalue: %0.8f" % (eigenvalue))



