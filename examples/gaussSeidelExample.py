"""

EXAMPLE

GAUSSSEIDEL: linalg-python
Solve a linear system of equations using the iterative Gauss-Seidel method.
This method only applies to positive-definite and/or symmetric systems.

By: Michael Wrona
Student, B.S. Aerospace Engineering
Iowa State University (Ames, IA)

"""


import numpy as np


def gaussSeidel(a, b):
    """
    GAUSS-SEIDEL METHOD (Iterative)
    Solve a linear system of equations using the Gauss-Seidel Method.
    'A': Coefficient matrix (n, n)
    'b': Solution matrix (n, 1)
    ** NOTE: 'A' and 'b' MUST be NumPy ARRAYS:
    >>> A = np.array([n, n])    # Ex: np.array([[2, -1], [4, 3]])
    >>> b = np.array([n])       # Ex: np.array([2, -1])
    Algorithm from 'Templates for the Solution of Linear Systems:
    Building Blocks for Iterative Methods' (Barrett et. al.) found
    at http://www.siam.org/books
    """

    tol = 1e-12  # Solution tolerance. Will break if reached
    maxIter = 12000  # Maximum number of iterations. Will break if reached
    iters = 1  # Iteration counter

    # Check that input arrays are the correct shapes
    aRows, aCols = np.shape(a)
    bRows, = np.shape(b)

    if aCols != bRows:
        print("ARRAY DIMENSIONS DO NOT AGREE!")
        print("Shape of array 'A' must have same number of rows 'b' has")
        return 0
    if aRows != aCols:
        print("ARRAY 'A' IS NOT SQUARE!")
        return

    # Check if input is Numpy array type
    if type(a) != np.ndarray or type(b) != np.ndarray:
        print("INPUT ARRAY IS NOT NUMPY ARRAY")
        return

    # # The 'checkPosDef.py' function can be used to check if system is PD:
    # # Check if coefficient matrix is positive definite
    # posDefCheck = checkPosDef(a)
    # if posDefCheck == False:
    #     return

    n = aRows  # System dimensions
    x = np.random.rand(n) * 5  # initial guess for solution
    sigmaTol = 1  # Used to calculate tolerance

    while True:
        if iters > maxIter:
            print("MAX. ITERS. REACHED IN ROUTINE 'gaussSeidel()'")
            break

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += a[i, j] * x[j]
            x[i] = (b[i] - sigma) / a[i, i]

        tolerance = abs(sigmaTol - sigma)
        if tolerance <= tol:
            print("TOLERANCE MET IN ROUTINE 'gaussSeidel()' \t %d ITERS." % (iters))
            break

        sigmaTol = sigma
        iters += 1

    return x  # Return Solution


# SOLUTION: [5.5, 8, 4.5] (PD)
a = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype="float64")
b = np.array([3, 6, 1], dtype="float64")
soln = gaussSeidel(a, b)  # Solve system

print("Solution:")
print(soln)
