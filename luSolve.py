"""

LUSOLVE: linalg-python
Solve a linear system of equations using LU-factorization.

By: Michael Wrona
Student, B.S. Aerospace Engineering
Iowa State University (Ames, IA)

"""


import numpy as np


def luSolve(a, b):
    """
    SOLVE LINEAR SYSTEM USING LU DECOMPOSITION
    Use LU decomposition to solve a linear system of equations.
    'a': Coefficient matrix [n, n]
    'b': Solution matrix [n, 1]
    ** NOTE: 'A' and 'b' MUST be NumPy ARRAYS:
    >>> a = np.array([n, n])    # Ex: np.array([[2, -1], [4, 3]])
    >>> b = np.array([n])       # Ex: np.array([2, -1])
    Algorithm translated from C# code: https://en.wikipedia.org/wiki/LU_decomposition
    """

    # Check if input matrix is square
    aRows, aCols = np.shape(a)
    bRows, = np.shape(b)
    if aRows != aCols:
        print("ARRAY 'A' IS NOT SQUARE!")
        return

    # Check if coef. matrix ad RHS vector have correct dimensions
    if aCols != bRows:
        print("INCOMPATABLE INPUT ARRAY DIMENSIONS IN ROUTINE 'luSolve()'")
        return

    # Check if input is Numpy array type
    if type(a) != np.ndarray or type(b) != np.ndarray:
        print("INPUT ARRAY IS NOT NUMPY ARRAY IN ROUTINE 'luSolve()'")
        return

    n = aRows
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

    # Find solution of Ly = b
    y = np.zeros([n], dtype="float64")
    for i in range(n):
        x = 0.0
        for k in range(i):
            x += lu[i, k] * y[k]
        y[i] = b[i] - x

    # Find solution of Ux = y
    soln = np.zeros([n], dtype="float64")
    for i in range(n - 1, -1, -1):
        x = 0.0
        for k in range(i + 1, n):
            x += lu[i, k] * soln[k]
        soln[i] = (1 / lu[i, i]) * (y[i] - x)

    return soln

