"""

GENSPDSYSTEM: linalg-python
Generate a dense, symmetric positive-definite system of linerar equations.
SPD systems are commonly solved using iterative solvers.

By: Michael Wrona
Student, B.S. Aerospace Engineering
Iowa State University (Ames, IA)

"""


import numpy as np


def genSPDSystem(n):
    """
    GENERATE SYMMETRIC-POSITIVE-DEFINITE SYSTEM OF LINEAR EQUATIONS
    Given a system size 'n,' generate a dense SPD system of linear equations
    with dimensions A[n, n] and b[n].
    Return coefficient matrix 'A' and RHS vector 'b'
    This can be used to test iterative solvers that need SPD systems
    >>> n =  3  # System size
    A, b = genSPDSystem(n)  # Return coefficients & RHS of system
    Uses method found on Stack Overflow forum: https://math.stackexchange.com/a/358092
    """
    if type(n) != int:
        print("INPUT 'n' TYPE MUST BE 'int' IN ROUTINE 'genSPDSystem()'")
        return

    Atemp = np.random.rand(n, n)  # Generate random coef. matrix 'A'
    Atrans = np.zeros([n, n], dtype="float64")
    A = np.zeros([n, n], dtype="float64")

    # Compute transpose
    for i in range(n):
        for j in range(n):
            Atrans[i, j] = Atemp[j, i]

    # Compute matrix multiplication
    for i in range(n):
        for j in range(n):
            out = 0
            for k in range(n):
                out += (Atemp[i, k] * Atrans[k, j]) + n
            A[i, j] = out
    b = np.random.rand(n)  # Generate RHS vector 'b'
    return A, b


"""
To confirm the system is positive-definite,
the determinant of the coefficient matrix should be positive and non-singular
"""
n = 3
A, b = genSPDSystem(n)

print("Coefficient matrix:")
print(A)
print("\nSolution/LHS:")
print(b)

print("Det(A) = %0.8f" % (np.linalg.det(A)))

