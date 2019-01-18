"""

CHECKPOSDEF: linalg-python
A rough check to determine if a square matrix is positive definite.

By: Michael Wrona
Student, B.S. Aerospace Engineering
Iowa State University (Ames, IA)

"""


import numpy as np


def checkPosDef(A):
    """
    CHECK IF MATRIX IS POSITIVE-DEFINITE
    Use Sylvester's criterion to check if a 
    square NumPy array MAY be non-positive-definite.
    Parameters
    INPUT:
    'A' : Input array of size n x n

    OUTPUT:
    BOOL 'True' if matrix is positive-definite
    BOOL 'False' if matrix is NOT positive-definite

    **NOTE: Only up to a upper-left 3x3 sub-matrix
    determinant is implemented in this code.
    >>> np.array([[1, 2], [3, 4]])
    >>> 0   # Bool False, because the matrix is not positive definite
    """

    def det2(A):
        """
        Calculate determinant of 2x2 system
        """
        det = (A[0, 0] * A[1, 1]) - (A[0, 1] - A[1, 0])
        return det

    def det3(A):
        """
        Calculate determinant of 3x3 system
        """
        det1 = A[0, 0] * ((A[1, 1] * A[2, 2]) - (A[1, 2] * A[2, 1]))
        det2 = A[0, 1] * ((A[1, 0] * A[2, 2]) - (A[1, 2] * A[2, 0]))
        det3 = A[0, 2] * ((A[1, 0] * A[2, 1]) - (A[1, 1] - A[2, 0]))
        det = det1 - det2 + det3
        return det

    aRows, aCols = np.shape(A)
    if aRows != aCols:
        print("ARRAY 'A' IS NOT SQUARE!")
        return

    # Check if input is Numpy array type
    if type(A) != np.ndarray:
        print("INPUT ARRAY IS NOT NUMPY ARRAY")
        return

    n = aRows
    status = True

    if A[0, 0] <= 0:
        # If upper-left element is zero
        print("MATRIX MAY NOT BE POSITIVE DEFINITE! \nBREAKING... \n")
        status = False
    elif n == 2:
        # if 'A' is a 2x2 matrix
        det = det2(A)
        if det <= 0:
            # print("Matrix may NOT be positive definite!")
            print("MATRIX MAY NOT BE POSITIVE DEFINITE! \nBREAKING... \n")
            status = False
    elif n == 3:
        # if 'A' is a 3x3 matrix
        det = det3(A)
        if det <= 0:
            print("MATRIX MAY NOT BE POSITIVE DEFINITE! \nBREAKING... \n")
            status = False
    else:
        det1x1 = A[0, 0]
        det2x2 = det2(A)
        det3x3 = det3(A)
        if det1x1 <= 0 or det2x2 <= 0 or det3x3 <= 0:
            print("MATRIX MAY NOT BE POSITIVE DEFINITE! \nBREAKING... \n")
            status = False

    # Output answer
    # True: Matrix is positive-definite
    # False: Matrix is NOT positive-definite
    return status
