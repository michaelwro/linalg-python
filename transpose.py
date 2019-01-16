"""

TRANSPOSE: linalg-python
Return the transpose of a 2D array.

By: Michael Wrona
Student, B.S. Aerospace Engineering
Iowa State University (Ames, IA)

"""


import numpy as np


def transpose(A):
    """
    TAKE TRANSPOSE OF MATRIX
    'A': Square/rectangular matrix
    **NOTE: Input matrix must be NUMPY ARRAY
    >>> A = np.array([[1, 2], [3, 4]])
    >>> transpose(A)
    >>> array([[1, 3], [2, 4]])
    """

    def transSquare(a):
        """
        Take transpose of array
        """
        n = len(a)
        outTrans = np.zeros([n, n], dtype="float64")
        for i in range(n):
            for j in range(n):
                outTrans[i, j] = a[j, i]
        return outTrans

    def transNonSquare(a):
        """
        Take transpose of non-square array
        """
        rows, cols = np.shape(a)
        outTrans = np.zeros([cols, rows], dtype="float64")

        for i in range(rows):
            for j in range(cols):
                outTrans[j, i] = a[i, j]
        return outTrans

    # Check if input is Numpy array type
    if type(A) != np.ndarray:
        print("INPUT ARRAY IS NOT NUMPY ARRAY")
        return

    # Get shape of input
    rows, cols = np.shape(A)

    if rows == cols:
        # If input is square matrix
        trans = transSquare(A)
    else:
        # If input is non-square matrix
        trans = transNonSquare(A)

    return trans
