"""

EXAMPLE

VECTORNORMP: linalg-python
Calculate the p-norm of a vector

By: Michael Wrona
Student, B.S. Aerospace Engineering
Iowa State University (Ames, IA)

"""


import numpy as np


def vectNormP(x, p):
    """
    CALCULATE THE 'p' NORM OF A VECTOR
    'x': Vector to calculate the norm of
    'p': TAKEN AS 2 NORMALLY (p-norm), no pun intended ;)
    **NOTE: 'x' MUST BE NUMPY ARRAY
    >>> x = np.array([1, 2, 3])
    >>> vectNorm(x, 1)
    >>> 6
    """
    # Check if input is Numpy array
    if type(x) != np.ndarray:
        print("INPUT IS NOT NUMPY ARRAY IN ROUTINE 'vectNorm()'")
        return

    n = len(x)

    # Raise each element to the power of 'p', find the sum of the elements
    sum = 0.0
    for i in range(n):
        sum += abs(x[i]) ** p

    return pow(sum, (1 / p))  # Raise sum to the power of 1/p


# Define vector
vec = np.array([2, 3, 4, 5, 6])

# Calculate norm (p = 2)
norm = vectNormP(vec, 2)

# Solution: 9.4868329805051
print("Vector norm: %0.6f" % (norm))
