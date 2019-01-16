"""

LINEAR ALGEBRA PACKAGE
Self-made functions relating to linear algebra

Math 373, Intro. to Scientific Computing
Iowa State University
By: Michael Wrona

INCLUDED FUNCTIONS:
- Transpose
- Matrix/matrix and Matrix/vector multiplication
- Determinant
- Positive definite check (Sylvester's Crit, up to 3x3)
- LU decomposition
- Generate diagonally-dominant symmetric-positive-definite system of linear equations (for iterative solvers)
- Solve linear system via LU-decomposition
- Solve linear system via Gaussian elimination
- Solve linear system via Gauss-Seidel method (iterative)
- Vector norm, Vector p-norm

"""


import time
import numpy as np


def transpose(A):
    """
    TAKE TRANSPOSE OF MATRIX
    'A': Square matrix
    **NOTE: Input matrix must be NUMPY ARRAY
    >>> transpose(A)
    """

    def transSquare(a):
        """
        Take transpose of array
        """
        n = len(a)
        outTrans = np.zeros([n, n], dtype='float64')
        for i in range(n):
            for j in range(n):
                outTrans[i, j] = a[j, i]
        return outTrans
    
    def transNonSquare(a):
        """
        Take transpose of non-square array
        """
        rows, cols = np.shape(a)
        outTrans = np.zeros([cols, rows], dtype='float64')

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
    aRows, aCols = np.shape(a)		# Get dimensions of matrices
    bRows, bCols = np.shape(b)

    # Check that matrix dimensions agree for multiplication
    if (aCols != bRows):
        print("MATRIX DIMENSIONS DO NOT AGREE!!")
        return
    
    # Check if input is Numpy array type
    if type(a) != np.ndarray or type(b) != np.ndarray:
        print("INPUT ARRAY IS NOT NUMPY ARRAY")
        return
    
    # Compute matrix multiplication (ijk method)
    outMat = np.zeros([aRows, bCols], dtype='float64')	# Shape of output matrix
    for i in range(aRows):
	    for j in range(bCols):
		    sum = 0
		    for k in range(aCols):
			    sum += (a[i, k] * b[k, j])
		    outMat[i, j] = sum
    
    return outMat


def matVecMult(A, b):
    """
    MATRIX-VECTOR MULTIPLICATION
    Find the product from multiplying a matrix and vector
    **NOTE: INPUTS MUST BE NUMPY ARRAYS
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([2, 3])
    >>> matVecMult(a, b)
    >>> array([8, 18])
    """
    aRows, aCols = np.shape(A)      # Get dimensions
    bRows = len(b)

    if type(A) != np.ndarray or type(b) != np.ndarray:
        # Check for proper data type
        print("INPUT IS NOT NUMPY ARRAY IN ROUTINE \'matVecMult()\'")
        return
    
    if int(aCols) != int(bRows):
        # Check for compatable dimensions of matrix/vector
        print("INCOMPATABLE ARRAY DIMENSIONS IN ROUTINE \'matVecMult()\'")
        return
    
    output = np.zeros([aRows], dtype="float64")     # Create output array
    for i in range(aRows):
        output[i] = 0.0
        for j in range(aCols):
            output[i] += (A[i, j] * b[j])
    
    return output       # Return answer


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
    if (aRows != aCols):
        print("ARRAY \'A\' IS NOT SQUARE!")
        return
    
    # Check if input is Numpy array type
    if type(A) != np.ndarray:
        print("INPUT ARRAY IS NOT NUMPY ARRAY")
        return
    
    n = aRows   # Matrix dimension
    lu = np.zeros([n, n], dtype="float64")

    # LU Factorization
    for i in range(n):
        for j in range(i, n):
            x = 0.0
            for k in range(i):
                x += (lu[i, k] * lu[k, j])
            lu[i, j] = a[i, j] - x
        for j in range(i+1, n):
            x = 0.0
            for k in range(i):
                x += (lu[j, k] * lu[k, i])
            lu[j, i] = (1 / lu[i, i]) * (a[j, i] - x)

    # Calculate determinant from diagonal elements
    det = 1.0
    for i in range(n):
        det *= lu[i, i]
    
    return det      # Return solution


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
    if (aRows != aCols):
        print("ARRAY \'A\' IS NOT SQUARE!")
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
    return status

def luDecompose(A, b):
    """
    SOLVE LINEAR SYSTEM USING LU DECOMPOSITION
    Use LU decomposition to solve a linear system of equations.
    'A': Coefficient matrix [n, n]
    'b': Solution matrix [n, 1]
    ** NOTE: 'A' and 'b' MUST be NumPy ARRAYS:
    >>> A = np.array([n, n])    # Ex: np.array([[2, -1], [4, 3]])
    >>> b = np.array([n])       # Ex: np.array([2, -1])
    """
    # Check if input matrix is square
    aRows, aCols = np.shape(A)
    bRows, = np.shape(b)
    if (aRows != aCols):
        print("ARRAY \'A\' IS NOT SQUARE IN ROUTINE \'luDecompose()\'")
        return

    # Check if input is Numpy array type
    if type(A) != np.ndarray or type(b) != np.ndarray:
        print("INPUT ARRAY IS NOT NUMPY ARRAY")
        return
    
    n = aRows

    lower = np.zeros([n, n], dtype='float64')
    upper = np.zeros([n, n], dtype='float64')

    # DECOMPOSE MATRIX
    for i in range(n):
        # UPPER TRIANGULAR MATRIX
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (lower[i, j] * upper[j, k])
            upper[i, k] = A[i, k] - sum
        
        # LOWER TRIANGULAR MATRIX
        for k in range(i, n):
            if i == k:
                lower[i, i] = 1     # Diagonal element as 1
            else:
                sum = 0
                for j in range(i):
                    sum += (lower[k, j] * upper[j, i])
                lower[k, i] = (A[k, i] - sum) / upper[i, i]
    
    return lower, upper


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
        print("INPUT \'n\' TYPE MUST BE \'int\' IN ROUTINE \'genSPDSystem()\'")
        return
    
    Atemp = np.random.rand(n, n)    # Generate random coef. matrix 'A'
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
			    out += (Atemp[i, k] *Atrans[k, j]) + n
		    A[i, j] = out
    b = np.random.rand(n)       # Generate RHS vector 'b'
    return A, b


def lupSolve(A, b):

    n, m = np.shape(A)
    p = np.arange(0, n+1)     # Unit Permutation matrix
    temp = np.zeros([n], dtype='float64')
    x = np.zeros([n], dtype='float64')      # Solution

    for i in range(n):
        maxA = 0.0
        imax = i

        for k in range(i, n):
            absA = np.abs(A[k, i])
            if absA > maxA:
                maxA = absA
                imax = k
        # Break if matrix is degenerate
        if maxA < 1e-14:
            print("MATRIX IS DEGENERATE, BREAKING...")
            return
        
        # Pivoting Time...
        if imax != i:
            # pivoting p
            j = p[i]
            p[i] = p[imax]
            p[imax] = j

            # Pivot rows of A
            temp = A[i, :]
            A[i, :] = A[imax, :]
            A[imax, :] = temp

            # Count number of pivots, starting from n
            p[n] += 1
        
        for j in range(i+1, n):
            A[j, i] /= A[i, i]
            for k in range(i+1, n):
                A[j, k] -= (A[j, i] * A[i, k])
    
    # Solve system of equations
    for i in range(n):
        x[i] = b[p[i]]
        for k in range(i):
            x[i] -= (A[i, k] * x[k])
    for i in range(n-1, -1, -1):
        for k in range(i+1, n):
            x[i] -= (A[i, k] * x[k])
        x[i] = x[i] / A[i, i]
    
    return x


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
    if (aRows != aCols):
        print("ARRAY \'A\' IS NOT SQUARE!")
        return

    # Check if coef. matrix ad RHS vector have correct dimensions
    if (aCols != bRows):
        print("INCOMPATABLE INPUT ARRAY DIMENSIONS IN ROUTINE \'luSolve()\'")
        return

    # Check if input is Numpy array type
    if type(a) != np.ndarray or type(b) != np.ndarray:
        print("INPUT ARRAY IS NOT NUMPY ARRAY IN ROUTINE \'luSolve()\'")
        return

    n = aRows
    lu = np.zeros([n, n], dtype="float64")

	# LU Factorization
    for i in range(n):
        for j in range(i, n):
            x = 0.0
            for k in range(i):
                x += (lu[i, k] * lu[k, j])
            lu[i, j] = a[i, j] - x
        for j in range(i+1, n):
            x = 0.0
            for k in range(i):
                x += (lu[j, k] * lu[k, i])
            lu[j, i] = (1 / lu[i, i]) * (a[j, i] - x)

    # Find solution of Ly = b
    y = np.zeros([n], dtype="float64")
    for i in range(n):
        x = 0.0
        for k in range(i):
            x += (lu[i, k] * y[k])
        y[i] = b[i] - x

    # Find solution of Ux = y
    soln = np.zeros([n], dtype="float64")
    for i in range(n-1, -1, -1):
        x = 0.0
        for k in range(i+1, n):
            x += (lu[i, k] * soln[k])
        soln[i] = (1 / lu[i, i]) * (y[i] - x)

    return soln
    

def gaussElim(A, b):
    """
    Use Gaussian Elimination to solve a linear system
    of equations.
    'A': Coefficient matrix (n, n)
    'b': Solution matrix (n, 1)
    ** NOTE: 'A' and 'b' MUST be NumPy ARRAYS:
    >>> A = np.array([n, n])    # Ex: np.array([[2, -1], [4, 3]])
    >>> b = np.array([n])       # Ex: np.array([2, -1])
    Example C++ code from: www.geeksforgeeks.org/gaussian-elimination
    """
    class gaussElimination(object):
        def __init__(self, A, b, n):
            self.n = n      # Size of coefficient array

            # Array used in calculations
            self.mat = np.zeros([self.n, self.n + 1], dtype='float32')
            self.mat[:n, :n] = A   # Populate array with 'A'
            self.mat[:, n] = b     # Populate array with 'b'

            self.x = np.zeros(n, dtype='float32')    # Create solution array

        def swapRow(self, i, j):
            # Function for elementary operation of swapping two rows
            for k in range(self.n + 1):
                temp = self.mat[i, k]
                self.mat[i, k] = self.mat[j, k]
                self.mat[j, k] = temp
    
        def backSub(self):
            a = self.mat[:, :self.n]     # Coefficient array
            b = self.mat[:, (self.n)]    # Whatever this array is called
            # Solve for solution
            for i in range((self.n - 1), -1, -1):
                for j in range((i + 1), self.n):
                    b[i] -= (a[i, j] * self.x[j])
                self.x[i] = b[i] / a[i, i]

        def forwardElim(self):
            # Decompose array to upper-diagonal form
            for k in range(self.n):
                # Initialize maximum value and index for pivot
                i_max = k
                v_max = self.mat[i_max, k]

                # find greater amplitude for pivot if any
                for i in range((k + 1), self.n):
                    if (np.abs(self.mat[i, k]) > v_max):
                        v_max = self.mat[i, k]
                        i_max = i
                
                # if a prinicipal diagonal element is zero, 
                # it denotes that matrix is singular, and 
                # will lead to a division-by-zero later.
                if (self.mat[k, i_max] == 0):
                    print("MATRIX IS SINGULAR")
                    return
                
                # Swap the greatest value row with current row
                if (i_max != k):
                    self.swapRow(k, i_max)
                
                for i in range((k + 1), self.n):
                    # factor f to set current row kth elemnt to 0,
                    # and subsequently remaining kth column to 0
                    f = self.mat[i, k] / self.mat[k, k]
                    for j in range((k + 1), (self.n + 1)):
                        # subtract fth multiple of corresponding kth 
                        # row element
                        self.mat[i, j] -= (self.mat[k, j] * f)

                    # filling lower triangular matrix with zeros
                    self.mat[i, k] = 0
                # print(self.mat)
            # print(self.mat)
            return
        
        def gausElim(self):
            # Decompose matrix to upper diagonal form
            singularFlag = self.forwardElim()    # Check if matrix is singular

            # Check if coefficient array is singular
            if singularFlag != -1:
                print("CEOF. ARRAY IS SINGULAR IN ROUTINE \'gaussElim()\'")
                self.mat = 0
            
            self.backSub()   # Get solution to system
            return self.x


    # Check that input arrays are the correct shapes
    aRows, aCols = np.shape(A)
    bRows, = np.shape(b)

    if (aCols != bRows):
        print("INCOMPATABLE INPUT ARRAY DIMENSIONS IN ROUTINE \'gaussElim()\'")
        return
    if (aRows != aCols):
        print("ARRAY \'A\' IS NOT SQUARE!")
        return

    # Check if input is Numpy array type
    if type(A) != np.ndarray or type(b) != np.ndarray:
        print("INPUT ARRAY IS NOT NUMPY ARRAY IN ROUTINE \'gaussElim()\'")
        return
    
    # Create augmented array of 'A' and 'b'
    n = aRows       # Number of unknowns
    mat = np.zeros([n, n + 1], dtype='float32')
    mat[:n, :n] = A     # Populate array with 'A'
    mat[:, n] = b       # Populate array with 'b'

    aa = gaussElimination(A, b, n)
    soln = aa.gausElim()
    return soln


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

    tol = 1e-12          # Solution tolerance. Will break if reached
    maxIter = 12000      # Maximum number of iterations. Will break if reached
    iters = 1           # Iteration counter

    # Check that input arrays are the correct shapes
    aRows, aCols = np.shape(a)
    bRows, = np.shape(b)
    
    if (aCols != bRows):
        print("ARRAY DIMENSIONS DO NOT AGREE!")
        print("Shape of array \'A\' must have same number of rows \'b\' has")
        return 0
    if (aRows != aCols):
        print("ARRAY \'A\' IS NOT SQUARE!")
        return

    # Check if input is Numpy array type
    if type(a) != np.ndarray or type(b) != np.ndarray:
        print("INPUT ARRAY IS NOT NUMPY ARRAY")
        return
    
    # Check if coefficient matrix is positive definite
    posDefCheck = checkPosDef(a)
    if posDefCheck == False:
        return
    
    n = aRows                       # System dimensions
    x = np.random.rand(n,) * 5     # initial guess for solution
    sigmaTol = 1                    # Used to calculate tolerance

    while True:
        if iters > maxIter:
            print("MAX. ITERS. REACHED IN ROUTINE \'gaussSeidel()\'")
            break
        
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += (a[i, j] * x[j])
            x[i] = (b[i] - sigma) / a[i, i]
        
        tolerance = abs(sigmaTol - sigma)
        if tolerance <= tol:
            print("TOLERANCE MET IN ROUTINE \'gaussSeidel()\' \t %d ITERS." % (iters))
            break

        sigmaTol = sigma
        iters += 1

    return x    # Return Solution


def vectNorm(v):
    """ Calculate the norm of a vector (p=2) """
    n = len(v)
    out = 0.0
    for i in range(n):
        out += pow(abs(v[i]), 2)
    return np.sqrt(out)


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
        print("INPUT IS NOT NUMPY ARRAY IN ROUTINE \'vectNorm()\'")
        return
    
    n = len(x)
    
    # Raise each element to the power of 'p', find the sum of the elements
    sum = 0.0
    for i in range(n):
        sum += (abs(x[i])**p)
    
    return pow(sum, (1 / p))        # Raise sum to the power of 1/p


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
        aRows, aCols = np.shape(A)      # Get dimensions
        bRows = len(b)
        output = np.zeros([aRows], dtype="float64")     # Create output array
        for i in range(aRows):
            output[i] = 0.0
            for j in range(aCols):
                output[i] += (A[i, j] * b[j])
        return output       # Return answer
    
    if type(A) != np.ndarray:
        # Check data type
        print("INPUT IS NOT NUMPY ARRAY IN ROUTINE \'dominantEig()\'")
        return
    
    n, aCols = np.shape(A)
    if n != aCols:
        # Check if input is a square matrix
        print("INPUT ARRAY IS NOT SQUARE IN ROUTINE \'dominantEig()\'")
        return
    
    tol = 1e-12         # Solution tolerance. Will break if met
    maxIter = 12000     # Max. number of iterations. Will break if met
    iters = 1
    lamTol = 0.0        # For calculating convergence

    w = np.zeros([n], dtype="float64")
    v = np.random.randn(n)      # Start with random initial condition/guess with norm(v0) = 1
    v = v / norm(v)

    while True:
        w = matVecMult(A, v)
        v = w / norm(w)
        prod = matVecMult(A, v)

        lam = 0.0
        for i in range(n):
            lam += (v[i] * prod[i])
        
        if iters >= maxIter:
            # Break if max. iters. is reached
            print("MAX ITERS. REACHED IN ROUTINE \'dominantEig()\'")
            break
        
        if abs(lam - lamTol) <= tol:
            # Break if within tolerance
            print("TOLERANCE MET IN ROUTINE \'dominantEig()\'")
            break
        
        iters += 1
        lamTol = lam

    return lam


# # SOLUTION: [0.4444, 0.66667] (PD)
# a = np.array([[3, 1],
#             [3, 4]], dtype='float64')
# b = np.array([2, 4], dtype='float64')

# # SOLUTION: [5.5, 8, 4.5] (PD)
# a = np.array([[2, -1, 0],
#             [-1, 2, -1],
#             [0, -1, 2]], dtype='float64')
# b = np.array([3, 6, 1], dtype = 'float64')

# # SOLUTION: [3, 1, 2] (NOT PD)
# a = np.array([[3, 2, -4],
#             [2, 3, 3],
#             [5, -3, 1]], dtype='float64')
# b = np.array([3, 15, 14], dtype='float64')


# # SOLUTION: [3, -2, 0, 5] (NOT PD)
# a = np.array([[1, 2, 5, 1],
#             [3, -4, 3, -2],
#             [4, 3, 2, -1],
#             [1, -2, -4, -1]], dtype='float64')
# b = np.array([4, 7, 1, 2], dtype='float64')

# # SOLUTION: [1, -2, 3, 4, 2, -1] (NOT PD)
# a = np.array([[1, 1, -2, 1, 3, -1],
#             [2, -1, 1, 2, 1, -3],
#             [1, 3, -3, -1, 2, 1],
#             [5, 2, -1, -1, 2, 1],
#             [-3, -1, 2, 3, 1, 3],
#             [4, 3, 1, -6, -3, -2]], dtype='float64')
# b = np.array([4, 20, -15, -3, 16, -27], dtype='float64')



matSizes = [5, 10, 15, 20]

for k in matSizes:
    # a = np.random.rand(k, k) * 5
    # b = np.random.rand(k) * 6
    a, b = genSPDSystem(k)

    t0 = time.time()
    solution = gaussSeidel(a, b)
    t1 = time.time()
    
    print("MY CODE Solution time for %dx%d system: %0.6f s." % (k, k, (t1-t0)))

    t2 = time.time()
    solution2 = np.linalg.solve(a, b)
    t3 = time.time()
    print("NUMPY Solution time for %dx%d system: %0.6f s. \n" % (k, k, (t3-t2)))


