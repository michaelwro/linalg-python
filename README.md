# linalg-python
A collection of common linear algebra algorithms implemented in Python 3

**Michael Wrona** _Student, B.S. Aerospace Engineering, Iowa State University (Ames, IA)_

## Included Features/Algorithms
* **Transpose:** Perform the transpose of a 2D array.
* **Matrix Multiplication:** Use the *'ijk'* algorithm to calculate a matrix-matrix product.
* **Matrix-Vector Product:** Calculate the matrix-vector product.
* **Determinant:** Use *LU-Factorization* to calculate the determinant of a matrix.
* **Positive-Definite Check:** Use *'Sylvester's Criterion'* (up to 3x3 determinant) to get a ***rough*** guess if a matrix is positive-definite.
* **LU-Factorization:** Calculate the *'LU-factorization'* of a square matrix.
* **Generate Symmetric Positive-Definite System:** Generate a dense SPD system of linear equations. Typically used to test iterative solvers.
* **Linear System of Equations Solvers:**
    * **LU-Solver:** Solve linear system via *'LU-factorization.'*
    * **Gaussian Elimination**
    * **Gauss-Seidel:** An iterative linear system solver.
* **Vector Norm:** Calculate the norm of a vector, including the p-norm (any p).
* **Dominant Eigenvalue:** Use *'power iteration'* to find the dominant (largest) eigenvalue of a square matrix.

## Required Packages
I aimed not to use operations beyond PEMDAS, square roots, and trig. to be the most readable and easiest to translate into other languages. I did use NumPy arrays for creating vectors/matrices.
* NumPy `pip install numpy`

