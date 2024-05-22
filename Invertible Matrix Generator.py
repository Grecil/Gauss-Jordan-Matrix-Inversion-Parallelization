import numpy as np
from scipy.linalg import inv

# Generate a random square matrix of order 600
A = np.random.rand(600, 600)

# Ensure the matrix is invertible by adding a multiple of the identity matrix
n = 600
identity = np.eye(n)
A = A + 2 * identity

# Check if the matrix is invertible
det = np.linalg.det(A)
if det == 0:
    print("Matrix is not invertible.")
else:
    print(f"Matrix is invertible with determinant: {det}")

    # Compute the inverse of the matrix
    A_inv = inv(A)

    # Save the matrix and its inverse to files
    np.savetxt("matrix.txt", A, delimiter=' ', fmt='%.3f')
    np.savetxt("inverse.txt", A_inv, delimiter=' ', fmt='%.6f')
    print("Matrix and its inverse saved to files 'matrix.txt' and 'inverse.txt'")
