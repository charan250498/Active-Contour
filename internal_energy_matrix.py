import numpy as np
from scipy import sparse

def get_matrix(alpha, beta, gamma, num_points):
    """Return the matrix for the internal energy minimization.
    # Arguments
        alpha: The alpha parameter.
        beta: The beta parameter.
        gamma: The gamma parameter.
        num_points: The number of points in the curve.
    # Returns
        The matrix for the internal energy minimization. (i.e. A + gamma * I)
    """
    row_of_A = np.array([2*alpha+6*beta, (-1)*(alpha+4*beta), beta])
    row_of_A = np.concatenate((row_of_A, np.array([0]*(num_points-5))))
    row_of_A = np.concatenate((row_of_A, np.array([beta, (-1)*(alpha+4*beta)])))

    A = np.copy(row_of_A)
    for i in range(1, num_points):
        A = np.vstack((A, np.roll(row_of_A, i)))

    M = A + gamma*np.eye(num_points)

    #return sparse.linalg.inv(sparse.csr_matrix(M))
    return sparse.linalg.spsolve(sparse.csr_matrix(M), np.eye(num_points))
