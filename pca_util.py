from numpy import mean, zeros, dot, cov, hstack, delete, linspace
from numpy import float64, float32
from numpy.linalg import eig

"""
Author: Marco Treglia
email: marcotreglia1@gmail.com
-- PCA Utilis --
"""

def axis_mean(data, respect, log=False, precision=False):
    """
        Define the mean of the feature per row or coulum (default option is rows)

        Arguments:
            data    --  target data
            respect --  'row': compute the mean respect the rows
                        'col': compute respect the column
            log     --  show dimensions of the data
            precision -- false: float32
                         true:  float64

        Return:
            mean -- mean of the features
    """
    row = data.shape[0]
    col = data.shape[1]

    # Show info
    if log:
        print('Shape: {} rows and {} column. '.format(row, col))

    # Precision
    p = float64 if precision is True else float32

    # Rows case
    if respect is 'row':
        axis = 1
        dim = row

    # Coulums case
    if respect is 'col':
        axis = 0
        dim = col

    _mean = mean(data.astype(p), axis = axis, dtype= p).reshape(dim, 1)

    if respect is 'col':
        _mean = _mean.T

    return _mean




def scatter_matrix(data):
    """
    Scatter matrix fuction.

    Arguments:
        data -- target data

    Returns:
        s_matrix -- Scatter matrix
    """

    # Mean respect rows or column
    _mean = axis_mean(data, 'row')

    # Scatter matrix
    s_matrix = dot(data - m, (data - m).T)

    return s_matrix



def covariance_matrix(data):
    """
    Scatter matrix fuction.

    Arguments:
        data -- target data

    Returns:
        cov_matrix -- Covariance matrix
    """

    cov_matrix = cov(data)

    return cov_matrix




def eigen(matrix):
    """
    EigenValues and EigenVector

    Arguments:
        matrix -- target matrix (covariance or scatter matrix)

    Returns:
        eig_value  -- EigenValues
        eig_vector -- EigenVector
    """
    dim = matrix.shape[0]

    e_val, e_vec = eig(matrix)

    return  e_val, e_vec



def trasformation_matrix(e_vec, components):
    """
    Create the trasformation matrix for reduce the dimensionality respect
    the number of components

    Arguments:
        eig_vector -- EigenVector
        components -- dimensionality of the output

    Returns:
        t_matrix -- transformation matrix
    """

    dim = e_vec.shape[1]
    assert dim >= components, 'The components are more then the dimensionality'

    # Vector to delete index
    vect_to_delete = linspace(components, dim - 1, dim - components, dtype=int, endpoint=True)

    t_matrix = delete(e_vec, vect_to_delete, axis=1)

    return t_matrix


def pca(data, components):
    """
        Reduce dimensionality with the principal components analisys

        Arguments:
            data -- target data
            components -- dimensionality of the output

        Returns:
            reduced_matrix -- data projected into a lower dimensionality
    """

    # Covariace matrix
    cov_matrix = covariance_matrix(data)
    # Eigen
    e_val, e_vec = eigen(cov_matrix)
    # Trasformation matrix
    t_matrix = trasformation_matrix(e_vec, components)
    # Mean vector
    mean_vec = axis_mean(data, 'row')
    # Center the data
    data = data - mean_vec
    # Reduced dimensionality matrix
    reduced_matrix = dot(t_matrix.T, data)

    return reduced_matrix
