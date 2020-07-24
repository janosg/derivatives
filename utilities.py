import numpy as np
from scipy.sparse import csr_matrix



def chol_params_to_lower_triangular_matrix(params):
    dim = number_of_triangular_elements_to_dimension(len(params))
    mat = np.zeros((dim, dim))
    mat[np.tril_indices(dim)] = params
    return mat


def cov_params_to_matrix(cov_params):
    """Build covariance matrix from 1d array with its lower triangular elements.
    Args:
        cov_params (np.array): 1d array with the lower triangular elements of a
            covariance matrix (in C-order)
    Returns:
        cov (np.array): a covariance matrix
    """
    lower = chol_params_to_lower_triangular_matrix(cov_params)
    cov = lower + np.tril(lower, k=-1).T
    return cov


def cov_matrix_to_params(cov):
    return cov[np.tril_indices(len(cov))]


def sdcorr_params_to_sds_and_corr(sdcorr_params):
    dim = number_of_triangular_elements_to_dimension(len(sdcorr_params))
    sds = np.array(sdcorr_params[:dim])
    corr = np.eye(dim)
    corr[np.tril_indices(dim, k=-1)] = sdcorr_params[dim:]
    corr += np.tril(corr, k=-1).T
    return sds, corr


def sds_and_corr_to_cov(sds, corr):
    diag = np.diag(sds)
    return diag @ corr @ diag


def cov_to_sds_and_corr(cov):
    sds = np.sqrt(np.diagonal(cov))
    diag = np.diag(1 / sds)
    corr = diag @ cov @ diag
    return sds, corr


def sdcorr_params_to_matrix(sdcorr_params):
    """Build covariance matrix out of variances and correlations.
    Args:
        sdcorr_params (np.array): 1d array with parameters. The dimensions of the
            covariance matrix are inferred automatically. The first dim parameters
            are assumed to be the variances. The remainder are the lower triangular
            elements (excluding the diagonal) of a correlation matrix.
    Returns:
        cov (np.array): a covariance matrix
    """
    return sds_and_corr_to_cov(*sdcorr_params_to_sds_and_corr(sdcorr_params))


def cov_matrix_to_sdcorr_params(cov):
    dim = len(cov)
    sds, corr = cov_to_sds_and_corr(cov)
    correlations = corr[np.tril_indices(dim, k=-1)]
    return np.hstack([sds, correlations])


def number_of_triangular_elements_to_dimension(num):
    """Calculate the dimension of a square matrix from number of triangular elements.
    Args:
        num (int): The number of upper or lower triangular elements in the matrix.
    Examples:
        >>> number_of_triangular_elements_to_dimension(6)
        3
        >>> number_of_triangular_elements_to_dimension(10)
        4
    """
    return int(np.sqrt(8 * num + 1) / 2 - 0.5)


def dimension_to_number_of_triangular_elements(dim):
    """Calculate number of triangular elements from the dimension of a square matrix.
    Args:
        dim (int): Dimension of a square matrix.
    """
    return int(dim * (dim + 1) / 2)


def commutation_matrix(dim):
    row  = np.arange(dim ** 2)
    col  = row.reshape((dim, dim), order='F').ravel()
    
    data = np.ones(dim ** 2, dtype=np.int8)
    
    sparse_matrix = csr_matrix(
        (data, (row, col)), 
        shape=(dim ** 2, dim ** 2)
    )
    
    arr = sparse_matrix.toarray()
    return arr


def _unit_vector_or_zeros(index, size):
    """Return unit vector or vector of all zeroes."""
    u = np.zeros(size, int)
    if index != -1:
        u[index] = 1
    return u


def elimination_matrix(dim):
    """Construct (row-wise) elimination matrix.

    Let A be a quadratic matrix. Let vec(A) be the column-wise vectorization of A. Let
    vech(A) be the row-wise half-vectorization of A. Then the corresponding elimination
    matrix L is such that: L vec(A) = vech(A)

    Example:
    >>> import numpy as np
    >>> dim = 25
    >>> M = np.random.randn(dim, dim)
    >>> vecM = M.ravel('F')
    >>> vechM = M[np.tril_indices(dim)]
    >>> L = elimination_matrix(dim)
    >>> (L @ vecM == vechM).all()

    """
    n = dimension_to_number_of_triangular_elements(dim)
    
    M = np.zeros((dim, dim), int) - 1
    M[np.tril_indices(dim)] = np.arange(n, dtype=int)
    
    columns = [_unit_vector_or_zeros(i, n) for i in M.ravel('F')]
    
    elim = np.column_stack(columns)
    return elim


def transformation_matrix(dim):  # not needed right now
    n = dimension_to_number_of_triangular_elements(dim)
    M = np.zeros((dim, dim)) + np.nan
    M[np.diag_indices(dim)] = np.arange(dim, dtype=int)
    M[np.tril_indices(dim, k=-1)] = np.arange(dim, n, dtype=int)
    
    m = M.ravel('F')
    num_na = np.count_nonzero(np.isnan(m))
    indices = m.argsort()[:-num_na]

    rows = [_unit_vector_or_zeros(i, dim ** 2) for i in indices]
    
    transformer = np.row_stack(rows)
    return transformer


def duplication_matrix(dim):  # not needed right now
    n = dimension_to_number_of_triangular_elements(dim)
    M = np.zeros((dim, dim), dtype=int) - 1
    M[np.tril_indices(dim)] = np.arange(n)
    rows = [_unit_vector_or_zeros(i, n) for i in M.ravel('F')]
    D = np.row_stack(rows)
    return D