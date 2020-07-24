"""Derivatives.

Remarks on the mathematical notation:

Let X denote the Cholesky factor of some covariance matrix S. I.e., X X^\top = S.
We write vec(A) for the (column-wise) vectorization of the matrix A and we write
vech(A) for the (row-wise) half vectorization of A. We denote the elimination matrix by
L, which fulfills L vec(A) = vech(A). For symmetric matrices A we define the duplication
matrix D, which fulfills D vech(A) = vec(A) and for lower-triangular matrices we define
the "lower-triangular" duplication matrix which again fulfills D vech(A) = vec(A), but
is not to be confused with the standard duplication matrix. In fact, for the
"lower-triangular" case we have D = L^\top. We denote the kronecker product by \otimes.

"""
import itertools
import numpy as np

from utilities import chol_params_to_lower_triangular_matrix
from utilities import cov_matrix_to_sdcorr_params
from utilities import number_of_triangular_elements_to_dimension
from utilities import dimension_to_number_of_triangular_elements
from utilities import cov_params_to_matrix

from utilities import commutation_matrix
from utilities import elimination_matrix
from utilities import duplication_matrix
from utilities import transformation_matrix

from estimagic.optimization.utilities import robust_cholesky


def derivative_covariance_from_internal(internal):
    """Derivative of ``covariance_from_internal``.

    The derivative can be found using matrix calculus. See for example 'Matrix
    Differential Calculus with Applications in Statistics and Econometrics' by Magnus
    and Neudecker. The following result is motivated by https://tinyurl.com/y4pbfxst,
    which is shortly presented again here. For notation see the explaination at the
    beginning of the script.

    .. math::

        \operatorname{vech}(X) = \text{internal} \\
        
        J' := 
        \frac{
            \mathrm{d} \operatorname{vec} ( S )}{
            \mathrm{d} \operatorname{vec} X
            } = ( I + K ) (X \otimes I) \\
        
        J = 
        \frac{
            \mathrm{d} \operatorname{vech} ( S )}{
            \mathrm{d} \operatorname{vech} X
            }
          = L J' D

    Args:
        internal (np.ndarray): Cholesky factors stored in an "internal" format.

    Returns:
        deriv: The Jacobian matrix.

    """
    chol = chol_params_to_lower_triangular_matrix(internal)
    dim = len(chol)
    
    K = commutation_matrix(dim)
    L = elimination_matrix(dim)
    
    left = np.eye(dim ** 2) + K
    right = np.kron(chol, np.eye(dim))

    intermediate = left @ right  # J'
    
    deriv = L @ intermediate @ L.T  # J
    return deriv


def derivative_covariance_to_internal(external):
    """Derivative of ``covariance_to_internal``.

    The derivative can be found using matrix calculus. See for example 'Matrix
    Differential Calculus with Applications in Statistics and Econometrics' by Magnus
    and Neudecker. The following result is motivated by https://tinyurl.com/y4pbfxst,
    which is shortly presented again here. For notation see the explaination at the
    beginning of the script.

    .. math::

        \operatorname{vech}(S) = \text{external} \\
        
        J = 
        \frac{
            \mathrm{d} \operatorname{vech} ( X )}{
            \mathrm{d} \operatorname{vech} S
            } = 
        ( \frac{
            \mathrm{d} \operatorname{vech} ( S )}{
            \mathrm{d} \operatorname{vech} X
            } )^{-1}
        = (``derivative_covariance_from_internal``(\operatorname{vech}(X))) ^{-1}

    Args:
        internal (np.ndarray): Cholesky factors stored in an "internal" format.

    Returns:
        deriv: The Jacobian matrix.

    """
    cov = cov_params_to_matrix(external)
    chol = robust_cholesky(cov)

    internal = chol[np.tril_indices(len(chol))]

    deriv = derivative_covariance_from_internal(internal)
    deriv = np.linalg.inv(deriv)
    return deriv



def derivative_probability_from_internal(internal):
    """Derivative of ``probability_from_internal``.
    The derivative can be found using matrix calculus. See for example 'Matrix
    Differential Calculus with Applications in Statistics and Econometrics' by Magnus
    and Neudecker. For notation see the explaination at the beginning of the script.

    .. math::

        1 := (1, \dots, 1)^\top \\
        I_m := \text{m dimensional Identity matrix} \\

        x = \text{internal} \\
        f: \mathbb{R}^m \to \mathbb{R}^m, x \mapsto \frac{1}{x^\top 1} x \\
        
        J(f)(x) = \frac{1}{\sigma} I_m - \frac{1}{\sigma^2} 1 x^\top

    Args:
        internal (np.ndarray): Internal (positive) values.

    Returns:
        deriv: The Jacobian matrix.

    """
    dim = len(internal)
    
    sigma = np.sum(internal)
    left = np.eye(dim)
    right = np.ones((dim, dim)) * (internal / sigma)
    
    deriv = (left - right.T) / sigma
    return deriv


def derivative_probability_to_internal(external):
    """Derivative of ``probability_to_internal``.

    The derivative can be found using matrix calculus. See for example 'Matrix
    Differential Calculus with Applications in Statistics and Econometrics' by Magnus
    and Neudecker. For notation see the explaination at the beginning of the script.

    .. math::

        e_k := \text{standard basis vector} \\
        x := \text{external} \\
        f: \mathbb{R}^m \to \mathbb{R}^m, x \mapsto \frac{1}{x_m} x \\

        J(f)(x) = 
        \frac{1}{x_m} \sum_{k=1}^{m-1} e_k e_k^\top - 
        \frac{1}{x_m^2}  [
            0, \dots, 0, \left ( \begin{matrix} x_{1:m-1} \\ 0 \end{matrix} \right ) 
            ]

    Args:
        external (np.ndarray): Array of probabilities; sums to one.

    Returns:
        deriv: The Jacobian matrix.

    """
    dim = len(external)
    
    deriv = np.eye(dim) / external[-1]
    deriv[:, -1] -= external / (external[-1] ** 2)
    deriv[-1, -1] = 0
    
    return deriv