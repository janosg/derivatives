"""Derivatives.

Remarks on the mathematical notation:
-------------------------------------

Let X denote the Cholesky factor of some covariance matrix S. I.e. X X^\top = S.
We write vec(A) for the (column-wise) vectorization of the matrix A and we write
vech(A) for the (row-wise) half vectorization of A. We denote the elimination
matrix by L, which fulfills L vec(A) = vech(A). For symmetric matrices A we
define the duplication matrix D, which fulfills D vech(A) = vec(A) and for
lower-triangular matrices we define the "lower-triangular" duplication matrix
which again fulfills D vech(A) = vec(A), but is not to be confused with the
standard duplication matrix. In fact, for the "lower-triangular" case we have
D = L^\top. We denote the kronecker product by \otimes.

Remarks on reference literature:
--------------------------------

The solutions on how to compute the derivatives implemented here can be found
using matrix calculus. See for example 'Matrix Differential Calculus with
Applications in Statistics and Econometrics' by Magnus and Neudecker.

"""
import numpy as np

from utilities import chol_params_to_lower_triangular_matrix
from utilities import cov_params_to_matrix
from utilities import sdcorr_params_to_matrix

from utilities import commutation_matrix
from utilities import elimination_matrix
from utilities import duplication_matrix
from utilities import transformation_matrix

from estimagic.optimization.utilities import robust_cholesky


def derivative_covariance_from_internal(internal):
    """Derivative of ``covariance_from_internal``.

    The following result is motivated by https://tinyurl.com/y4pbfxst, which is
    shortly presented again here. For notation see the explaination at the
    beginning of the module.

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

    intermediate = left @ right

    deriv = L @ intermediate @ L.T
    return deriv


def derivative_covariance_to_internal(external):
    """Derivative of ``covariance_to_internal``.

    The following result is motivated by https://tinyurl.com/y4pbfxst, which is
    shortly presented again here. For notation see the explaination at the
    beginning of the module.

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
        external (np.ndarray): Row-wise half-vectorized covariance matrix.

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

    .. math::

        e_k := \text{standard basis vector} \\
        x := \text{external} \\
        f: \mathbb{R}^m \to \mathbb{R}^m, x \mapsto \frac{1}{x_m} x \\

        J(f)(x) = 
        \frac{1}{x_m} \sum_{k=1}^{m-1} e_k e_k^\top - 
        \frac{1}{x_m^2}  [
            0, 
            \dots,
            0,
            \left ( \begin{matrix} x_{1:m-1} \\ 0 \end{matrix} \right ) 
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


def derivative_sdcorr_from_internal(internal):
    """Derivative of ``sdcorr_from_internal``.

    The following result is motivated by https://tinyurl.com/y6ytlyd9; however
    since the question was formulated with an error the result here is adjusted
    slightly. In particular, in the answer by user 'greg', the matrix A should
    have been defined as A = diag(norm(x_1), ..., norm(x_n)), where x_i denotes
    the i-th row of X (the Cholesky factor). For notation see the explaination
    at the beginning of the module.

    ============================================================================

    Explaination on the result.
    ---------------------------

    We want to differentiate the graph

      internal --> cholesky --> cov --> corr-mat --> mod. corr-mat --> external
    
    where mod. corr-mat denotes the modified correlation matrix which has the
    standard deviations stored on its diagonal. Let x := internal and
    p := external. Then we want to compute the quantity (d p / d x). As before
    we consider an intermediate result first. Namely we define A as above,
    V := inverse(A) and P := V S V + A - I. The attentive reader might now
    notice that P is the modified correlation matrix. At last we write
    x' := vec(X) and p' := vec(P). Using the result stated in the tinyurl above,
    adjusted for the different matrix A, we can compute the quantity (d p'/d x')
    
    Finally, since we can define transformation matrices T and L to get p = T p'
    and x = L x' (where L denotes the elimination matrix with corresponding
    duplication matrix D), we can get our final result as

                        d p / d x = T (d p' / d x' ) D .

    Args:
        internal (np.ndarray): Cholesky factors stored in an "internal" format.

    Returns:
        deriv: The Jacobian matrix.

    """
    X = chol_params_to_lower_triangular_matrix(internal)
    dim = len(X)

    I = np.eye(dim)
    S = X @ X.T

    #  the wrong formulation in the tinyurl stated: A = np.multiply(I, X)
    A = np.sqrt(np.multiply(I, S))

    V = np.linalg.inv(A)

    K = commutation_matrix(dim)
    Y = np.diag(I.ravel("F"))

    #  with the wrong formulation in the tinyurl we would have had U = Y
    norms = np.sqrt((X ** 2).sum(axis=1).reshape(-1, 1))
    XX = X / norms
    U = Y @ np.kron(I, XX) @ K

    N = np.kron(I, X) @ K + np.kron(X, I)

    VS = V @ S
    B = np.kron(V, V)
    H = np.kron(VS, I)
    J = np.kron(I, VS)

    intermediate = U + B @ N - (H + J) @ B @ U

    T = transformation_matrix(dim)
    D = duplication_matrix(dim)

    deriv = T @ intermediate @ D
    return deriv


def derivative_sdcorr_to_internal(external):
    """Derivative of ``sdcorr_to_internal``.

    Args:
        external (np.ndarray): Row-wise half-vectorized modified correlation
            matrix.

    Returns:
        deriv: The Jacobian matrix.

    """
    cov = sdcorr_params_to_matrix(external)
    chol = robust_cholesky(cov)

    internal = chol[np.tril_indices(len(chol))]

    deriv = derivative_sdcorr_from_internal(internal)
    deriv = np.linalg.inv(deriv)
    return deriv
