from utilities import chol_params_to_lower_triangular_matrix
from utilities import cov_matrix_to_sdcorr_params
from utilities import cov_params_to_matrix
from utilities import robust_cholesky
from utilities import sdcorr_params_to_matrix
import numpy as np


def covariance_to_internal(external_values):
    """Do a cholesky reparametrization."""
    cov = cov_params_to_matrix(external_values)
    chol = robust_cholesky(cov)
    return chol[np.tril_indices(len(cov))]


def covariance_from_internal(internal_values):
    """Undo a cholesky reparametrization."""
    chol = chol_params_to_lower_triangular_matrix(internal_values)
    cov = chol @ chol.T
    return cov[np.tril_indices(len(chol))]


def sdcorr_to_internal(external_values):
    """Convert sdcorr to cov and do a cholesky reparametrization."""
    cov = sdcorr_params_to_matrix(external_values)
    chol = robust_cholesky(cov)
    return chol[np.tril_indices(len(cov))]


def sdcorr_from_internal(internal_values):
    """Undo a cholesky reparametrization."""
    chol = chol_params_to_lower_triangular_matrix(internal_values)
    cov = chol @ chol.T
    return cov_matrix_to_sdcorr_params(cov)


def probability_to_internal(external_values):
    """Reparametrize probability constrained parameters to internal."""
    return external_values / external_values[-1]


def probability_from_internal(internal_values):
    """Reparametrize probability constrained parameters from internal."""
    return internal_values / internal_values.sum()
