import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from jax import jacfwd

from utilities import cov_matrix_to_sdcorr_params

from derivatives import jacobian_covariance_from_internal
from derivatives import jacobian_covariance_to_internal
from derivatives import jacobian_probability_from_internal
from derivatives import jacobian_probability_to_internal
from derivatives import jacobian_sdcorr_from_internal
from derivatives import jacobian_sdcorr_to_internal

from kernel_transformations_jax import covariance_from_internal
from kernel_transformations_jax import covariance_to_internal
from kernel_transformations_jax import probability_from_internal
from kernel_transformations_jax import probability_to_internal
from kernel_transformations_jax import sdcorr_from_internal
from kernel_transformations_jax import sdcorr_to_internal


DIMENSIONS = list(range(10, 30))
SEEDS = list(range(5))


def get_internal_cholesky(dim, seed=0):
    """Return random internal cholesky values given dimension."""
    np.random.seed(seed)
    chol = np.tril(np.random.randn(dim, dim))
    internal = chol[np.tril_indices(len(chol))]
    return internal


def get_external_covariance(dim, seed=0):
    """Return random external covariance values given dimension."""
    np.random.seed(seed)
    data = np.random.randn(dim, 1000)
    cov = np.cov(data)
    external = cov[np.tril_indices(dim)]
    return external


def get_internal_probability(dim, seed=0):
    """Return random internal positive values given dimension."""
    np.random.seed(seed)
    internal = np.random.uniform(size=dim)
    return internal


def get_external_probability(dim, seed=0):
    """Return random internal positive values that sum to one."""
    internal = get_internal_probability(dim, seed)
    external = internal / internal.sum()
    return external


def get_external_sdcorr(dim, seed=0):
    """Return random external sdcorr values given dimension."""
    np.random.seed(seed)
    X = np.random.randn(dim, 1000)
    cov = np.cov(X)
    external = cov_matrix_to_sdcorr_params(cov)
    return external


@pytest.fixture
def jax_derivatives():
    out = {
        "covariance_from": jacfwd(covariance_from_internal),
        "covariance_to": jacfwd(covariance_to_internal),
        "probability_from": jacfwd(probability_from_internal),
        "probability_to": jacfwd(probability_to_internal),
        "sdcorr_from": jacfwd(sdcorr_from_internal),
        "sdcorr_to": jacfwd(sdcorr_to_internal),
    }
    return out


@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_derivative_covariance_from_internal(dim, seed, jax_derivatives):
    internal = get_internal_cholesky(dim)
    jax_deriv = jax_derivatives["covariance_from"](internal)
    deriv = jacobian_covariance_from_internal(internal)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)


@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_derivative_covariance_to_internal(dim, seed, jax_derivatives):
    external = get_external_covariance(dim)
    jax_deriv = jax_derivatives["covariance_to"](external)
    deriv = jacobian_covariance_to_internal(external)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)


@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_derivative_probability_from_internal(dim, seed, jax_derivatives):
    internal = get_internal_probability(dim)
    jax_deriv = jax_derivatives["probability_from"](internal)
    deriv = jacobian_probability_from_internal(internal)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)


@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_derivative_probability_to_internal(dim, seed, jax_derivatives):
    external = get_external_probability(dim)
    jax_deriv = jax_derivatives["probability_to"](external)
    deriv = jacobian_probability_to_internal(external)
    assert_array_almost_equal(deriv, jax_deriv, decimal=3)


@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_derivative_sdcorr_from_internal(dim, seed, jax_derivatives):
    internal = get_internal_cholesky(dim)
    jax_deriv = jax_derivatives["sdcorr_from"](internal)
    deriv = jacobian_sdcorr_from_internal(internal)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)


@pytest.mark.parametrize("dim", DIMENSIONS)
@pytest.mark.parametrize("seed", SEEDS)
def test_derivative_sdcorr_to_internal(dim, seed, jax_derivatives):
    external = get_external_sdcorr(dim)
    jax_deriv = jax_derivatives["sdcorr_to"](external)
    deriv = jacobian_sdcorr_to_internal(external)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)
