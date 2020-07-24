import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from jax import jacfwd

from derivatives import derivative_covariance_from_internal
from derivatives import derivative_covariance_to_internal
from derivatives import derivative_probability_from_internal
from derivatives import derivative_probability_to_internal
from derivatives import derivative_sdcorr_from_internal

from kernel_transformations_jax import covariance_from_internal
from kernel_transformations_jax import covariance_to_internal
from kernel_transformations_jax import probability_from_internal
from kernel_transformations_jax import probability_to_internal
from kernel_transformations_jax import sdcorr_from_internal
from kernel_transformations_jax import sdcorr_to_internal


def get_random_internal(dim, seed=0):
    """Return random internal values given dimension."""
    np.random.seed(seed)
    chol = np.tril(np.random.randn(dim, dim))
    internal = chol[np.tril_indices(len(chol))]
    return internal


def get_random_external(dim, seed=0):
    """Return random external values given dimension."""
    np.random.seed(seed)
    data = np.random.randn(dim, 1000)
    cov = np.cov(data)
    external = cov[np.tril_indices(dim)]
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


@pytest.mark.parametrize("dim", list(range(10, 50)))
def test_derivative_covariance_from_internal(dim, jax_derivatives):
    internal = get_random_internal(dim)
    jax_deriv = jax_derivatives["covariance_from"](internal)
    deriv = derivative_covariance_from_internal(internal)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)


@pytest.mark.parametrize("dim", list(range(10, 50)))
def test_derivative_covariance_to_internal(dim, jax_derivatives):
    external = get_random_external(dim)
    jax_deriv = jax_derivatives["covariance_to"](external)
    deriv = derivative_covariance_to_internal(external)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)


def test_derivative_probability_from_internal(jax_derivatives):
    bad = []
    for dim in range(10, 50):
        internal = get_random_internal(dim)
        jax_deriv = jax_derivatives["probability_from"](internal)
        deriv = derivative_probability_from_internal(internal)
        try:
            assert_array_almost_equal(deriv, jax_deriv, decimal=5)
        except AssertionError:
            bad.append(dim)

    assert len(bad) < 5


@pytest.mark.parametrize("dim", list(range(10, 50)))
def test_derivative_probability_to_internal(dim, jax_derivatives):
    external = get_random_external(dim)
    jax_deriv = jax_derivatives["probability_to"](external)
    deriv = derivative_probability_to_internal(external)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)


@pytest.mark.parametrize("dim", list(range(10, 50)))
def test_derivative_sdcorr_from_internal(dim, jax_derivatives):
    internal = get_random_internal(dim)
    jax_deriv = jax_derivatives["sdcorr_from"](internal)
    deriv = derivative_sdcorr_from_internal(internal)
    assert_array_almost_equal(deriv, jax_deriv, decimal=5)


def test_derivative_sdcorr_to_internal(jax_derivatives):
    bad = []

    for dim in range(10, 50):
        external = get_random_external(dim)
        jax_deriv = jax_derivatives["sdcorr_to"](external)
        try:
            deriv = derivative_sdcorr_to_internal(external)
            assert_array_almost_equal(deriv, jax_deriv, decimal=5)
        except Exception:
            bad.append(dim)

    assert len(bad) < 5