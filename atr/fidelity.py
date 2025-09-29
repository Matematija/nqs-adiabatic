from typing import Callable

from jax import numpy as jnp
from jax import Array

import equinox as eqx

from .operator import local_operator_expect
from .utils import vmap, abs2, filter_vmap
from .utils import Ansatz, Scalar


def _local_fidelity_factors(
    target_logpsi, target_samples, logpsi, x, logpsi_tt=None, *, chunk_size=None
):

    logpsi_at = vmap(logpsi, chunk_size=chunk_size)(target_samples)

    if logpsi_tt is None:
        logpsi_tt = vmap(target_logpsi, chunk_size=chunk_size)(target_samples)

    r1 = jnp.exp(logpsi_at - logpsi_tt)
    r2 = jnp.exp(target_logpsi(x) - logpsi(x))

    return r1, r2


def _cv_correction(r1, r2):
    return jnp.mean(abs2(r1)) * abs2(r2) - 1


_LogWaveFn = Callable[[Array], Scalar]


class LocalFidelity(eqx.Module):

    target_logpsi: _LogWaveFn
    target_samples: Array
    target_values: Array

    chunk_size: int | None = eqx.field(static=True)
    cv: bool = eqx.field(static=True)
    return_both: bool = eqx.field(static=True)

    def __init__(
        self,
        target_logpsi: _LogWaveFn,
        target_samples: Array,
        cv: bool = True,
        return_both: bool = False,
        *,
        chunk_size: int | None = None,
    ):

        self.target_logpsi = target_logpsi
        self.target_samples = target_samples
        self.target_values = vmap(target_logpsi, chunk_size=chunk_size)(target_samples)

        self.cv = cv
        self.return_both = return_both
        self.chunk_size = chunk_size

    def __call__(self, logpsi: Ansatz, x: Array) -> Array:

        r1, r2 = _local_fidelity_factors(
            self.target_logpsi,
            self.target_samples,
            logpsi,
            x,
            logpsi_tt=self.target_values,
            chunk_size=self.chunk_size,
        )

        F_loc = jnp.mean(r1) * r2

        if not self.cv:
            return F_loc

        F_loc_corr = F_loc - 0.5 * _cv_correction(r1, r2)

        if self.return_both:
            return (F_loc, F_loc_corr)
        else:
            return F_loc_corr


class Fidelity(eqx.Module):

    target_logpsi: _LogWaveFn
    target_samples: Array
    target_values: Array
    chunk_size: int | None = eqx.field(static=True)

    def __init__(
        self, target_logpsi: _LogWaveFn, target_samples: Array, *, chunk_size: int | None = None
    ):
        self.target_logpsi = target_logpsi
        self.target_samples = target_samples
        self.target_values = filter_vmap(target_logpsi, chunk_size=chunk_size)(target_samples)
        self.chunk_size = chunk_size

    def __call__(self, logpsi: Ansatz, samples: Array) -> Scalar:

        _factors = lambda x: _local_fidelity_factors(
            self.target_logpsi,
            self.target_samples,
            logpsi,
            x,
            logpsi_tt=self.target_values,
            chunk_size=self.chunk_size,
        )

        r1, r2 = vmap(_factors, chunk_size=self.chunk_size)(samples)
        F_loc = jnp.mean(r1) * r2
        # apply_cv = lambda F_loc: F_loc - 0.5 * _cv_correction(r1, r2)

        return local_operator_expect(F_loc, logpsi, samples, chunk_size=self.chunk_size)
