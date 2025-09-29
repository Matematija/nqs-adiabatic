from typing import Any

import jax
from jax import numpy as jnp
from jax import Array

import equinox as eqx

from .base import Operator
from ..utils import Scalar, Ansatz


class _WithOperator(eqx.Module):

    operator: Operator
    logpsi: Ansatz

    def __call__(self, x: Array, *args: Any) -> Scalar:
        y, Oxy = self.operator.conn(x)
        logpsi_y = jax.vmap(lambda x: self.logpsi(x, *args))(y)
        return jax.nn.logsumexp(logpsi_y, b=Oxy)


def apply_operator(operator: Operator, logpsi: Ansatz) -> _WithOperator:
    return _WithOperator(operator, logpsi)


def eval_local_operator(operator: Operator, logpsi: Ansatz, x: Array, *args: Any) -> Scalar:
    y, Oxy = operator.conn(x)
    logpsi_y = jax.vmap(lambda x: logpsi(x, *args))(y)
    return Oxy @ jnp.exp(logpsi_y - logpsi(x, *args))


class LocalOperator(eqx.Module):

    operator: Operator
    logpsi: Ansatz

    def __call__(self, x: Array, *args: Any) -> Scalar:
        return eval_local_operator(self.operator, self.logpsi, x, *args)
