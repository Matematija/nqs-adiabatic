import abc
from typing import Callable, NamedTuple, Sequence

import jax
from jax import numpy as jnp
from jax import Array

import equinox as eqx

from ..utils import is_real, Scalar

LogWaveFn = Callable[[Array], Array]


class MatrixElements(NamedTuple):

    conn: Array
    mels: Array

    def __len__(self):
        return len(self.mels)


class Operator(eqx.Module):

    attrs: dict

    def __init__(self, **attrs):
        self.attrs = attrs

    def __getattr__(self, name):
        if name in self.attrs:
            return self.attrs[name]
        else:
            return getattr(super(), name)

    def __mul__(self, scalar: Scalar) -> "Operator":
        return LinearCombination([self], [scalar], **self.attrs)

    def __neg__(self) -> "Operator":
        return LinearCombination([self], [-1.0], **self.attrs)

    def __add__(self, other: "Operator") -> "Operator":
        return LinearCombination([self, other], [1.0, 1.0], **self.attrs)

    def __rmul__(self, scalar: Scalar) -> "Operator":
        return self.__mul__(scalar)

    def __sub__(self, other: "Operator") -> "Operator":
        return self + (-other)

    @abc.abstractmethod
    def conn(self, x: Array) -> MatrixElements:
        pass

    @property
    def is_hermitian(self):
        return False


class LinearCombination(Operator):

    operators: Sequence[Operator]
    coeffs: Array

    def __init__(self, operators: Sequence[Operator], coeffs: Array, **attrs):
        super().__init__(**attrs)
        self.operators = operators
        self.coeffs = jnp.asarray(coeffs)

    def __mul__(self, scalar: Scalar):
        return LinearCombination(self.operators, self.coeffs * scalar, **self.attrs)

    def __neg__(self):
        return LinearCombination(self.operators, -self.coeffs, **self.attrs)

    def __add__(self, other: Operator):

        if isinstance(other, LinearCombination):
            operators = self.operators + other.operators
            coeffs = jnp.concatenate([self.coeffs, other.coeffs])
        else:
            operators = self.operators + [other]
            coeffs = jnp.concatenate([self.coeffs, jnp.array([1.0])])

        attrs = dict(self.attrs, **other.attrs)
        return LinearCombination(operators, coeffs, **attrs)

    def conn(self, x: Array) -> MatrixElements:

        conn, mels = [], []

        for i, op in enumerate(self.operators):
            conn_, mels_ = op.conn(x)
            conn.append(conn_)
            mels.append(self.coeffs[i] * mels_)

        return jnp.concatenate(conn), jnp.concatenate(mels)

    @property
    def is_hermitian(self):
        return all(op.is_hermitian for op in self.operators) & is_real(self.coeffs)


class WrappedOperator(Operator):

    operator: Operator

    def __init__(self, operator: Operator):
        super().__init__()
        self.operator = operator

    def __getattr__(self, name: str):
        return getattr(self.operator, name)

    def __mul__(self, scalar: Scalar):
        return self.operator.__mul__(scalar)

    def __neg__(self):
        return self.operator.__neg__()

    def __add__(self, other: "Operator"):
        return self.operator.__add__(other)

    def __sub__(self, other: "Operator"):
        return self.operator.__sub__(other)

    def conn(self, x: Array) -> MatrixElements:
        return self.operator.conn(x)

    @property
    def is_hermitian(self):
        return self.operator.is_hermitian
