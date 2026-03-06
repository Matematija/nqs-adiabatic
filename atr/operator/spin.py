from functools import partial
from math import prod

import jax
from jax import lax
from jax import numpy as jnp
from jax import Array

import equinox as eqx

from ..graph import Graph
from .base import Operator, MatrixElements
from ..utils import vmap, default_complex_dtype, default_int_dtype, Scalar, Shape, DType


class SpinOperator(Operator):
    def __init__(self, graph: Graph):
        super().__init__(graph=graph)


@eqx.filter_jit
def spin_basis(shape: Shape | int, dtype: DType | None = None):

    if isinstance(shape, int):
        shape = (shape,)

    if dtype is None:
        dtype = default_int_dtype()

    n_spins = prod(shape)
    spin_idxs = jnp.arange(n_spins, dtype=dtype)
    basis_idxs = jnp.arange(2**n_spins, dtype=dtype)

    flat_basis = jax.vmap(lambda basis_idx: (basis_idx >> spin_idxs) & 1)(basis_idxs)

    return jnp.reshape(2 * flat_basis - 1, (-1, *shape)).astype(dtype)


def _to_absolute_index(x):
    x = jnp.ravel((x + 1) // 2).astype(default_int_dtype())
    b = 2 ** jnp.arange(x.size, dtype=x.dtype)
    return b @ x


def _to_relative_index(x, basis, sentinel=None):

    x = x.ravel().astype(basis.dtype)

    if sentinel is None:
        sentinel = -basis.shape[0] - 1

    flags = jnp.all(x == lax.collapse(basis, 1), axis=-1)
    return jnp.where(flags.any(), flags.argmax(), sentinel)


def to_index(x: Array, basis: Array | None = None, sentinel: Scalar | None = None) -> Scalar:
    if basis is None:
        return _to_absolute_index(x)
    else:
        return _to_relative_index(x, basis, sentinel)


def to_dense(
    op: SpinOperator,
    basis: Array | None = None,
    *,
    chunk_size: int | None = None,
    dtype: DType | None = None
) -> Array:

    if basis is None:
        basis = spin_basis(op.graph.num_nodes)
    else:
        basis = basis.astype(default_int_dtype())

    to_index = jax.vmap(lambda x: _to_relative_index(x, basis))

    dtype = dtype or default_complex_dtype()
    empty_row = jnp.zeros(basis.shape[0], dtype=dtype)

    @partial(vmap, chunk_size=chunk_size)
    def _matrix_row(x):
        conn, mels = op.conn(x)
        idxs = to_index(conn)
        return empty_row.at[idxs].set(mels, mode="drop")

    return _matrix_row(basis)


def to_sparse(
    op: SpinOperator,
    basis: Array | None = None,
    *,
    chunk_size: int | None = None,
    dtype: DType | None = None
):
    raise NotImplementedError("Sparse representations not implemented yet.")


def conn_z(_: Graph, x: Array) -> MatrixElements:
    return MatrixElements(x[None, ...], x.sum().reshape(1))


def conn_x(_: Graph, x: Array) -> MatrixElements:

    conn = jnp.broadcast_to(x.ravel(), (x.size, x.size))
    conn = conn.at[jnp.diag_indices_from(conn)].multiply(-1).reshape(-1, *x.shape)
    mels = jnp.ones(x.size)

    return MatrixElements(conn, mels)


def conn_xx(graph: Graph, x: Array) -> MatrixElements:

    x_ = jnp.ravel(x)

    conn = jax.vmap(lambda edge: x_.at[edge].mul(-1))(graph.edges)
    conn = conn.reshape(graph.num_edges, *x.shape)
    mels = graph.edge_weights

    return MatrixElements(conn, mels)


def conn_zz(graph: Graph, x: Array) -> MatrixElements:
    edge_products = jnp.ravel(x)[graph.edges].prod(axis=-1)
    edge_total = graph.edge_weights @ edge_products
    return MatrixElements(x[None, ...], edge_total[None])


####################################################################################################


class PauliOperator(SpinOperator):
    @property
    def is_hermitian(self) -> bool:
        return True


class I(PauliOperator):
    def conn(self, x: Array) -> MatrixElements:
        return MatrixElements(x[None, ...], jnp.array([1.0]))


class X(PauliOperator):
    def conn(self, x: Array) -> MatrixElements:
        return conn_x(self.graph, x)


class Z(PauliOperator):
    def conn(self, x: Array) -> MatrixElements:
        return conn_z(self.graph, x)


class XX(PauliOperator):
    def conn(self, x: Array) -> MatrixElements:
        return conn_xx(self.graph, x)


class ZZ(PauliOperator):
    def conn(self, x: Array) -> MatrixElements:
        return conn_zz(self.graph, x)


class TransverseFieldIsing(PauliOperator):

    J: Scalar
    h: Scalar
    rotated: bool = eqx.field(static=True)

    def __init__(self, graph: Graph, J: Scalar, h: Scalar, rotated: bool = False):
        super().__init__(graph)
        self.J, self.h, self.rotated = J, h, rotated

    def conn(self, x: Array) -> MatrixElements:

        if self.rotated:
            conn_1, mels_1 = conn_xx(self.graph, x)
            conn_2, mels_2 = conn_z(self.graph, x)
        else:
            conn_1, mels_1 = conn_zz(self.graph, x)
            conn_2, mels_2 = conn_x(self.graph, x)

        conn = jnp.concatenate([conn_1, conn_2], axis=0)
        mels = jnp.concatenate([-self.J * mels_1, -self.h * mels_2])

        return MatrixElements(conn, mels)
