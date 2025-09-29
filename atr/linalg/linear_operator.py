from typing import Literal, Callable, Iterable, Any

import jax
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_transpose, tree_structure
from jax.flatten_util import ravel_pytree

import equinox as eqx
import lineax as lx

from ..utils import vmap, PyTree


# TO DO: Add a version where .mv method can be directly overloaded
class FunctionLinearOperator(lx.FunctionLinearOperator):

    in_chunk_size: int | None = eqx.field(static=True)
    out_chunk_size: int | None = eqx.field(static=True)
    materialise_strategy: Literal["fwd", "bwd"] | None = eqx.field(static=True)

    def __init__(
        self,
        fn: Callable[[PyTree], PyTree],
        input_structure: PyTree,
        tags: Iterable[str] = (),
        in_chunk_size: int | None = None,
        out_chunk_size: int | None = None,
        materialise_strategy: Literal["fwd", "bwd"] | None = None,
    ):

        super().__init__(fn, input_structure=input_structure, tags=tags)
        self.in_chunk_size = in_chunk_size
        self.out_chunk_size = out_chunk_size
        self.materialise_strategy = materialise_strategy

    def transpose(self) -> "FunctionLinearOperator":

        if lx.symmetric_tag in self.tags:
            return self

        transpose_fn = jax.linear_transpose(self.fn, self.in_structure())

        def _transpose_fn(vector):
            out = transpose_fn(vector)
            return out[0] if len(out) == 1 else out

        if self.materialise_strategy is not None:

            if self.materialise_strategy == "fwd":
                materialise_strategy = "bwd"
            elif self.materialise_strategy == "bwd":
                materialise_strategy = "fwd"
            else:
                raise ValueError(f"Unknown materialise strategy: {self.materialise_strategy}")

        else:
            materialise_strategy = None

        return FunctionLinearOperator(
            _transpose_fn,
            input_structure=self.out_structure(),
            tags=lx.transpose_tags(self.tags),
            in_chunk_size=self.out_chunk_size,
            out_chunk_size=self.in_chunk_size,
            materialise_strategy=materialise_strategy,
        )


def _strip_weak_dtype(tree):

    # Mostly copied from:
    # https://github.com/patrick-kidger/lineax/blob/main/lineax/_misc.py

    def strip(leaf):
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding)
        else:
            return leaf

    return tree_map(strip, tree)


def _fwd_materialise(operator: FunctionLinearOperator) -> PyTree:

    flat, re = _strip_weak_dtype(eqx.filter_eval_shape(ravel_pytree, operator.in_structure()))
    I = jnp.eye(flat.size, dtype=flat.dtype)

    mat_vec = operator.fn
    mat = vmap(lambda x: mat_vec(re(x)), out_axes=-1, chunk_size=operator.in_chunk_size)(I)

    def _unravel(leaf):
        *prefix, length = leaf.shape
        unravel_fn = vmap(re, chunk_size=operator.out_chunk_size)
        unraveled = unravel_fn(leaf.reshape(-1, length))
        return tree_map(lambda l: l.reshape(*prefix, *l.shape[1:]), unraveled)

    return tree_map(_unravel, mat)


def _bwd_materialise(operator: FunctionLinearOperator) -> PyTree:

    flat, re = _strip_weak_dtype(eqx.filter_eval_shape(ravel_pytree, operator.out_structure()))
    I = jnp.eye(flat.size, dtype=flat.dtype)

    vec_mat = jax.linear_transpose(operator.fn, operator.in_structure())

    def vec_mat_(x):
        val = vec_mat(re(x))
        return val[0] if len(val) == 1 else val

    mat = vmap(vec_mat_, out_axes=0, chunk_size=operator.out_chunk_size)(I)

    def _unravel(leaf):
        length, *suffix = leaf.shape
        unravel_fn = vmap(re, in_axes=-1, out_axes=-1, chunk_size=operator.in_chunk_size)
        unraveled = unravel_fn(leaf.reshape(length, -1))
        return tree_map(lambda l: l.reshape(*l.shape[:-1], *suffix), unraveled)

    out_tree = tree_map(_unravel, mat)

    in_treedef = tree_structure(operator.in_structure())
    out_treedef = tree_structure(operator.out_structure())

    return tree_transpose(in_treedef, out_treedef, out_tree)


def _auto_materialise_strategy(operator: FunctionLinearOperator) -> str:
    if operator.in_size() <= operator.out_size():
        return "fwd"
    else:
        return "bwd"


@lx.materialise.register(FunctionLinearOperator)
def _(operator: FunctionLinearOperator) -> lx.PyTreeLinearOperator:

    materialise_strategy = operator.materialise_strategy

    if materialise_strategy is None:
        materialise_strategy = _auto_materialise_strategy(operator)

    if materialise_strategy == "fwd":
        matrix_tree = _fwd_materialise(operator)
    elif materialise_strategy == "bwd":
        matrix_tree = _bwd_materialise(operator)
    else:
        raise ValueError(f"Unknown materialise strategy: {materialise_strategy}")

    return lx.PyTreeLinearOperator(matrix_tree, operator.out_structure(), operator.tags)


class JacobianLinearOperator(FunctionLinearOperator):

    _fn_jac: Callable[[PyTree, Any], PyTree]
    x: PyTree

    def __init__(self, fn: Callable[[PyTree, Any], PyTree], x: Any, *args, **kwargs):

        self._fn_jac, self.x = fn, x
        mat_vec = lambda v: jax.jvp(fn, (x,), (v,))[1]
        input_structure = jax.eval_shape(lambda: x)

        super().__init__(mat_vec, input_structure, *args, **kwargs)


@lx.linearise.register(JacobianLinearOperator)
def _(operator: JacobianLinearOperator) -> FunctionLinearOperator:

    _, fwd = jax.linearize(operator._fn_jac, operator.x)

    return FunctionLinearOperator(
        fwd,
        input_structure=operator.in_structure(),
        tags=operator.tags,
        in_chunk_size=operator.in_chunk_size,
        out_chunk_size=operator.out_chunk_size,
        materialise_strategy=operator.materialise_strategy,
    )
