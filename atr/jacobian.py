from typing import Any, Literal
from functools import partial
import warnings

from math import sqrt

import jax
from jax import numpy as jnp
from jax import Array

import equinox as eqx
import lineax as lx

from .linalg import linear_solve, JacobianLinearOperator
from .utils import vmap, tree_size, tree_dot, is_complex_tree, ComplexValue
from .utils import Ansatz, PyTree

warnings.filterwarnings("ignore", category=jnp.ComplexWarning)


def _model_partition(model, axis=0, chunk_size=None):

    params, backbone = eqx.partition(model, eqx.is_inexact_array)

    @partial(vmap, in_axes=(None, axis), chunk_size=chunk_size)
    def logpsi_fn(p, x):
        return eqx.combine(p, backbone)(x)

    return params, logpsi_fn


def _mean_grad(logpsi_fn, params, samples, axis=0):
    n_samples = samples.shape[axis]
    out, back = jax.vjp(lambda p: ComplexValue(logpsi_fn(p, samples)), params)
    cotangents = jax.tree.map(lambda l: jnp.full_like(l, 1 / n_samples), out)
    return back(cotangents)[0]


def jacobian(
    logpsi: Ansatz,
    samples: Array,
    centered: bool = False,
    axis: int = 0,
    chunk_size: int | None = None,
    mode: Literal["fwd", "bwd"] | None = "bwd",
) -> JacobianLinearOperator:

    out_shape_dtype = eqx.filter_eval_shape(logpsi, jnp.take(samples, 0, axis))
    output_size = tree_size(out_shape_dtype)

    if not output_size == 1:
        raise ValueError(f"Function must return a single value, got {output_size}.")

    params, logpsi_fn = _model_partition(logpsi, axis, chunk_size)

    if is_complex_tree(params):
        raise ValueError("Complex parameters are not supported.")

    if centered:
        mean_grad = _mean_grad(logpsi_fn, params, samples, axis=axis)

    def eval_logpsi(p):
        val = logpsi_fn(p, samples)
        if centered:
            val = val - tree_dot(mean_grad, p)
        return ComplexValue(val)

    return JacobianLinearOperator(
        eval_logpsi, params, out_chunk_size=chunk_size, materialise_strategy=mode
    )


def natural_gradient(
    logpsi: Ansatz,
    samples: Array,
    local_vals: Array,
    solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
    options: dict[str, Any] = {},
    axis: int = 0,
    diag_shift: float | Array | None = None,
    dense: bool = True,
    chunk_size: int | None = None,
) -> PyTree:

    J = jacobian(logpsi, samples, centered=True, axis=axis, chunk_size=chunk_size)

    if dense:
        J = lx.materialise(J)

    eps = local_vals - local_vals.mean()
    scale = sqrt(samples.shape[axis])

    return linear_solve(
        J / scale,
        ComplexValue(eps / scale),
        preconditioner="normal",
        solver=solver,
        options=options,
        diag_shift=diag_shift,
    )
