from typing import Any, Literal
from functools import partial
from math import sqrt

from jax import numpy as jnp
from jax import lax
from jax import Array

import lineax as lx

from .jacobian import jacobian, _mean_grad, _model_partition
from .linalg import linear_solve, JacobianLinearOperator
from .operator import LocalOperator, Operator
from .utils import vmap, is_complex_tree, tree_dot, ComplexValue, Scalar, Ansatz


def projected_jacobian(
    logpsi: Ansatz,
    samples: Array,
    operator: Operator,
    shift: Scalar,
    mode: Literal["fwd", "bwd"] | None = "bwd",
    axis: int = 0,
    chunk_size: int | None = None,
) -> JacobianLinearOperator:

    shift = jnp.asarray(shift)

    params, logpsi_fn = _model_partition(logpsi, axis, chunk_size)
    mean_grad = _mean_grad(logpsi_fn, params, samples, axis=axis)

    if is_complex_tree(params):
        raise ValueError("Complex parameters are not supported.")

    @partial(vmap, in_axes=(None, axis), chunk_size=chunk_size)
    def f(params, x):

        y, Oxy = operator.conn(x)
        y = jnp.concatenate([y, x[None]])
        Oxy = jnp.concatenate([Oxy, -shift[None]])
        Oxy = lax.stop_gradient(Oxy)

        logpsi_y = logpsi_fn(params, y)
        shifted_logpsi_y = logpsi_y - tree_dot(params, mean_grad)
        logpsi_ratio = jnp.exp(logpsi_y - logpsi_y[-1])
        logpsi_ratio = lax.stop_gradient(logpsi_ratio)

        return ComplexValue(Oxy @ (logpsi_ratio * shifted_logpsi_y))

    return JacobianLinearOperator(
        lambda p: f(p, samples), params, out_chunk_size=chunk_size, materialise_strategy=mode
    )


def inverse_power_update(
    logpsi: Ansatz,
    samples: Array,
    operator: Operator,
    shift: Scalar,
    solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
    options: dict[str, Any] | None = None,
    diag_shift: float | Array | None = None,
    dense: bool = True,
    axis: int = 0,
    chunk_size: int | None = None,
) -> Array:

    J = jacobian(logpsi, samples, centered=True, axis=axis, chunk_size=chunk_size)
    PJ = projected_jacobian(logpsi, samples, operator, shift, axis=axis, chunk_size=chunk_size)

    if dense:
        J = lx.materialise(J)
        PJ = lx.materialise(PJ)

    local_op = LocalOperator(operator, logpsi)
    local_op_vals = vmap(local_op, in_axes=(axis,), chunk_size=chunk_size)(samples)
    rhs = local_op_vals - local_op_vals.mean()

    scale = sqrt(samples.shape[axis])

    return linear_solve(
        PJ / scale,
        ComplexValue(-rhs / scale),
        preconditioner=J.transpose() / scale,
        solver=solver,
        # tags=[lx.symmetric_tag],
        options=options,
        diag_shift=diag_shift,
    )
