from typing import Callable, Any

from jax import numpy as jnp
from jax import Array

import equinox as eqx

from .base import Operator
from .local import eval_local_operator
from ..utils import abs2, vmap, Ansatz, ComplexValue


@eqx.filter_custom_vjp
def _hermitian_expect(diff_args, samples, axis, chunk_size, hooks):

    values, _, _ = diff_args
    fwd_hook, _ = hooks

    if fwd_hook is not None:
        values = fwd_hook(values)

    return jnp.mean(values, axis=axis)


@_hermitian_expect.def_fwd
def _hermitian_expect_fwd(perturbed, diff_args, samples, axis, chunk_size, hooks):

    values, _, _ = diff_args
    fwd_hook, _ = hooks

    mean = jnp.mean(values, axis=axis)
    centered_vals = values - jnp.expand_dims(mean, axis=axis)
    fwd_val = mean if fwd_hook is None else jnp.mean(fwd_hook(values), axis=axis)

    return fwd_val, centered_vals


@_hermitian_expect.def_bwd
def _hermitian_expect_bwd(
    centered_vals, grad, perturbed, diff_args, samples, axis, chunk_size, hooks
):

    _, logpsi, args = diff_args
    _, bwd_hook = hooks
    n_samples = samples.shape[axis]

    if bwd_hook is not None:
        centered_vals = bwd_hook(centered_vals)

    y = 2 * grad * centered_vals / n_samples
    in_axes = (0,) + (None,) * len(args)

    def f(logpsi, args):
        val = vmap(logpsi, in_axes=in_axes, chunk_size=chunk_size)(samples, *args)
        return ComplexValue(val)

    _, back = eqx.filter_vjp(f, logpsi, args)
    d_logpsi, d_args = back(ComplexValue(y))

    return None, d_logpsi, d_args


####################################################################################


def local_operator_expect(
    values: Array,
    logpsi: Ansatz | None = None,
    samples: Array | None = None,
    *args: Any,
    keep_imag: bool = False,
    fwd_hook: Callable | None = None,
    bwd_hook: Callable | None = None,
    axis: int = 0,
    chunk_size: int | None = None,
) -> Array:

    expval = _hermitian_expect(
        (values, logpsi, args), samples, axis, chunk_size, (fwd_hook, bwd_hook)
    )

    return expval if keep_imag else jnp.real(expval)


def _eval_local_op(operator, logpsi, samples, *args, chunk_size=None):
    local_op_fn = lambda x: eval_local_operator(operator, logpsi, x, *args)
    return vmap(local_op_fn, chunk_size=chunk_size)(samples)


def operator_expect(
    operator: Operator,
    logpsi: Ansatz,
    samples: Array,
    *args: Any,
    keep_imag: bool = False,
    fwd_hook: Callable | None = None,
    bwd_hook: Callable | None = None,
    axis: int = 0,
    chunk_size: int | None = None,
) -> Array:

    local_op_vals = _eval_local_op(operator, logpsi, samples, chunk_size=chunk_size)

    return local_operator_expect(
        local_op_vals,
        logpsi,
        samples,
        *args,
        keep_imag=keep_imag,
        fwd_hook=fwd_hook,
        bwd_hook=bwd_hook,
        axis=axis,
        chunk_size=chunk_size,
    )


def operator_expect_and_variance(
    operator: Operator,
    logpsi: Ansatz,
    samples: Array,
    *args: Any,
    keep_imag: bool = False,
    axis: int = 0,
    chunk_size: int | None = None,
) -> tuple[Array, Array]:

    local_op_vals = _eval_local_op(operator, logpsi, samples, *args, chunk_size=chunk_size)

    mean = local_operator_expect(
        local_op_vals, logpsi, samples, *args, keep_imag=True, axis=axis, chunk_size=chunk_size
    )

    shift = jnp.expand_dims(mean, axis=axis)
    var = jnp.mean(abs2(local_op_vals - shift))

    if not keep_imag:
        mean = jnp.real(mean)

    return mean, var
