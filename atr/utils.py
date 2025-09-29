from typing import Callable, Sequence
from functools import partial

import numpy as np

import jax
from jax import numpy as jnp
from jax import lax
from jax import Array

import equinox as eqx

from jaxtyping import Scalar, DTypeLike, PyTree, Key

Shape = Sequence[int]
DType = DTypeLike

Ansatz = eqx.Module
AnsatzFn = Callable[[PyTree, Array], Array]


class ComplexValue(eqx.Module):

    real: Array
    imag: Array | None = None

    def __init__(self, arr: Array):

        if jnp.iscomplexobj(arr):
            self.real = jnp.real(arr)
            self.imag = jnp.imag(arr)
        else:
            self.real = arr

    def __jax_array__(self) -> Array:
        return self.real if self.is_real else lax.complex(self.real, self.imag)

    @property
    def is_real(self) -> bool:
        return self.imag is None


def unpack_complex_tree(tree: PyTree) -> PyTree:
    return jax.tree.map(lambda l: ComplexValue(l) if jnp.iscomplexobj(l) else l, tree)


def pack_complex_tree(tree: PyTree) -> PyTree:
    return jax.tree.map(jnp.asarray, tree, is_leaf=lambda x: isinstance(x, ComplexValue))


def select_component(f: Callable, i: int, axis: int = 0) -> Callable:
    return lambda *args, **kwargs: jnp.take(f(*args, **kwargs), i, axis=axis)


def curry(f: Callable) -> Callable:
    return partial(partial, f)


def tree_size(tree: PyTree) -> int:
    return sum(jnp.size(l) for l in jax.tree.leaves(tree))


def tree_shape(tree: PyTree) -> PyTree:
    return jax.tree.map(jnp.shape, tree)


def tree_dot(left: PyTree, right: PyTree, conj: bool = False) -> Scalar:

    if conj:
        left = jax.tree.map(jnp.conjugate, left)

    aux = jax.tree.map(lambda a, b: jnp.tensordot(a, b, axes=a.ndim), left, right)
    return jax.tree.reduce(jnp.add, aux)


def abs2(z: Array) -> Array:
    return jnp.real(z) ** 2 + jnp.imag(z) ** 2


##### DType utilities #####


def default_dtype() -> DType:
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


def default_int_dtype() -> DType:
    return jnp.int64 if jax.config.x64_enabled else jnp.int32


def default_complex_dtype() -> DType:
    return jnp.complex128 if jax.config.x64_enabled else jnp.complex64


def is_complex_dtype(dtype: DType) -> bool:
    return jnp.issubdtype(dtype, jnp.complexfloating)


def is_complex(arr: Array) -> bool:
    return is_complex_dtype(arr.dtype)


def is_complex_tree(tree: PyTree) -> bool:
    return any(is_complex(l) for l in jax.tree.leaves(tree))


def is_real_dtype(dtype: DType) -> bool:
    return jnp.issubdtype(dtype, jnp.floating)


def is_real(arr: Array) -> bool:
    return is_real_dtype(arr.dtype)


def real_dtype(dtype: DType) -> DType:

    if is_complex_dtype(dtype):
        if dtype == np.dtype("complex64"):
            return np.dtype("float32")
        elif dtype == np.dtype("complex128"):
            return np.dtype("float64")
        else:
            raise TypeError(f"Unknown complex dtype {dtype}.")
    else:
        return np.dtype(dtype)


def complex_dtype(dtype: DType) -> DType:

    if is_real_dtype(dtype):
        if dtype == np.dtype("float32"):
            return np.dtype("complex64")
        elif dtype == np.dtype("float64"):
            return np.dtype("complex128")
        else:
            raise TypeError(f"Unknown real dtype {dtype}.")
    else:
        return np.dtype(dtype)


def to_complex(arr: Array) -> Array:

    dtype = arr.dtype

    if is_real_dtype(dtype):
        return arr.astype(complex_dtype(dtype))
    else:
        return arr


####################################################################################################


def _argnums_partial(fun, args, dyn_argnums):

    sentinel = object()
    args_template = [sentinel] * len(args)
    dyn_args = []

    for i, arg in enumerate(args):
        if i in dyn_argnums:
            dyn_args.append(arg)
        else:
            args_template[i] = arg

    def fun_partial(*new_dyn_args):

        arg_iter = iter(new_dyn_args)

        interpolated_args = tuple(
            next(arg_iter) if arg == sentinel else arg for arg in args_template
        )

        return fun(*interpolated_args)

    return fun_partial, dyn_args


def _transpose_vmap_output(y, oax):
    if oax is None or oax == 0:
        return y
    else:
        return jnp.moveaxis(y, 0, oax)


def _transpose_vmap_outputs(outputs, axes):  # What a mess this is

    if len(axes) == 1:
        axes, outputs = (axes,), (outputs,)
        unpack = True
    else:
        unpack = False

    assert len(outputs) == len(axes)

    out = tuple(
        jax.tree.map(lambda l: _transpose_vmap_output(l, oax), leaf)
        for leaf, oax in zip(outputs, axes)
    )

    return out[0] if unpack else out


def _to_shape(x: int | tuple) -> tuple:
    return (x,) if isinstance(x, int) else x


def vmap(
    fun: Callable,
    in_axes: int | Shape = 0,
    out_axes: int | Shape = 0,
    chunk_size: int | None = None,
    *args,
    **kwargs,
) -> Callable:

    if chunk_size is None:
        return jax.vmap(fun, in_axes, out_axes, *args, **kwargs)

    in_axes = _to_shape(in_axes)
    argnums = tuple(i for i, ix in enumerate(in_axes) if ix is not None)

    if not set(in_axes).issubset((0, None)):
        _in_axes = [ix % len(in_axes) for ix in in_axes if ix is not None]

        def preprocess_dyn_args(dyn_args):
            return jax.tree.map(jnp.moveaxis, dyn_args, _in_axes, [0] * len(_in_axes))

    else:
        preprocess_dyn_args = lambda x: x

    if not set(_to_shape(out_axes)).issubset((0, None)):
        postprocess_output = _transpose_vmap_outputs
    else:
        postprocess_output = lambda x, *_: x

    def f_chunked(*args, **kwargs):

        f_partial, dyn_args = _argnums_partial(partial(fun, **kwargs), args, argnums)
        dyn_args = preprocess_dyn_args(dyn_args)

        out = lax.map(lambda args: f_partial(*args), dyn_args, batch_size=chunk_size)

        return postprocess_output(out, _to_shape(out_axes))

    return f_chunked


class _VmapWrapper(eqx.Module):

    _fun: Callable
    _in_axes: int | Shape
    _out_axes: int | Shape
    _chunk_size: int | None

    @property
    def __wrapped__(self):
        return self._fun

    def __call__(self, *args, **kwargs):

        dynamic_args, static_args = eqx.partition(args, eqx.is_inexact_array)

        @partial(vmap, in_axes=self._in_axes, out_axes=self._out_axes, chunk_size=self._chunk_size)
        def vmap_aux(*dyn_args):
            args = eqx.combine(dyn_args, static_args)
            return self._fun(*args, **kwargs)

        return vmap_aux(*dynamic_args)


def filter_vmap(
    fun: Callable,
    in_axes: int | Shape = 0,
    out_axes: int | Shape = 0,
    chunk_size: int | None = None,
) -> Callable:

    return _VmapWrapper(fun, in_axes, out_axes, chunk_size)


####################################################################################################


def value_and_grad(f, has_aux=False):
    """
    A complex-aware AD helper
    """

    def get_grad(value, back):

        one = jnp.ones_like(value)

        if is_complex(value):
            (grad_re,) = back(one)
            (grad_im,) = back(-1j * one)
            grad = jax.tree.map(lax.complex, grad_re, grad_im)
        else:
            (grad,) = back(one)

        return grad

    if has_aux:

        def value_and_grad_fn(p, *args):
            value, back, aux = jax.vjp(lambda p: f(p, *args), p, has_aux=True)
            return (value, aux), get_grad(value, back)

    else:

        def value_and_grad_fn(p, *args):
            value, back = jax.vjp(lambda p: f(p, *args), p, has_aux=False)
            return value, get_grad(value, back)

    return value_and_grad_fn


class _ValueAndGradWrapper(eqx.Module):

    _fun: Callable
    _has_aux: bool

    @property
    def __wrapped__(self):
        return self._fun

    def __call__(self, diff_arg, *args, **kwargs):

        dynamic_arg, static_arg = eqx.partition(diff_arg, eqx.is_inexact_array)

        @partial(value_and_grad, has_aux=self._has_aux)
        def value_and_grad_aux(dyn_arg):
            diff_arg = eqx.combine(dyn_arg, static_arg)
            return self._fun(diff_arg, *args, **kwargs)

        return value_and_grad_aux(dynamic_arg)


def filter_value_and_grad(fun: Callable, has_aux: bool = False) -> _ValueAndGradWrapper:
    return _ValueAndGradWrapper(fun, has_aux)


###############################################################################################


class Trajectory(eqx.Module):

    t: Array = eqx.field(converter=jnp.asarray)
    y: PyTree
    data: PyTree | None = None

    @property
    def has_data(self):
        return self.data is not None

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i: int) -> "Trajectory":

        ti = self.t[i]
        yi = jax.tree.map(lambda y: y[i], self.y)

        if self.has_data:
            data = jax.tree.map(lambda l: l[i], self.data)
        else:
            data = None

        return ti, yi, data

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
