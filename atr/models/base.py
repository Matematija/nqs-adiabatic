from typing import Any, Callable, Sequence

import jax
from jax import numpy as jnp
from jax import Array

import equinox as eqx
from equinox import nn

from ..utils import Shape, Key, DType


Activation = Callable[[Array], Array]
ParamInitFn = Callable[[Key, Shape, DType], Array]

default_kernel_init = jax.nn.initializers.variance_scaling(
    0.1, mode="fan_in", distribution="truncated_normal"
)

default_bias_init = jax.nn.initializers.zeros


# class _CustomInit(eqx.Module):
#     def __init__(
#         self,
#         *args,
#         key: Key,
#         kernel_init: ParamInitFn = default_kernel_init,
#         bias_init: ParamInitFn = default_bias_init,
#         **kwargs,
#     ):

#         super().__init__(*args, **kwargs, key=key)
#         key, weights_key, bias_key = jax.random.split(key, 3)

#         self.weight = kernel_init(weights_key, self.weight.shape, self.weight.dtype)

#         if self.use_bias:
#             self.bias = bias_init(bias_key, self.bias.shape, self.bias.dtype)


class Linear(eqx.Module):

    wrapped: nn.Linear

    def __init__(
        self,
        *args,
        kernel_init: ParamInitFn = default_kernel_init,
        bias_init: ParamInitFn = default_bias_init,
        **kwargs,
    ):

        key = kwargs.pop("key")
        key, wkey, bkey = jax.random.split(key, 3)
        self.wrapped = nn.Linear(*args, **kwargs, key=key)

        weight_ = kernel_init(wkey, self.wrapped.weight.shape, self.wrapped.weight.dtype)
        self.wrapped = eqx.tree_at(lambda m: m.weight, self.wrapped, weight_)

        if self.wrapped.use_bias:
            bias_ = bias_init(bkey, self.wrapped.bias.shape, self.wrapped.bias.dtype)
            self.wrapped = eqx.tree_at(lambda m: m.bias, self.wrapped, bias_)

    def __call__(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)


def _antiperiodic(row: Array, pad_width: tuple[int, int], iaxis: int, kwargs: dict) -> Array:
    before, after = pad_width
    n = len(row) - before - after
    if before > 0:
        row = row.at[:before].set(-row[n : n + before])  # last `before` of original
    if after > 0:
        row = row.at[n + before :].set(-row[before : before + after])  # first `after` of original
    return row


def _bc_pad(x, kernel_size, bc):

    d = len(kernel_size)
    batch_pad_width = [(0, 0)] * (x.ndim - d)

    modes = {"p": "wrap", "o": "constant", "a": _antiperiodic}

    for dim, (k, c) in enumerate(zip(kernel_size, bc)):
        pad_width = batch_pad_width + [(0, 0)] * dim + [(k // 2,) * 2] + [(0, 0)] * (d - dim - 1)
        x = jnp.pad(x, pad_width, mode=modes[c])

    return x


def _parse_bc(bc: str | Sequence[str], ndim: int) -> str:
    if not isinstance(bc, str):
        bc = "".join(bc)
    return bc + bc[-1] * (ndim - len(bc))


class Conv(eqx.Module):

    wrapped: nn.Conv
    bc: str

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        bc: str | Sequence[str] = "p",
        stride: int | Sequence[int] = 1,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        dtype: Any | None = None,
        kernel_init: ParamInitFn = default_kernel_init,
        bias_init: ParamInitFn = default_bias_init,
        *,
        key: Key,
    ):

        key, wkey, bkey = jax.random.split(key, 3)

        self.bc = _parse_bc(bc, num_spatial_dims)

        self.wrapped = nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,
            stride=stride,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            dtype=dtype,
            key=key,
        )

        weight_ = kernel_init(wkey, self.wrapped.weight.shape, self.wrapped.weight.dtype)
        self.wrapped = eqx.tree_at(lambda m: m.weight, self.wrapped, weight_)

        if self.wrapped.use_bias:
            bias_ = bias_init(bkey, self.wrapped.bias.shape, self.wrapped.bias.dtype)
            self.wrapped = eqx.tree_at(lambda m: m.bias, self.wrapped, bias_)

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        x = _bc_pad(x, self.wrapped.kernel_size, self.bc)
        return self.wrapped(x, *args, **kwargs)


# class FFTConv(eqx.Module):

#     kernel: Array
#     bias: Array | None
#     _fft: Callable[[Array], Array] = eqx.field(static=True)
#     _ifft: Callable[[Array], Array] = eqx.field(static=True)

#     def __init__(
#         self,
#         dims: int | Sequence[int],
#         in_channels: int,
#         out_channels: int,
#         pbc: bool = True,
#         use_bias: bool = True,
#         kernel_init: ParamInitFn = default_kernel_init,
#         bias_init: ParamInitFn = default_bias_init,
#         in_dtype: DType = None,
#         out_dtype: DType = None,
#         *,
#         key: Key,
#     ):

#         if in_dtype is None:
#             in_dtype = default_dtype()

#         if out_dtype is None:
#             out_dtype = default_dtype()

#         dims = (dims,) if isinstance(dims, int) else tuple(dims)
#         padded_shape = dims if pbc else tuple(2 * d - 1 for d in dims)

#         if is_complex_dtype(out_dtype):
#             self._fft = lambda x: jnp.fft.fftn(x, padded_shape)
#             self._ifft = lambda x: jnp.fft.ifftn(x, padded_shape)
#         else:

#             if is_complex_dtype(in_dtype):
#                 raise ValueError("Input dtype must be real when output dtype is real.")

#             self._fft = lambda x: jnp.fft.rfftn(x, padded_shape)
#             self._ifft = lambda x: jnp.fft.irfftn(x, padded_shape)

#         kernel_key, bias_key = jax.random.split(key)

#         kernel_info = jax.eval_shape(lambda: self._fft(jnp.zeros(dims)))
#         kernel_shape = (out_channels, in_channels) + kernel_info.shape
#         kernel_dtype = kernel_info.dtype

#         self.kernel = kernel_init(kernel_key, kernel_shape, kernel_dtype)

#         if use_bias:
#             bias_shape = (out_channels,) + (1,) * len(dims)
#             self.bias = bias_init(bias_key, bias_shape, out_dtype)
#         else:
#             self.bias = None

#     def __call__(self, x: Array) -> Array:

#         x_ = self._fft(x).astype(self.kernel.dtype)
#         y_ = jnp.einsum("ab...,b...->a...", self.kernel, x_)
#         y = self._ifft(y_)

#         if self.bias is not None:
#             y += self.bias

#         return y
