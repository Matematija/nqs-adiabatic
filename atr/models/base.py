from typing import Callable, Sequence

import jax
from jax import numpy as jnp
from jax import Array

import equinox as eqx
from equinox import nn

from ..utils import default_dtype, is_complex_dtype, Shape, Key, DType


Activation = Callable[[Array], Array]
ParamInitFn = Callable[[Key, Shape, DType], Array]

default_kernel_init = jax.nn.initializers.variance_scaling(
    0.1, mode="fan_in", distribution="truncated_normal"
)

default_bias_init = jax.nn.initializers.zeros


class _CustomInit(eqx.Module):
    def __init__(
        self,
        *args,
        key: Key,
        kernel_init: ParamInitFn = default_kernel_init,
        bias_init: ParamInitFn = default_bias_init,
        **kwargs,
    ):

        super().__init__(*args, **kwargs, key=key)
        key, weights_key, bias_key = jax.random.split(key, 3)

        self.weight = kernel_init(weights_key, self.weight.shape, self.weight.dtype)

        if self.use_bias:
            self.bias = bias_init(bias_key, self.bias.shape, self.bias.dtype)


class Linear(_CustomInit, nn.Linear):
    pass


class Conv(_CustomInit, nn.Conv):
    pass


class FFTConv(eqx.Module):

    kernel: Array
    bias: Array | None
    _fft: Callable[[Array], Array] = eqx.field(static=True)
    _ifft: Callable[[Array], Array] = eqx.field(static=True)

    def __init__(
        self,
        dims: int | Sequence[int],
        in_channels: int,
        out_channels: int,
        pbc: bool = True,
        use_bias: bool = True,
        kernel_init: ParamInitFn = default_kernel_init,
        bias_init: ParamInitFn = default_bias_init,
        in_dtype: DType = None,
        out_dtype: DType = None,
        *,
        key: Key,
    ):

        if in_dtype is None:
            in_dtype = default_dtype()

        if out_dtype is None:
            out_dtype = default_dtype()

        dims = (dims,) if isinstance(dims, int) else tuple(dims)
        padded_shape = dims if pbc else tuple(2 * d - 1 for d in dims)

        if is_complex_dtype(out_dtype):
            self._fft = lambda x: jnp.fft.fftn(x, padded_shape)
            self._ifft = lambda x: jnp.fft.ifftn(x, padded_shape)
        else:

            if is_complex_dtype(in_dtype):
                raise ValueError("Input dtype must be real when output dtype is real.")

            self._fft = lambda x: jnp.fft.rfftn(x, padded_shape)
            self._ifft = lambda x: jnp.fft.irfftn(x, padded_shape)

        kernel_key, bias_key = jax.random.split(key)

        kernel_info = jax.eval_shape(lambda: self._fft(jnp.zeros(dims)))
        kernel_shape = (out_channels, in_channels) + kernel_info.shape
        kernel_dtype = kernel_info.dtype

        self.kernel = kernel_init(kernel_key, kernel_shape, kernel_dtype)

        if use_bias:
            bias_shape = (out_channels,) + (1,) * len(dims)
            self.bias = bias_init(bias_key, bias_shape, out_dtype)
        else:
            self.bias = None

    def __call__(self, x: Array) -> Array:

        x_ = self._fft(x).astype(self.kernel.dtype)
        y_ = jnp.einsum("ab...,b...->a...", self.kernel, x_)
        y = self._ifft(y_)

        if self.bias is not None:
            y += self.bias

        return y
