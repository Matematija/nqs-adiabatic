from typing import Callable

import jax
from jax import numpy as jnp
import jax.random as jr
from jax import Array

import equinox as eqx
from equinox import nn

from .base import Linear, Conv, FFTConv, Activation
from .utils import real_params
from ..utils import Key, DType, Shape, default_complex_dtype


def log_cosh(x: Array) -> Array:
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


@real_params
class RestrictedBoltzmannMachine(eqx.Module):

    hidden: Linear
    visible: Linear | None
    activation: Activation

    def __init__(
        self,
        n_spins: int,
        alpha: float = 2.0,
        use_visible_bias: bool = True,
        use_hidden_bias: bool = True,
        activation: Activation = log_cosh,
        dtype: DType | None = None,
        *,
        key: Key,
    ):

        if dtype is None:
            dtype = default_complex_dtype()

        self.activation = activation

        key1, key2 = jax.random.split(key, 2)
        n_hidden = int(alpha * n_spins)

        self.hidden = Linear(n_spins, n_hidden, use_bias=use_hidden_bias, dtype=dtype, key=key1)

        if use_visible_bias:
            self.visible = Linear(n_spins, "scalar", use_bias=False, dtype=dtype, key=key2)
        else:
            self.visible = None

    def __call__(self, x: Array) -> Array:

        x_ = jnp.ravel(x)

        h = self.hidden(x_)
        y = jnp.sum(self.activation(h))

        if self.visible is not None:
            y += self.visible(x_)

        return y


@real_params
class ConvRBM(eqx.Module):

    conv: Conv
    activation: Activation

    def __init__(
        self,
        dims: int | Shape,
        n_channels: int = 4,
        pbc: bool = True,
        use_bias: bool = True,
        activation: Activation = log_cosh,
        *,
        key: Key,
    ):

        self.activation = activation
        out_dtype = default_complex_dtype()

        self.conv = FFTConv(
            dims,
            in_channels=1,
            out_channels=n_channels,
            pbc=pbc,
            use_bias=use_bias,
            out_dtype=out_dtype,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        return self.activation(self.conv(x[None, ...])).sum()


####################################################################################################


class SpinEmbedding(eqx.Module):

    conv: nn.Conv

    def __init__(self, dims: int | Shape, embed_dim: int, *, key: Key):

        ndim = len(dims) if not isinstance(dims, int) else 1

        self.conv = nn.Conv(
            num_spatial_dims=ndim,
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=1,
            use_bias=False,
            key=key,
        )

    def __call__(self, x: Array) -> Array:
        return self.conv(x[None, ...])


class ResidualBlock(eqx.Module):

    conv1: nn.Conv
    conv2: nn.Conv
    norm: nn.LayerNorm | nn.Identity
    activation: Activation

    def __init__(
        self,
        dims: int | Shape,
        n_channels: int,
        kernel_size: int | Shape = 3,
        pbc: bool = True,
        enhancement: float = 2.0,
        activation: Activation = jax.nn.silu,
        use_bias: bool = True,
        norm: bool = True,
        *,
        key: Key,
    ):

        key1, key2 = jax.random.split(key, 2)
        num_spatial_dims = len(dims) if not isinstance(dims, int) else 1
        mid_channels = int(enhancement * n_channels)
        self.activation = activation
        padding_mode = "CIRCULAR" if pbc else "ZEROS"

        make_conv = lambda in_features, out_features, key: nn.Conv(
            num_spatial_dims,
            in_features,
            out_features,
            kernel_size,
            padding="SAME",
            padding_mode=padding_mode,
            use_bias=use_bias,
            key=key,
        )

        self.conv1 = make_conv(n_channels, mid_channels, key=key1)
        self.conv2 = make_conv(mid_channels, n_channels, key=key2)
        self.norm = nn.LayerNorm(dims) if norm else nn.Identity()

    def __call__(self, x: Array) -> Array:
        y = self.conv1(jax.vmap(self.norm)(x))
        y = self.conv2(self.activation(y))
        return x + y


class ResidualEncoder(eqx.Module):

    embedding: SpinEmbedding
    blocks: list[ResidualBlock]
    norm: nn.LayerNorm | nn.Identity

    def __init__(
        self,
        dims: int | Shape,
        n_channels: int,
        n_blocks: int = 1,
        kernel_size: int | Shape = 3,
        pbc: bool = True,
        enhancement: float = 2.0,
        activation: Callable = jax.nn.silu,
        use_norms: bool = True,
        use_bias: bool = True,
        *,
        key: Key,
    ):

        embed_key, block_key = jax.random.split(key, 2)

        make_block = lambda key: ResidualBlock(
            dims,
            n_channels,
            kernel_size=kernel_size,
            pbc=pbc,
            enhancement=enhancement,
            activation=activation,
            norm=use_norms,
            use_bias=use_bias,
            key=key,
        )

        self.embedding = SpinEmbedding(dims, n_channels, key=embed_key)
        self.blocks = [make_block(jr.fold_in(block_key, i)) for i in range(n_blocks)]
        self.norm = nn.LayerNorm(dims) if use_norms else nn.Identity()

    def __call__(self, x: Array) -> Array:

        h = self.embedding(x)

        for block in self.blocks:
            h = block(h)

        h = jax.vmap(self.norm)(h)
        return jax.nn.soft_sign(h)


@real_params
class RBMHead(eqx.Module):

    conv: Conv
    skip_conv: Conv

    def __init__(
        self,
        dims: int | Shape,
        n_hidden_channels: int,
        pbc: bool = True,
        use_bias: bool = True,
        *,
        key: Key,
    ):

        key, skip_key = jax.random.split(key, 2)
        out_dtype = default_complex_dtype()

        self.conv = FFTConv(
            dims,
            n_hidden_channels,
            n_hidden_channels,
            pbc=pbc,
            out_dtype=out_dtype,
            use_bias=False,
            key=key,
        )

        self.skip_conv = FFTConv(
            dims,
            in_channels=1,
            out_channels=n_hidden_channels,
            pbc=pbc,
            out_dtype=out_dtype,
            use_bias=use_bias,
            key=skip_key,
        )

    def __call__(self, x: Array, h: Array) -> Array:
        x_ = self.skip_conv(x[None, ...])
        h_ = self.conv(h)
        return log_cosh(x_ + h_).mean(axis=0).sum()


class ResidualRBM(eqx.Module):

    encoder: ResidualEncoder
    proj: RBMHead

    def __init__(
        self,
        dims: int | Shape,
        n_channels: int,
        n_blocks: int = 1,
        kernel_size: int | Shape = 3,
        pbc: bool = True,
        enhancement: float = 2.0,
        activation: Callable = jax.nn.silu,
        use_norms: bool = True,
        use_bias: bool = True,
        *,
        key: Key,
    ):

        encoder_key, proj_key = jax.random.split(key, 2)

        self.encoder = ResidualEncoder(
            dims,
            n_channels,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            pbc=pbc,
            enhancement=enhancement,
            activation=activation,
            use_norms=use_norms,
            use_bias=use_bias,
            key=encoder_key,
        )

        self.proj = RBMHead(dims, n_hidden_channels=n_channels, use_bias=use_bias, key=proj_key)

    def __call__(self, x: Array) -> Array:
        return self.proj(x, self.encoder(x))
