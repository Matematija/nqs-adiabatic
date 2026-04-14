from typing import NamedTuple, Callable
from functools import partial
from math import prod

import jax
from jax import numpy as jnp
from jax import lax
from jax import random as jr
from jax import Array

import equinox as eqx

from .utils import Scalar, Shape, DType, Key, default_dtype

LogProbFn = Callable[[Array], Scalar]
LogWaveFn = Callable[[Array], Scalar]
InitFn = Callable[[Shape, DType, Key], Array]
Proposal = Callable[[Array, Key], Array]


def mh_accept(log_prob: LogProbFn, x: Array, x_: Array, key: Key) -> bool:
    r = jnp.log(jr.uniform(key))
    return log_prob(x_) - log_prob(x) > r


class MCState(NamedTuple):

    x: Array
    accepted: Array
    key: Key

    @classmethod
    def init(cls, init_fn: InitFn, dims: Shape | int, dtype: DType, key: Key):
        key1, key2 = jr.split(key, 2)
        x0 = init_fn(dims, dtype, key1)
        accepted = jnp.array(True, dtype=jnp.bool_)
        return cls(x0, accepted, key2)


def step(log_prob: LogProbFn, proposal: Proposal, state: MCState) -> MCState:

    key1, key2, key3 = jr.split(state.key, 3)

    x_ = proposal(state.x, key1)
    accept = mh_accept(log_prob, state.x, x_, key2)
    x_new = jnp.where(accept, x_, state.x)

    return MCState(x_new, accept, key3)


def sweep(log_prob: LogProbFn, proposal: Proposal, state: MCState, n_steps: int) -> MCState:
    body_fn = lambda _, state: step(log_prob, proposal, state)
    return lax.fori_loop(0, n_steps, body_fn, state)


class Sampler(eqx.Module):

    proposal: Proposal
    init_fn: InitFn
    dims: Shape
    n_chains: int
    sweep_size: int
    dtype: DType = eqx.field(static=True)

    def __init__(
        self,
        dims: int | Shape,
        proposal: Proposal,
        init_fn: InitFn,
        *,
        n_chains: int = 1,
        sweep_size: int | None = None,
        dtype: DType | None = None
    ):

        if isinstance(dims, int):
            dims = (dims,)

        if sweep_size is None:
            sweep_size = prod(dims)

        if dtype is None:
            dtype = default_dtype()

        self.proposal = proposal
        self.init_fn = init_fn
        self.dims = dims
        self.n_chains = n_chains
        self.sweep_size = sweep_size
        self.dtype = dtype

    def init(self, log_prob: LogProbFn, warmup: int, key: Key) -> MCState:
        def init_chain_state(key):  # warmup
            init_state = MCState.init(self.init_fn, self.dims, self.dtype, key)
            return sweep(log_prob, self.proposal, init_state, warmup * self.sweep_size)

        keys = jr.split(key, self.n_chains)
        return jax.vmap(init_chain_state)(keys)

    def __call__(
        self, log_prob: LogProbFn, n_samples: int, state: MCState
    ) -> tuple[Array, MCState]:
        @partial(jax.vmap, in_axes=(0, None))
        def step_chain(state, _):
            state = sweep(log_prob, self.proposal, state, self.sweep_size)
            return state, state.x

        state, samples = lax.scan(step_chain, init=state, xs=None, length=n_samples)
        return samples, state


class StateSampler(eqx.Module):

    sampler: Sampler

    def __init__(self, *args, **kwargs):
        self.sampler = Sampler(*args, **kwargs)

    def init(self, *args, **kwargs) -> MCState:
        return self.sampler.init(*args, **kwargs)

    def __call__(self, logpsi: LogWaveFn, n_samples: int, state: MCState) -> tuple[Array, MCState]:
        log_prob = lambda *args, **kwargs: 2 * jnp.real(logpsi(*args, **kwargs))
        return self.sampler(log_prob, n_samples, state)


###############################################################################################


def random_spin_init_fn(shape: Shape, dtype: DType, key: Key) -> Array:
    spins = jnp.array([-1, 1], dtype=dtype)
    return jr.choice(key, spins, shape)


def zero_mag_init_fn(shape: Shape, dtype: DType, key: Key) -> Array:

    n_spins = prod(shape)
    assert n_spins % 2 == 0, "Number of spins must be even for zero magnetization"

    flip_idxs = jr.choice(key, n_spins, (n_spins // 2,), replace=False)

    return jnp.ones(n_spins, dtype=dtype).at[flip_idxs].set(-1).reshape(shape)


class SpinFlip(eqx.Module):

    max_flips: int = 1

    def __call__(self, x: Array, key: Key) -> Array:

        shape = x.shape
        x = x.ravel()

        # i = jr.randint(key, (self.num_flips,), 0, x.size)
        # return x.at[i].multiply(-1).reshape(shape)

        key1, key2 = jr.split(key, 2)
        vals = jnp.array([-1, 1], dtype=x.dtype)
        i = jr.choice(key1, x.size, (self.max_flips,), replace=False)
        flip = jr.choice(key2, vals, (self.max_flips,))

        return x.at[i].multiply(flip).reshape(shape)


class SpinExchange(eqx.Module):
    def __call__(self, x: Array, key: Key) -> Array:
        shape = x.shape
        x = x.ravel()
        i, j = jr.choice(key, x.size, (2,), replace=False)
        return x.at[i].set(x[j]).at[j].set(x[i]).reshape(shape)


class SpinSampler(eqx.Module):

    sampler: Sampler

    def __init__(
        self,
        *args,
        zero_mag: bool = False,
        proposal: Proposal | None = None,
        init_fn: InitFn | None = None,
        **kwargs
    ):

        if init_fn is None:
            init_fn = zero_mag_init_fn if zero_mag else random_spin_init_fn

        if proposal is None:
            proposal = SpinExchange() if zero_mag else SpinFlip()

        self.sampler = Sampler(*args, proposal=proposal, init_fn=init_fn, **kwargs)

    def init(self, log_prob: LogProbFn, warmup: int, key: Key) -> MCState:
        return self.sampler.init(log_prob, warmup, key)

    def __call__(self, logpsi: LogWaveFn, n_samples: int, state: MCState) -> tuple[Array, MCState]:
        log_prob = lambda *args, **kwargs: 2 * jnp.real(logpsi(*args, **kwargs))
        return self.sampler(log_prob, n_samples, state)
