from typing import NamedTuple, Callable
from functools import partial
from math import prod

import jax
from jax import lax
from jax import numpy as jnp
from jax import Array

import equinox as eqx

from .utils import Scalar, Shape, DType, Key, default_dtype

LogProbFn = Callable[[Array], Scalar]
LogWaveFn = Callable[[Array], Scalar]
InitFn = Callable[[Shape, DType, Key], Array]
Proposal = Callable[[Array, Key], Array]


def mh_accept(log_prob: LogProbFn, x: Array, x_: Array, key: Key) -> bool:
    r = jnp.log(jax.random.uniform(key))
    return log_prob(x_) - log_prob(x) > r


class MCState(NamedTuple):

    x: Array
    accepted: Array
    key: Key

    @classmethod
    def initialize(cls, init_fn: InitFn, dims: Shape | int, dtype: DType, key: Key):
        key1, key2 = jax.random.split(key, 2)
        x0 = init_fn(dims, dtype, key1)
        accepted = jnp.array(True, dtype=jnp.bool_)
        return cls(x0, accepted, key2)


def step(log_prob: LogProbFn, proposal: Proposal, state: MCState) -> MCState:

    key1, key2, key3 = jax.random.split(state.key, 3)

    x_ = proposal(state.x, key1)
    accept = mh_accept(log_prob, state.x, x_, key2)
    x_new = jnp.where(accept, x_, state.x)

    return MCState(x_new, accept, key3)


def sweep(log_prob: LogProbFn, proposal: Proposal, state: MCState, n_steps: int) -> MCState:
    body_fn = lambda _, state: step(log_prob, proposal, state)
    return lax.fori_loop(0, n_steps, body_fn, state)


def sample_chain(
    log_prob: LogProbFn, proposal: Proposal, state: MCState, n_samples: int, sweep_size: int
) -> tuple[MCState, Array]:
    def body_fn(state, _):
        state = sweep(log_prob, proposal, state, sweep_size)
        return state, state.x

    return lax.scan(body_fn, init=state, xs=None, length=n_samples)


class Sampler(eqx.Module):

    proposal: Proposal
    init_fn: InitFn
    dims: Shape
    n_samples: int
    n_chains: int
    warmup: int
    sweep_size: int
    dtype: DType = eqx.field(static=True)

    def __init__(
        self,
        dims: int | Shape,
        n_samples: int,
        proposal: Proposal,
        init_fn: InitFn,
        *,
        n_chains: int = 1,
        warmup: int | None = None,
        sweep_size: int | None = None,
        dtype: DType | None = None
    ):

        if isinstance(dims, int):
            dims = (dims,)

        if sweep_size is None:
            sweep_size = prod(dims)

        if warmup is None:
            warmup = max(n_samples // 5, prod(dims))

        if dtype is None:
            dtype = default_dtype()

        self.proposal = proposal
        self.init_fn = init_fn
        self.dims = dims
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.warmup = warmup
        self.sweep_size = sweep_size
        self.dtype = dtype

    def _sample_chain(self, log_prob: LogProbFn, key: Key) -> Array:
        state = MCState.initialize(self.init_fn, self.dims, self.dtype, key)
        state = sweep(log_prob, self.proposal, state, self.warmup * self.sweep_size)  # warmup
        _, samples = sample_chain(log_prob, self.proposal, state, self.n_samples, self.sweep_size)
        return samples

    def __call__(self, log_prob: LogProbFn, key: Key) -> Array:
        keys = jax.random.split(key, self.n_chains)
        return jax.vmap(partial(self._sample_chain, log_prob))(keys)


class StateSampler(Sampler):
    def __call__(self, logpsi: LogWaveFn, key: Key) -> Array:
        log_prob = lambda *args, **kwargs: 2 * jnp.real(logpsi(*args, **kwargs))
        return super().__call__(log_prob, key)


###############################################################################################


def random_spin_init_fn(shape: Shape, dtype: DType, key: Key) -> Array:
    spins = jnp.array([-1, 1], dtype=dtype)
    return jax.random.choice(key, spins, shape)


def zero_mag_init_fn(shape: Shape, dtype: DType, key: Key) -> Array:

    n_spins = prod(shape)
    assert n_spins % 2 == 0, "Number of spins must be even for zero magnetization"

    flip_idxs = jax.random.choice(key, n_spins, (n_spins // 2,), replace=False)

    return jnp.ones(n_spins, dtype=dtype).at[flip_idxs].set(-1).reshape(shape)


def random_spin_flip(x: Array, key: Key) -> Array:
    shape = x.shape
    x = x.ravel()
    i = jax.random.randint(key, (1,), 0, x.size)
    return x.at[i].multiply(-1).reshape(shape)


def random_spin_exchange(x: Array, key: Key) -> Array:
    shape = x.shape
    x = x.ravel()
    i, j = jax.random.choice(key, x.size, (2,), replace=False)
    return x.at[i].set(x[j]).at[j].set(x[i]).reshape(shape)


class _SpinSampler(eqx.Module):
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
            proposal = random_spin_exchange if zero_mag else random_spin_flip

        super().__init__(*args, proposal=proposal, init_fn=init_fn, **kwargs)


class SpinSampler(_SpinSampler, StateSampler):
    pass
