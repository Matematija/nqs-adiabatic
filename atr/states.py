import abc
from typing import Literal

import jax
from jax import numpy as jnp
from jax import Array
import itertools

import equinox as eqx

from .operator import XX, X
from .graph import Cube, Graph
from .operator.spin import to_dense, to_index
from .utils import default_complex_dtype, default_int_dtype, DType


_idx_dtype = default_int_dtype()


def _sort_basis(basis: Array) -> Array:
    idxs = jax.vmap(to_index)(basis)
    return basis[jnp.argsort(idxs)]


def _maybe_reshape(arr, graph):

    if hasattr(graph, "shape"):
        arr = arr.reshape(-1, *graph.shape)

    return arr


def _standardize_basis(basis, graph):
    return _maybe_reshape(_sort_basis(basis), graph)


def ising_ground_state_basis(graph: Graph) -> Array:
    basis = jnp.ones((2, graph.num_nodes), dtype=_idx_dtype)
    basis = basis.at[0].set(-1)
    return _standardize_basis(basis, graph)


def transverse_field_ground_state_basis(graph: Graph) -> Array:
    basis = jnp.ones((1, graph.num_nodes), dtype=_idx_dtype)
    return _standardize_basis(basis, graph)


def _excited_basis_1d_obc(n_spins):
    basis = jnp.ones((n_spins - 1, n_spins), dtype=_idx_dtype)
    return basis.at[jnp.tril_indices_from(basis)].set(-1)


def _excited_basis_1d_pbc(n_spins):

    start_idx, stop_idx = jnp.triu_indices(n_spins, 1)
    n_states = len(start_idx)
    basis = jnp.ones((n_states, n_spins), dtype=_idx_dtype)

    idx = jnp.arange(n_states)
    basis = basis.at[idx, start_idx].mul(-1).at[idx, stop_idx].mul(-1)
    return jnp.cumprod(basis, axis=1)


def _excited_basis_2d_obc(nx, ny):
    basis = jnp.ones((4, nx, ny), dtype=_idx_dtype)
    return basis.at[0, 0, 0].set(-1).at[1, 0, -1].set(-1).at[2, -1, 0].set(-1).at[3, -1, -1].set(-1)


def _excited_basis_2d_pbc(nx, ny):

    basis = jnp.ones((nx * ny, nx, ny), dtype=_idx_dtype)

    ib = jnp.arange(nx * ny)
    ix, iy = jnp.unravel_index(ib, (nx, ny))

    return basis.at[ib, ix, iy].set(-1)


def ising_excitation_basis(graph: Cube) -> Array:

    if not isinstance(graph, Cube):
        raise NotImplementedError("Ising models are supported only on `Cube` graphs.")

    if graph.ndim == 1:
        if graph.pbc:
            basis = _excited_basis_1d_pbc(*graph.shape)
        else:
            basis = _excited_basis_1d_obc(*graph.shape)
    elif graph.ndim == 2:
        if graph.pbc:
            basis = _excited_basis_2d_pbc(*graph.shape)
        else:
            basis = _excited_basis_2d_obc(*graph.shape)
    else:
        raise NotImplementedError("Ising models are supported only on 1D and 2D graphs.")

    return _standardize_basis(basis, graph)


def transverse_field_excitation_basis(graph: Graph) -> Array:
    basis = jnp.ones((graph.num_nodes, graph.num_nodes), dtype=_idx_dtype)
    basis = basis.at[jnp.diag_indices(graph.num_nodes)].set(-1)
    return _standardize_basis(basis, graph)


def transverse_field_second_excitation_basis(graph: Graph) -> Array:
    N = graph.num_nodes
    base_state = jnp.ones(N, dtype=_idx_dtype)

    flip_pairs = list(itertools.combinations(range(N), 2))
    basis_states = []
    for i, j in flip_pairs:
        state = base_state.at[jnp.array([i, j])].set(-1)
        basis_states.append(state)

    basis = jnp.stack(basis_states)
    return _standardize_basis(basis, graph)


def hadamard_element(left: Array, right: Array) -> Array:

    bits_in = (1 + left.astype(_idx_dtype)) // 2
    bits_out = (1 + right.astype(_idx_dtype)) // 2
    bits = jnp.logical_or(bits_in, bits_out).astype(_idx_dtype)

    return 2 * bits - 1


def z2_expand(x: Array) -> Array:
    expanded = jnp.concatenate([x, -x], axis=0)
    return _sort_basis(expanded)


####################################################################################################


def _dtype_converter(dtype):
    return dtype or default_complex_dtype()


class Spectrum(eqx.Module):

    dtype: DType = eqx.field(static=True, converter=_dtype_converter)

    @abc.abstractmethod
    def ground_state(self, x: Array) -> Array:
        pass

    @abc.abstractmethod
    def excited_state(self, x: Array) -> Array:
        pass


class TransverseFieldSpectrum(Spectrum):

    graph: Graph
    excited_basis: Array
    unitary: Array | None = None
    energy_shifts: Array | None = None
    second_excited_basis: Array | None = None
    unitary_second: Array | None = None

    def __init__(
        self,
        graph: Graph,
        corrected: bool = True,
        build_second_excited_basis: bool = False,
        dtype: DType | None = None,
    ):

        super().__init__(dtype)

        self.graph = graph
        self.excited_basis = transverse_field_excitation_basis(graph)

        if build_second_excited_basis:
            self.second_excited_basis = transverse_field_second_excitation_basis(graph)

        if corrected:

            vmat = to_dense(-XX(graph), self.excited_basis)
            dE, U = jnp.linalg.eigh(vmat)
            self.unitary = U.T.conj().astype(self.dtype)
            self.energy_shifts = dE.astype(self.dtype)

            if build_second_excited_basis:
                vmat_second = to_dense(-XX(graph), self.second_excited_basis)
                _, U_second = jnp.linalg.eigh(vmat_second)
                self.unitary_second = U_second.T.conj().astype(self.dtype)

    @property
    def is_corrected(self) -> bool:
        return self.unitary is not None

    def ground_state(self, x: Array) -> Array:
        return jnp.array(1, dtype=self.dtype)  # / 2 ** (x.size / 2)

    def excited_state(self, x: Array) -> Array:

        psi = x.ravel().astype(self.dtype)  # / 2 ** (x.size / 2)

        if self.is_corrected:
            psi = self.unitary @ psi

        return psi

    def second_excited_state(self, x: Array) -> Array:
        y_flat = self.second_excited_basis.reshape(self.second_excited_basis.shape[0], -1)
        x_flat = x.reshape(1, -1)
        x_bitstring = ((x_flat + 1) // 2).astype(int)
        y_bitstring = ((y_flat - 1) // 2).astype(int)
        psi = (-1) ** (jnp.sum(x_bitstring * y_bitstring, axis=1) % 2)
        if self.is_corrected:
            psi = self.unitary_second @ psi
        return psi


class IsingSpectrum(Spectrum):

    graph: Graph
    ground_basis: Array
    excited_basis: Array

    is_expanded: bool = eqx.field(static=True)
    unitary: Array | None = eqx.field(default=None)

    def __init__(
        self,
        graph: Graph,
        corrected: bool = True,
        expanded: bool = True,
        dtype: DType | None = None,
    ):

        super().__init__(dtype)

        self.graph = graph
        self.ground_basis = ising_ground_state_basis(graph)
        self.excited_basis = ising_excitation_basis(graph)
        self.is_expanded = expanded

        if expanded:
            self.ground_basis = z2_expand(self.ground_basis)
            self.excited_basis = z2_expand(self.excited_basis)

        if corrected:
            vmat = to_dense(-X(graph), self.excited_basis)
            _, U = jnp.linalg.eigh(vmat)
            self.unitary = U.T.conj().astype(self.dtype)

    @property
    def is_corrected(self) -> bool:
        return self.unitary is not None

    def ground_state(
        self, x: Array, *, kind: Literal["singlet", "triplet", "+", "-"] = "+"
    ) -> Array:

        if kind == "singlet":
            plus = jnp.array(1, dtype=self.dtype)
            minus = (-1) ** x.size * jnp.prod(x).astype(self.dtype)
            return (plus - minus) / 2  # / 2 ** ((x.size + 1) / 2)
        elif kind == "triplet":
            plus = jnp.array(1, dtype=self.dtype)
            minus = (-1) ** x.size * jnp.prod(x).astype(self.dtype)
            return (plus + minus) / 2  # / 2 ** ((x.size + 1) / 2)
        elif kind == "+":
            return jnp.array(1, dtype=self.dtype)  # / 2 ** (x.size / 2)
        elif kind == "-":
            return (-1) ** x.size * jnp.prod(x).astype(self.dtype)  # / 2 ** (x.size / 2)
        else:
            raise ValueError(f"Invalid ground state kind: {kind}")

    def excited_state(self, x: Array) -> Array:

        vals = hadamard_element(x, self.excited_basis)
        axes = tuple(range(-x.ndim, 0))
        psi = jnp.prod(vals, axis=axes).astype(self.dtype)  # / 2 ** (x.size / 2)

        if self.is_corrected:
            psi = self.unitary @ psi

        return psi
