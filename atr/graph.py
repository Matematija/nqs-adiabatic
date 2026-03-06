from math import prod

from jax import numpy as jnp
from jax.tree_util import tree_leaves
from jax import Array

import equinox as eqx
from einops import rearrange

from .utils import PyTree, default_int_dtype, default_dtype


def _neighbor_list_regular(edges, num_nodes):
    edges = jnp.concatenate([edges, edges[:, ::-1]], axis=0)
    sorted_edgelist = edges[edges[:, 0].argsort()]
    return sorted_edgelist[:, 1].reshape(num_nodes, -1)


def _neighbor_list_general(edges, num_nodes):
    edges = jnp.concatenate([edges, edges[:, ::-1]], axis=0)
    sorted_edgelist = edges[edges[:, 0].argsort()]
    src, dst = jnp.unstack(sorted_edgelist, axis=1)
    _, split_idx = jnp.unique(src, return_index=True, size=num_nodes)
    return jnp.split(dst, split_idx[1:])


class Graph(eqx.Module):

    edges: Array
    nodes: PyTree
    edge_weights: Array
    num_edges: int = eqx.field(static=True)
    num_nodes: int = eqx.field(static=True)
    is_regular: bool = eqx.field(static=True, default=False)

    def __init__(
        self, edges: Array, nodes: PyTree, regular: bool = False, edge_weights: Array | None = None
    ):

        node_arrs = tree_leaves(nodes)
        num_nodes = len(node_arrs[0])

        if any(l.shape[0] != num_nodes for l in node_arrs):
            raise ValueError("Invalid node shape.")

        self.edges = edges.astype(default_int_dtype())
        self.nodes = nodes
        self.num_nodes = num_nodes
        self.num_edges = edges.shape[0]
        self.is_regular = regular

        if edge_weights is None:
            self.edge_weights = jnp.ones(self.num_edges, dtype=default_dtype())
        else:
            if edge_weights.shape != (self.num_edges,):
                raise ValueError(
                    f"edge_weights must have shape ({self.num_edges},), got {edge_weights.shape}."
                )
            self.edge_weights = edge_weights

    @property
    def neighbor_list(self):
        if self.is_regular:
            return _neighbor_list_regular(self.edges, self.num_nodes)
        else:
            return _neighbor_list_general(self.edges, self.num_nodes)


@eqx.filter_jit
def _cube_edges_and_weights(shape: tuple, bc: tuple[str, ...]) -> tuple[Array, Array]:

    num_nodes = prod(shape)
    idx = jnp.arange(num_nodes).reshape(shape)
    node_coords = rearrange(jnp.indices(shape), "d ... -> (...) d")

    edges_per_axis, weights_per_axis = [], []

    for i, bc_token in enumerate(bc):

        edgelist_i = jnp.stack([idx, jnp.roll(idx, -1, axis=i)], axis=-1)

        if bc_token == "o":
            edgelist_i = jnp.delete(edgelist_i, -1, axis=i)
            edge_i = edgelist_i.reshape(-1, 2)
            w = jnp.ones(edge_i.shape[0], dtype=default_dtype())

        elif bc_token == "a":
            edge_i = edgelist_i.reshape(-1, 2)
            is_boundary = node_coords[edge_i[:, 0], i] == shape[i] - 1
            w = jnp.where(is_boundary, -1.0, 1.0).astype(default_dtype())

        elif bc_token == "p":
            edge_i = edgelist_i.reshape(-1, 2)
            w = jnp.ones(edge_i.shape[0], dtype=default_dtype())

        else:
            raise ValueError(f"Invalid boundary condition {bc_token} for axis {i}.")

        edges_per_axis.append(edge_i)
        weights_per_axis.append(w)

    edges = jnp.concatenate(edges_per_axis, axis=0)
    edge_weights = jnp.concatenate(weights_per_axis)
    return edges, edge_weights


class Cube(Graph):

    shape: tuple[int, ...] = eqx.field(static=True)
    bc: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, shape: int | tuple, bc: str | tuple[str, ...] = "p"):

        if isinstance(shape, int):
            shape = (shape,)

        if isinstance(bc, str):
            bc = (bc,) * len(shape)

        bc = tuple(val.lower() for val in bc)

        if len(bc) != len(shape):
            raise ValueError(f"bc must have length {len(shape)}, got {len(bc)}.")

        if not all(b in ("p", "o", "a") for b in bc):
            raise ValueError(f"bc values must be 'p', 'o', or 'a', got {bc}.")

        edges, edge_weights = _cube_edges_and_weights(shape, bc)
        nodes = rearrange(jnp.indices(shape), "d ... -> (...) d")
        is_regular = all(b != "o" for b in bc)

        super().__init__(edges, nodes, regular=is_regular, edge_weights=edge_weights)
        self.shape = shape
        self.bc = bc

    @property
    def ndim(self):
        return len(self.shape)
