from math import prod

from jax import numpy as jnp
from jax.tree_util import tree_leaves
from jax import Array

import equinox as eqx
from einops import rearrange

from .utils import PyTree, default_int_dtype


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
    num_edges: int = eqx.field(static=True)
    num_nodes: int = eqx.field(static=True)
    is_regular: bool = eqx.field(static=True, default=False)

    def __init__(self, edges: Array, nodes: PyTree, regular: bool = False):

        node_arrs = tree_leaves(nodes)
        num_nodes = len(node_arrs[0])

        if any(l.shape[0] != num_nodes for l in node_arrs):
            raise ValueError("Invalid node shape.")

        self.edges = edges.astype(default_int_dtype())
        self.nodes = nodes
        self.num_nodes = num_nodes
        self.num_edges = edges.shape[0]
        self.is_regular = regular

    @property
    def neighbor_list(self):
        if self.is_regular:
            return _neighbor_list_regular(self.edges, self.num_nodes)
        else:
            return _neighbor_list_general(self.edges, self.num_nodes)


@eqx.filter_jit
def _cube_edges(shape, pbc):

    num_nodes = prod(shape)
    idx = jnp.arange(num_nodes).reshape(shape)

    neighbors = []

    for i in range(len(shape)):

        edgelist_i = jnp.stack([idx, jnp.roll(idx, -1, axis=i)], axis=-1)

        if not pbc:
            edgelist_i = jnp.delete(edgelist_i, -1, axis=i)

        neighbors.append(edgelist_i.reshape(-1, 2))

    return jnp.concatenate(neighbors, axis=0)


class Cube(Graph):

    shape: tuple[int, ...] = eqx.field(static=True)
    pbc: bool = eqx.field(static=True)

    def __init__(self, shape: int | tuple, pbc: bool = True):

        if isinstance(shape, int):
            shape = (shape,)

        edges = _cube_edges(shape, pbc)
        nodes = rearrange(jnp.indices(shape), "d ... -> (...) d")

        super().__init__(edges, nodes)
        self.shape = shape
        self.pbc = pbc
        self.is_regular = pbc

    @property
    def ndim(self):
        return len(self.shape)
