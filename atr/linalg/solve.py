from typing import Any, Literal, Iterable

import jax
from jax import numpy as jnp
from jax import lax
from jax.scipy.linalg import svd, eigh
from jax import Array

import lineax as lx

from lineax._solver.misc import (
    ravel_vector,
    pack_structures,
    unravel_solution,
)  # TO DO: Try not relying on lineax internals

from ..utils import PyTree, Scalar


def _identity_like(op: lx.AbstractLinearOperator) -> lx.IdentityLinearOperator:
    return lx.IdentityLinearOperator(op.in_structure(), op.out_structure())


def _apply_diag_shift(op, shift):

    if shift is None:
        return op

    return op + shift * _identity_like(op)


def _get_preconditioner(matrix, preconditioner):

    if preconditioner == "normal":
        preconditioner = matrix.transpose()
        tags = [lx.positive_semidefinite_tag, lx.symmetric_tag]

    elif isinstance(preconditioner, lx.AbstractLinearOperator):
        if (
            preconditioner.in_structure() != matrix.out_structure()
            or preconditioner.out_structure() != matrix.in_structure()
        ):
            raise ValueError("Matrix and its transpose do not match.")

        tags = []

    else:
        raise ValueError("If given, preconditioner must be a linear operator or 'normal'.")

    return preconditioner, tags


def linear_solve(
    matrix: lx.AbstractLinearOperator,
    vector: PyTree,
    preconditioner: lx.AbstractLinearOperator | Literal["normal"] | None = None,
    solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
    tags: Iterable[Any] = (),
    options: dict[str, Any] = {},
    diag_shift: Scalar | None = None,
) -> Array:

    if preconditioner is None:
        sol = lx.linear_solve(matrix, vector, solver, options=options)
        return sol.value

    p_matrix, tags_ = _get_preconditioner(matrix, preconditioner)
    tags = tuple(set(tags).union(tags_))

    if matrix.out_size() >= matrix.in_size():
        S = _apply_diag_shift(p_matrix @ matrix, diag_shift)
        S = lx.TaggedLinearOperator(S, tags)
        sol = lx.linear_solve(S, p_matrix.mv(vector), solver, options=options)
        y = sol.value
    else:
        G = _apply_diag_shift(matrix @ p_matrix, diag_shift)
        G = lx.TaggedLinearOperator(G, tags)
        sol = lx.linear_solve(G, vector, solver, options=options)
        y = p_matrix.mv(sol.value)

    return y


#####################################################################################################


@jax.jit
def inv_smooth(s: Array, rcond: Scalar, acond: Scalar) -> Array:

    eps = jnp.finfo(s.dtype).eps

    acond = jnp.maximum(acond, eps)
    cutoff = jnp.maximum(acond, rcond * s.max())
    reg = jnp.reciprocal(1.0 + (cutoff / s) ** 6)
    s_inv_safe = jnp.where(s > eps, 1 / s, 0.0)

    return s_inv_safe * reg, jnp.sum(reg)  # Effective rank


def resolve_rcond(rcond, n, m, dtype):
    if rcond is None:
        return 2 * jnp.finfo(dtype).eps * max(n, m)
    else:
        return jnp.where(rcond < 0, jnp.finfo(dtype).eps, rcond)


def solve_soft_svd(
    u: Array, s: Array, vh: Array, vec: Array, rcond: Scalar, acond: Scalar
) -> tuple[Array, Scalar]:

    (m, _), (_, n) = u.shape, vh.shape

    rcond = resolve_rcond(rcond, n, m, s.dtype)
    rcond = jnp.asarray(rcond, dtype=s.dtype)

    s_inv, eff_rank = inv_smooth(s, rcond, acond)

    uhb = jnp.matmul(u.T.conj(), vec, precision=lax.Precision.HIGHEST)
    sol = jnp.matmul(vh.T.conj(), s_inv * uhb, precision=lax.Precision.HIGHEST)

    return sol, eff_rank


class SoftSpectralSolver(lx.SVD):

    rcond: Scalar = 1e-8
    acond: Scalar = 0.0

    def init(self, operator: lx.AbstractLinearOperator, options: dict[str, Any]):

        del options

        structures = pack_structures(operator)

        # if (operator.in_size() == operator.out_size()) and lx.is_positive_semidefinite(operator):
        if lx.is_symmetric(operator) and lx.is_positive_semidefinite(operator):
            s, U = eigh(operator.as_matrix())
            s, U = s[::-1], U[:, ::-1]
            svd_factors = (U, s, U.T.conj())
        else:
            svd_factors = svd(operator.as_matrix(), full_matrices=False)

        return svd_factors, structures

    def compute(
        self, state: Any, vector: PyTree, options: dict[str, Any]
    ) -> tuple[PyTree, lx.RESULTS, dict[str, Any]]:

        del options

        (u, s, vh), structures = state
        vec_flat = ravel_vector(vector, structures)

        sol, eff_rank = solve_soft_svd(u, s, vh, vec_flat, rcond=self.rcond, acond=self.acond)
        sol = unravel_solution(sol, structures)

        return sol, lx.RESULTS.successful, {"eff_rank": eff_rank}
