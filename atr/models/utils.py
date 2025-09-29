from typing import Callable, Any
from functools import wraps

import jax
from jax import Array

import equinox as eqx

from ..utils import is_real, unpack_complex_tree, pack_complex_tree, tree_size
from ..utils import Ansatz, PyTree


def param_count(model: eqx.Module, is_param: Callable = eqx.is_inexact_array) -> int:
    return tree_size(model_params(model, is_param))


def model_params(model: Ansatz, is_param: Callable[[Any], bool] = eqx.is_inexact_array) -> PyTree:
    params = eqx.filter(model, is_param)
    return unpack_complex_tree(params)


def model_backbone(model: Ansatz, is_param: Callable[[Any], bool] = eqx.is_inexact_array) -> Any:
    return eqx.filter(model, is_param, inverse=True)


def model_partition(
    model: Ansatz, is_param: Callable[[Any], bool] = eqx.is_inexact_array
) -> tuple[PyTree, Any]:
    params, backbone = eqx.partition(model, is_param)
    return unpack_complex_tree(params), backbone


def model_combine(params: PyTree, backbone: Any) -> Ansatz:
    params = pack_complex_tree(params)
    return eqx.combine(params, backbone)


def model_split(model: Ansatz, is_param: Callable[[Any], bool] = eqx.is_inexact_array):

    params, backbone = eqx.partition(model, is_param)
    params = unpack_complex_tree(params)

    def eval_model(params, *args, **kwargs):
        params = pack_complex_tree(params)
        model = eqx.combine(params, backbone)
        return model(*args, **kwargs)

    return eval_model, params


def set_params(model: Ansatz, params: PyTree) -> Ansatz:
    params = pack_complex_tree(params)
    backbone = model_backbone(model)
    return eqx.combine(params, backbone)


class _RealWrapper(eqx.Module):

    params: PyTree
    backbone: Any = eqx.field(static=True)

    def __call__(self, *args, **kwargs) -> Array:
        params = pack_complex_tree(self.params)
        model = model_combine(params, self.backbone)
        return model(*args, **kwargs)


def as_real_params(
    model: Ansatz, is_param: Callable[[Any], bool] = eqx.is_inexact_array
) -> _RealWrapper:

    if isinstance(model, _RealWrapper):
        return model

    params, backbone = eqx.partition(model, is_param)

    if all(is_real(p) for p in jax.tree.leaves(params)):
        return model

    return _RealWrapper(unpack_complex_tree(params), backbone)


def real_params(model_cls):
    @wraps(model_cls)
    def wrapper(*args, **kwargs):
        model = model_cls(*args, **kwargs)
        return as_real_params(model)

    return wrapper
