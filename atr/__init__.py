from .linalg import linear_solve
from .operator import LocalOperator, apply_operator, operator_expect, local_operator_expect
from .jacobian import natural_gradient
from .transport import inverse_power_update
from .fidelity import Fidelity
from .utils import vmap, value_and_grad, filter_vmap, filter_value_and_grad
