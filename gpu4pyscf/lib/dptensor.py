# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Intel GPU: @alvarovm

from gpu4pyscf.lib import logger
from importlib.util import find_spec

has_dpnp = find_spec("dpnp")

if has_dpnp:
    try:
        import dpnp

    except ImportError as e:
        raise ImportError("dpnp is installed, but could not be imported!") from e

contract_engine = 'dpnp'  # default contraction engine

# override the 'contract' function if einsum is customized or cutensor is not found
if contract_engine is not None:
    einsum = None
    if contract_engine == 'dpnp':
        einsum = dpnp.einsum
    else:
        raise RuntimeError('unknown tensor contraction engine.')

    import warnings
    warnings.warn(f'using {contract_engine} as the tensor contraction engine.')
    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        if out is None:
            return dpnp.asarray(einsum(pattern, a, b), order='C')
        else:
            out[:] = alpha*einsum(pattern, a, b) + beta*out
            return dpnp.asarray(out, order='C')