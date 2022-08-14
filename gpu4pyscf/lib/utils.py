# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
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

import functools

def patch_cpu_kernel(cpu_kernel):
    '''Generate a decorator to patch cpu function to gpu function'''
    def patch(gpu_kernel):
        @functools.wraps(cpu_kernel)
        def hybrid_kernel(method, *args, **kwargs):
            if getattr(method, 'device', 'cpu') == 'gpu':
                return gpu_kernel(method, *args, **kwargs)
            else:
                return cpu_kernel(method, *args, **kwargs)
        hybrid_kernel.__package__ = 'gpu4pyscf'
        return hybrid_kernel
    return patch
