# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Homogeneous periodic electric fields for GPU KRKS.

apply(mf, field=..., solver='scf') reuses the GPU KRKS SCF driver.
solver='cg' or orbital_optimizer=callback minimizes the electric enthalpy.
"""
from .scf import apply, ExternalField, ExternalFieldSCF
from .field_state import FieldState, FieldResult
__all__ = ['apply', 'ExternalField', 'ExternalFieldSCF', 'FieldState',
           'FieldResult']
