# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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

'''
gCP/SRB correction of the composite 3c methods (B97-3c, r2SCAN-3c)

The correction depends on the geometry only, so the CPU implementation of
pyscf.scf.gcp (PySCF >= 2.15) is used as is.
'''

try:
    from pyscf.scf.gcp import check_gcp, get_gcp
except ImportError:
    from pyscf import scf

    # PySCF < 2.15 has no gCP/SRB correction, and its parse_xc reads
    # 'r2scan-3c' and 'b97-3c' as the bare functionals. Refuse these methods
    # rather than run them without their correction.
    _GCP_XC = {'b97-3c', 'b97_3c', 'gga_xc_b97_3c', 'r2scan-3c', 'r2scan_3c'}

    def check_gcp(mf, gcp=None):
        if gcp is None:
            gcp = getattr(mf, 'gcp', None)
        if (gcp is not False and isinstance(mf, scf.hf.KohnShamDFT)
                and isinstance(mf.xc, str) and mf.xc.lower() in _GCP_XC):
            raise NotImplementedError(
                f'{mf.xc} requires PySCF >= 2.15 for its gCP/SRB correction')
        return False

    def get_gcp(mf, gcp=None, verbose=None):
        return 0.
