# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

import cupy as cp
import gpu4pyscf.lib as lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import pack_tril, unpack_tril, get_avail_mem

class PM6DIIS(lib.diis.DIIS):
    """
    Highly optimized DIIS tailored for PM6.
    Since the overlap matrix S = I, the error vector simplifies to FD - DF,
    eliminating the need for expensive diagonalizations and basis transformations.
    """
    incore = None

    def __init__(self, mf=None, filename=None):
        super().__init__(mf, filename)
        self.rollback = False
        self.space = 8

    def update(self, s, d, f, *args, **kwargs):
        errvec = self._sdf_err_vec(d, f)
        
        if self.incore is None:
            mem_avail = get_avail_mem()
            self.incore = errvec.nbytes * 2 * (20 + self.space) < mem_avail
            if not self.incore:
                logger.debug(self, 'Large system detected. DIIS intermediates '
                             'are saved in the host memory')
        
        nao = f.shape[-1]
        
        errvec = pack_tril(errvec.reshape(-1, nao, nao))
        f_tril = pack_tril(f.reshape(-1, nao, nao))
        
        xnew = lib.diis.DIIS.update(self, f_tril, xerr=errvec)
        
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
            
        return unpack_tril(xnew).reshape(f.shape)

    def get_num_vec(self):
        if self.rollback:
            return self._head
        else:
            return len(self._bookkeep)

    def _sdf_err_vec(self, d, f):
        """
        Calculate [F, D] = FD - DF.
        Since F and D are symmetric, (FD)^T = DF.
        Therefore, Error = FD - (FD)^T.
        This requires only ONE matrix multiplication!
        """
        fd = f @ d
        
        if f.ndim == 2: # RHF
            errvec = fd - fd.conj().T
        else:           # UHF
            errvec = fd - fd.conj().transpose(0, 2, 1)
            
        return errvec