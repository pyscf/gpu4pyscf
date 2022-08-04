from pyscf.scf import ghf
from gpu4pyscf.scf.hf import get_jk as _get_jk_nr
from gpu4pyscf.scf.hf import _eigh
from gpu4pyscf.lib.utils import patch_cpu_kernel

def get_jk(mol=None, dm=None, hermi=0, with_j=True, with_k=True,
           omega=None):
    if mol is None: mol = self.mol
    if dm is None: dm = self.make_rdm1()
    nao = mol.nao
    dm = numpy.asarray(dm)

    def jkbuild(mol, dm, hermi, with_j, with_k, omega=None):
        return _get_jk_nr(mf, mol, dm, hermi, with_j, with_k, omega)

    if nao == dm.shape[-1]:
        vj, vk = jkbuild(mol, dm, hermi, with_j, with_k, omega)
    else:  # GHF density matrix, shape (2N,2N)
        vj, vk = ghf.get_jk(mol, dm, hermi, with_j, with_k, jkbuild, omega)
    return vj, vk

class GHF(ghf.GHF):
    device = 'gpu'

    @patch_cpu_kernel(ghf.GHF.get_jk)
    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        return get_jk(mol, dm, hermi, with_j, with_k, omega)

    _eigh = patch_cpu_kernel(ghf.GHF._eigh)(_eigh)
