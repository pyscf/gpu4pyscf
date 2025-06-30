import cupy as cp
import numpy as np

import pyscf.pbc.grad.rhf as cpu_rhf
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.gto.pseudo import pp_int

import gpu4pyscf.grad.rhf as mol_rhf
from gpu4pyscf.lib.cupy_helper import return_cupy_array


class GradientsBase(mol_rhf.GradientsBase, cpu_rhf.GradientsBase):
    get_ovlp = cpu_rhf.GradientsBase.get_ovlp
    grad_nuc = return_cupy_array(cpu_rhf.GradientsBase.grad_nuc)


class Gradients(GradientsBase, mol_rhf.Gradients, cpu_rhf.Gradients):

    def get_veff(self, mol=None, dm=None, kpt=None, verbose=None):
        mf = self.base
        xc_code = getattr(mf, "xc", None)
        return mf.with_df.get_veff_ip1(dm, xc_code=xc_code, kpt=kpt)

    def grad_elec(
        self,
        mo_energy=None,
        mo_coeff=None,
        mo_occ=None,
        atmlst=None,
        kpts=np.zeros((1, 3)),
    ):
        mf = self.base
        mol = mf.mol

        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        dm0_cpu = dm0.get()

        dme0 = self.make_rdm1e(mo_energy, mo_coeff, mo_occ)

        if atmlst is None:
            atmlst = range(mol.natm)

        if not gamma_point(kpts):
            raise NotImplementedError

        de = mf.with_df.get_veff_ip1(dm0, xc_code=mf.xc)
        de -= mf.with_df.get_ovlp_ip1(dme0, kpts=kpts)
        
        de += cp.asarray(mf.with_df.vpploc_part1_nuc_grad(dm0_cpu))
        de += cp.asarray(pp_int.vpploc_part2_nuc_grad(mol, dm0_cpu))
        de += cp.asarray(pp_int.vppnl_nuc_grad(mol, dm0_cpu))
        core_hamiltonian_gradient = mol.pbc_intor("int1e_ipkin")
        kinetic_contribution = cpu_rhf._contract_vhf_dm(
            self, core_hamiltonian_gradient, dm0_cpu
        )
        de -= cp.asarray(kinetic_contribution) * 2

        return de
