"""Finite-field mixin unit tests with a stub KRKS backend; no SCF integrals."""
from types import SimpleNamespace
import numpy as np
import cupy as cp
import pytest
from gpu4pyscf.pbc.dft.krks import KRKS
from gpu4pyscf.pbc.properties.external_field import apply
from gpu4pyscf.pbc.properties.external_field.scf import ExternalFieldSCF


class StubKRKS(KRKS):
    # Bypass the production kpts descriptor, which owns a density-fitting object.
    kpts=None

    def __init__(self):
        self.cell=SimpleNamespace(spin=0)
        self.kpts=np.zeros((1,3))
        self.conv_tol=1e-10
        self.conv_tol_grad=1e-8
        self._numint=object()
        self.converged=True
        self.max_cycle=3
        self.callback=None
        self.scf_summary={}

    def copy(self):
        obj=object.__new__(type(self))
        obj.__dict__.update(self.__dict__)
        return obj

    def istype(self,name):
        return name=='KRKS'

    def reset(self,cell=None):
        if cell is not None:
            self.cell=cell
        return self


@pytest.fixture
def mf(monkeypatch):
    from gpu4pyscf.pbc.properties.berry import kpt_string
    monkeypatch.setattr(kpt_string,'KPointMesh',lambda *args:None)
    return StubKRKS()


def test_apply_preserves_backend_and_undo(mf):
    ff=apply(mf,[0,0,.001])
    assert isinstance(ff,KRKS) and isinstance(ff,ExternalFieldSCF)
    assert ff._scf is not mf and ff._scf._numint is mf._numint
    assert ff.undo_external_field().__class__ is mf.__class__
    replaced=apply(ff,[.001,0,0],solver='cg')
    assert replaced.undo_external_field().__class__ is mf.__class__
    np.testing.assert_array_equal(replaced.field,[.001,0,0])


@pytest.mark.parametrize('field',[[np.nan,0,0],[0,0],[0,0,np.inf]])
def test_invalid_field(mf,field):
    with pytest.raises(ValueError):
        apply(mf,field)


def test_guards(mf):
    with pytest.raises(TypeError):
        apply(object())
    with pytest.raises(ValueError):
        apply(mf,solver='berry')
    mf.smearing_method='fermi'
    with pytest.raises(NotImplementedError):
        apply(mf)
    del mf.smearing_method
    mf.cell.spin=1
    with pytest.raises(NotImplementedError):
        apply(mf)
    mf.cell.spin=0
    ff=apply(mf)
    for method in ('Gradients','newton','stress','Hessian','get_bands','as_scanner','to_cpu','to_gpu'):
        with pytest.raises(NotImplementedError):
            getattr(ff,method)()


def test_reset_releases_state(mf):
    ff=apply(mf)
    called=[]
    state=SimpleNamespace(reset=lambda:called.append(True))
    ff._field_state=ff._functional=state
    ff._branch=(np.zeros(1),)
    ff.ao_links=object()
    ff.converged=True
    ff.reset()
    assert called and ff._field_state is None and ff._branch is None
    assert not hasattr(ff,'ao_links') and not ff.converged


def test_joint_convergence_and_user_check(mf):
    ff=apply(mf)
    env=dict(e_tot=1.,last_hf_e=1.,norm_gorb=0.,norm_ddm=0.)
    assert ff.check_convergence(env)
    for key,value in [('e_tot',1.+2*ff.conv_tol),
                      ('norm_gorb',2*ff.conv_tol_grad),('norm_ddm',2*ff.conv_tol_dm)]:
        assert not ff.check_convergence(dict(env,**{key:value}))
    ff._user_check_convergence=lambda env:False
    assert not ff.check_convergence(env)
    result=SimpleNamespace(polarization=np.zeros(3))
    state=(1.,1.,cp.zeros((1,2,2)),None,result,cp.zeros((1,1,1)))
    assert ff._is_converged(state,state)
    bad=list(state)
    bad[5]=cp.ones((1,1,1))
    assert not ff._is_converged(bad,state)


def test_cg_kernel_dispatch_and_failure_state(mf,monkeypatch):
    from gpu4pyscf.pbc.properties.external_field.optimizer import LineSearchError
    ff=apply(mf,[.001,0,0],solver='cg')
    coeff=cp.eye(2,dtype=cp.complex128)[None]
    occ=cp.asarray([[2.,0.]])
    initial=object()
    accepted=[]
    calls=[]
    monkeypatch.setattr(ff,'_initialize_field',lambda c,o:c)
    monkeypatch.setattr(ff,'evaluate_orbitals',lambda *args:initial)
    monkeypatch.setattr(ff,'_accept',lambda c,s:accepted.append((c,s)))
    def optimizer(controller,c):
        calls.append((controller,c))
        controller.converged=True
        controller.e_tot=-1.
        return c,initial
    ff.orbital_optimizer=optimizer
    assert ff.kernel(mo_coeff=coeff,mo_occ=occ)==-1.
    assert calls and len(accepted)==2 and ff.converged
    def fail(controller,c):
        raise LineSearchError(c,initial)
    ff.orbital_optimizer=fail
    with pytest.raises(LineSearchError):
        ff.kernel(mo_coeff=coeff,mo_occ=occ)
    assert accepted[-1][1] is initial and not ff.converged
