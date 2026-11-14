"""External-field tests: fast stub-based mixin unit tests and slow SCF checks."""
from types import SimpleNamespace
import numpy as np
import cupy as cp
import pytest
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.tools import pbc
from gpu4pyscf.pbc.dft.krks import KRKS
from gpu4pyscf.pbc.properties.external_field import apply
from gpu4pyscf.pbc.properties.external_field.scf import ExternalFieldSCF


# ---------------------------------------------------------------------------
# Fast unit tests: stubbed KRKS backend, no SCF integrals.
# ---------------------------------------------------------------------------
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
    ef=apply(mf,[0,0,.001])
    assert isinstance(ef,KRKS) and isinstance(ef,ExternalFieldSCF)
    assert ef._scf is not mf and ef._scf._numint is mf._numint
    assert ef.undo_external_field().__class__ is mf.__class__
    replaced=apply(ef,[.001,0,0],solver='cg')
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
    ef=apply(mf)
    for method in ('Gradients','newton','stress','Hessian','get_bands','as_scanner','to_cpu','to_gpu'):
        with pytest.raises(NotImplementedError):
            getattr(ef,method)()


def test_reset_releases_state(mf):
    ef=apply(mf)
    called=[]
    state=SimpleNamespace(reset=lambda:called.append(True))
    ef._field_state=ef._functional=state
    ef._branch=(np.zeros(1),)
    ef.ao_links=object()
    ef.converged=True
    ef.reset()
    assert called and ef._field_state is None and ef._branch is None
    assert not hasattr(ef,'ao_links') and not ef.converged


def test_joint_convergence_and_user_check(mf):
    ef=apply(mf)
    env=dict(e_tot=1.,last_hf_e=1.,norm_gorb=0.,norm_ddm=0.)
    assert ef.check_convergence(env)
    for key,value in [('e_tot',1.+2*ef.conv_tol),
                      ('norm_gorb',2*ef.conv_tol_grad),('norm_ddm',2*ef.conv_tol_dm)]:
        assert not ef.check_convergence(dict(env,**{key:value}))
    ef._user_check_convergence=lambda env:False
    assert not ef.check_convergence(env)
    result=SimpleNamespace(polarization=np.zeros(3))
    state=(1.,1.,cp.zeros((1,2,2)),None,result,cp.zeros((1,1,1)))
    assert ef._is_converged(state,state)
    bad=list(state)
    bad[5]=cp.ones((1,1,1))
    assert not ef._is_converged(bad,state)


def test_cg_kernel_dispatch_and_failure_state(mf,monkeypatch):
    from gpu4pyscf.pbc.properties.external_field.optimizer import LineSearchError
    ef=apply(mf,[.001,0,0],solver='cg')
    coeff=cp.eye(2,dtype=cp.complex128)[None]
    occ=cp.asarray([[2.,0.]])
    initial=object()
    accepted=[]
    calls=[]
    monkeypatch.setattr(ef,'_initialize_field',lambda c,o:c)
    monkeypatch.setattr(ef,'evaluate_orbitals',lambda *args:initial)
    monkeypatch.setattr(ef,'_accept',lambda c,s:accepted.append((c,s)))
    def optimizer(controller,c):
        calls.append((controller,c))
        controller.converged=True
        controller.e_tot=-1.
        return c,initial
    ef.orbital_optimizer=optimizer
    assert ef.kernel(mo_coeff=coeff,mo_occ=occ)==-1.
    assert calls and len(accepted)==2 and ef.converged
    def fail(controller,c):
        raise LineSearchError(c,initial)
    ef.orbital_optimizer=fail
    with pytest.raises(LineSearchError):
        ef.kernel(mo_coeff=coeff,mo_occ=occ)
    assert accepted[-1][1] is initial and not ef.converged


# ---------------------------------------------------------------------------
# Slow tests: small real periodic external-field SCF calculations.
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def helium():
    cell = gto.M(a=np.eye(3)*5, atom='He 1 1 1', basis='cc-pvdz',
                 unit='Bohr', verbose=0, precision=1e-8)
    old_threads, old_nproc = lib.num_threads(), pbc.nproc
    lib.num_threads(2)
    pbc.nproc = 1
    try:
        mf = KRKS(cell, cell.make_kpts([2,2,2]), xc='LDA,VWN')
        mf = mf.multigrid_numint()
        mf.conv_tol = 1e-12
        mf.kernel()
        assert mf.converged
        yield mf
    finally:
        lib.num_threads(old_threads)
        pbc.nproc = old_nproc


@pytest.mark.slow
def test_zero_field_reduces_to_krks(helium):
    ef = apply(helium, field=[0.,0.,0.]).run()
    assert ef.converged and isinstance(ef.mo_coeff, cp.ndarray)
    assert abs(ef.e_tot-helium.e_tot) < 1e-12
    cp.testing.assert_allclose(ef.make_rdm1(ef.mo_coeff, ef.mo_occ),
                               helium.make_rdm1(), atol=1e-12, rtol=0)
    assert abs(ef.e_field) < 1e-15


@pytest.mark.slow
@pytest.mark.parametrize('sign', [-1, 1])
def test_external_field_scf_cg_agreement_and_restart(helium, sign, tmp_path):
    results = []
    for solver in ('scf', 'cg'):
        ef = apply(helium, field=[0.,0.,sign*1e-4], solver=solver)
        ef.conv_tol = 1e-10
        ef.conv_tol_grad = 1e-9
        ef.conv_tol_dm = 1e-8
        ef.max_cycle = 150
        ef.run()
        assert ef.converged
        assert sign*ef.polarization()[2] > 0
        # The reported field energy must agree with the reported polarization.
        np.testing.assert_allclose(ef.e_field,
            -helium.cell.vol*np.dot(ef.field, ef.polarization()), atol=1e-14, rtol=0)
        results.append(ef)
    scf, cg = results
    np.testing.assert_allclose(scf.polarization(), cg.polarization(), atol=2e-10, rtol=0)
    cp.testing.assert_allclose(scf.make_rdm1(scf.mo_coeff, scf.mo_occ),
                               cg.make_rdm1(cg.mo_coeff, cg.mo_occ), atol=2e-8, rtol=0)
    assert abs(scf.e_tot-cg.e_tot) < 1e-12
    path = tmp_path/'branch.json'
    cg.save_branch(path)
    restored = apply(helium, field=cg.field, solver='scf').load_branch(path)
    restored.conv_tol_grad = 1e-9
    restored.conv_tol_dm = 1e-8
    restored.kernel(mo_coeff=cg.mo_coeff, mo_occ=cg.mo_occ)
    assert restored.converged
    np.testing.assert_allclose(restored.polarization(), cg.polarization(), atol=2e-10, rtol=0)
