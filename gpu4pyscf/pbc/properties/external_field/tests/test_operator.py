"""Synthetic link/operator and branch unit tests; no AO integrals or SCF."""
import numpy as np
import cupy as cp
import pytest
from cupyx.scipy.linalg import expm
from types import SimpleNamespace
from gpu4pyscf.pbc.properties.external_field import FieldState


def model(mesh=(3,2,1)):
    lattice=np.array([[6.,0,1.],[1.,7.,0],[0,1.,8.]])
    reciprocal=2*np.pi*np.linalg.inv(lattice).T
    fractional=np.array(list(np.ndindex(mesh)))/np.array(mesh)
    kpts=fractional@reciprocal
    cell=SimpleNamespace(dimension=3,charge=0,vol=np.linalg.det(lattice),
        lattice_vectors=lambda:lattice, reciprocal_vectors=lambda:reciprocal,
        get_scaled_kpts=lambda k:k@np.linalg.inv(reciprocal),
        atom_charges=lambda:np.array([2.]),atom_coords=lambda:np.array([[1.,2.,3.]]))
    nk,nao,nocc=len(kpts),4,2
    rng=np.random.default_rng(42)
    s=np.broadcast_to(np.diag([1,1.4,2,3]),(nk,nao,nao)).copy()
    c=np.stack([np.linalg.qr(np.eye(nao)+.04*(rng.normal(size=(nao,nao))+1j*rng.normal(size=(nao,nao))))[0] for _ in kpts])
    c=c/np.sqrt(np.diagonal(s,axis1=1,axis2=2))[:,:,None]
    links=np.broadcast_to(s,(3,nk,nao,nao)).astype(complex).copy()
    # Momentum-transfer operators with nontrivial Berry phase/field response.
    for d in range(3):
        a=rng.normal(size=(nao,nao))+1j*rng.normal(size=(nao,nao))
        a=(a+a.conj().T)*.04
        links[d]=cp.asnumpy(cp.sqrt(cp.asarray(s))@expm(cp.asarray(1j*a))@cp.sqrt(cp.asarray(s)))
    fun=FieldState(cell,kpts,[.001,-.0007,.0003],ao_links=links)
    return fun,cp.asarray(c),cp.asarray(s),nocc


def test_duals_hermitian_and_gauge():
    fun,c,s,nocc=model()
    r=fun.evaluate(c[:,:,:nocc],s)
    co=c[:,:,:nocc]
    for qp,qm in zip(r.dual_forward,r.dual_backward):
        cp.testing.assert_allclose(co.conj().swapaxes(-1, -2)@qp,np.broadcast_to(np.eye(nocc),(len(c),nocc,nocc)),atol=1e-12)
        cp.testing.assert_allclose(co.conj().swapaxes(-1, -2)@(qm-qp),0,atol=1e-12)
    cp.testing.assert_allclose(r.hamiltonian,r.hamiltonian.conj().swapaxes(-1, -2),atol=1e-13)
    cp.testing.assert_allclose(r.hamiltonian@co,r.gradient,atol=1e-13)
    rng=np.random.default_rng(7)
    u=np.stack([np.linalg.qr(rng.normal(size=(nocc,nocc))+1j*rng.normal(size=(nocc,nocc)))[0] for _ in c])
    rotated=fun.evaluate(co@cp.asarray(u),s,reference=r.phases)
    cp.testing.assert_allclose(rotated.polarization,r.polarization,atol=1e-13)
    cp.testing.assert_allclose(rotated.gradient,r.gradient@cp.asarray(u),atol=1e-12)


@pytest.mark.parametrize('mesh',[(3,2,1),(1,1,1),(2,2,2)])
def test_enthalpy_direction_derivative(mesh):
    fun,c,s,nocc=model(mesh)
    co=c[:,:,:nocc]
    r=fun.evaluate(co,s)
    rng=np.random.default_rng(32)
    a=cp.asarray(rng.normal(size=c.shape)+1j*rng.normal(size=c.shape))
    a=a-a.conj().swapaxes(-1, -2)
    a[:,:nocc,:nocc]=0
    a[:,nocc:,nocc:]=0
    dc=c@a
    h=1e-5
    plus=fun.evaluate((c@cp.stack([expm(h*x) for x in a]))[:,:,:nocc],s,reference=r.phases)
    minus=fun.evaluate((c@cp.stack([expm(-h*x) for x in a]))[:,:,:nocc],s,reference=r.phases)
    analytic=float(4/len(c)*cp.vdot(dc[:,:,:nocc],r.gradient).real)
    cp.testing.assert_allclose((plus.enthalpy-minus.enthalpy)/(2*h),analytic,rtol=1e-7,atol=1e-10)


def test_singular_and_invalid_mesh():
    fun,c,s,nocc=model()
    fun.ao_links[:]=0
    with pytest.raises(np.linalg.LinAlgError):
        fun.evaluate(c[:,:,:nocc],s)
    with pytest.raises(ValueError):
        FieldState(fun.cell,fun.topology.kpts[:-1],[0,0,0])



def test_trial_does_not_commit_branch():
    state,c,s,nocc=model()
    accepted=state.evaluate(c[:,:,:nocc],s)
    state.accept(accepted)
    branch=tuple(p.copy() for p in state.branch)
    state.evaluate(c[:,:,:nocc],s,reference=branch,with_hamiltonian=False)
    for actual,expected in zip(state.branch,branch):
        np.testing.assert_array_equal(actual,expected)
    assert state.result is accepted


def test_branch_roundtrip_and_reset(tmp_path):
    state,c,s,nocc=model()
    with pytest.raises(RuntimeError):
        state.save_branch(tmp_path/'branch.json',nocc)
    result=state.evaluate(c[:,:,:nocc],s)
    state.accept(result)
    path=tmp_path/'branch.json'
    state.save_branch(path,nocc)
    restored,_,_,_=model()
    restored.load_branch(path)
    for actual,expected in zip(restored.branch,state.branch):
        np.testing.assert_array_equal(actual,expected)
    assert restored.branch_nocc==nocc
    import json
    data=json.loads(path.read_text())
    data['kpts'][0][0]+=.1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        restored.load_branch(path)
    state.reset()
    assert state.ao_links is None and state.branch is None and state.result is None
