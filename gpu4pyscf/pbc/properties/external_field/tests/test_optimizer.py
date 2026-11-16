"""Orbital CG unit tests with an analytic two-level model and stub callbacks."""
from types import SimpleNamespace
import cupy as cp
import numpy as np
import pytest
from gpu4pyscf.pbc.properties.external_field import optimizer


class ModelController:
    nocc=1
    max_cycle=60
    max_stepsize=.15
    converged=False
    _initial_branch='initial'

    def __init__(self):
        self.references=[]
        self.history=[]

    def evaluate_orbitals(self,c,reference=None):
        self.references.append(reference)
        f=cp.asarray([[[0.,0.],[0.,1.]]],dtype=cp.complex128)
        co=c[:,:,:1]
        dm=2*co@co.conj().transpose(0,2,1)
        energy=float(cp.trace(dm[0]@f[0]).real)
        g=4*c[:,:,1:].conj().transpose(0,2,1)@f@co
        return energy,energy,dm,f,SimpleNamespace(phases='trial'),g

    def reference_state(self,state):
        return 'accepted'

    def _record(self,cycle,state,old):
        self.history.append((state,old))

    def _is_converged(self,state,old):
        return float(cp.linalg.norm(state[5]))<1e-8


def test_cg_decreases_model_energy_and_preserves_metric():
    angle=.3
    c=cp.asarray([[[np.cos(angle),-np.sin(angle)],
                   [np.sin(angle),np.cos(angle)]]],dtype=cp.complex128)
    controller=ModelController()
    initial=controller.evaluate_orbitals(c)[0]
    controller.references=[]
    final,state=optimizer.minimize(controller,c)
    assert controller.converged and state[0]<initial
    cp.testing.assert_allclose(final.conj().transpose(0,2,1)@final,
                              cp.eye(2)[None],atol=1e-12)
    assert controller.references[0]=='initial'
    assert all(r=='accepted' for r in controller.references[1:])
    assert all(s[0]<=old[0]+1e-13 for s,old in controller.history)


def test_failed_search_keeps_last_accepted_orbitals(monkeypatch):
    c=cp.eye(2,dtype=cp.complex128)[None]
    state=(0.,0.,cp.eye(2)[None],cp.diag(cp.asarray([0.,1.]))[None],
           SimpleNamespace(phases='accepted'),cp.ones((1,1,1),dtype=cp.complex128))
    controller=SimpleNamespace(nocc=1,max_cycle=1,max_stepsize=.15,
        evaluate_orbitals=lambda *args,**kwargs:state,reference_state=lambda s:'accepted')
    def rotate(coeff,*args):
        return coeff+1,cp.eye(2)[None]
    monkeypatch.setattr(optimizer,'rotation',rotate)
    with pytest.raises(optimizer.LineSearchError) as error:
        optimizer.minimize(controller,c)
    assert error.value.coeff is c and error.value.state is state


def test_rotation_adapter_uses_local_helpers(monkeypatch):
    c=cp.eye(3,dtype=cp.complex128)[None]
    d=cp.asarray([[[.1+.2j],[.3-.1j]]])
    seen=[]
    def update(dx,occ):
        seen.append((dx,occ))
        return c
    monkeypatch.setattr(optimizer,'update_rotate_matrix',update)
    monkeypatch.setattr(optimizer,'rotate_mo',lambda coeff,u:coeff@u)
    rotated,u=optimizer.rotation(c,d,.5,1)
    cp.testing.assert_allclose(seen[0][0],(.5*d).ravel())
    cp.testing.assert_array_equal(seen[0][1],cp.asarray([[2.,0.,0.]]))
    cp.testing.assert_array_equal(rotated,c)


def test_local_rotation_helpers():
    from cupyx.scipy.linalg import expm
    # Unequal occupied counts exercise virtual/occupied rotation packing.
    occ=cp.asarray([[2.,0.,0.],[2.,2.,0.]])
    dx=cp.asarray([.01+.02j,-.03+.01j,.02-.01j,.04+.03j])
    a=cp.zeros((2,3,3),dtype=cp.complex128)
    a[0,1,0],a[0,2,0],a[1,2,0],a[1,2,1]=dx
    a-=a.conj().transpose(0,2,1)
    expected=cp.stack([expm(ak) for ak in a])
    u=optimizer.update_rotate_matrix(dx,occ)
    cp.testing.assert_allclose(u,expected,atol=1e-14)
    cp.testing.assert_allclose(u.conj().transpose(0,2,1)@u,
                              cp.broadcast_to(cp.eye(3),(2,3,3)),atol=1e-14)
    cp.testing.assert_allclose(optimizer.update_rotate_matrix(dx,occ,u0=u),
                              u@u,atol=1e-14)
    c=cp.arange(24).reshape(2,4,3).astype(cp.complex128)
    cp.testing.assert_allclose(optimizer.rotate_mo(c,u),c@u,atol=1e-13)
