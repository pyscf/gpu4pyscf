import numpy as np
import cupy as cp
from gpu4pyscf.fci.direct_spin1 import contract_2e
from pyscf.fci import cistring, direct_spin1

def test_contract_2e():
    norb = 12
    nelec = 12
    npair = norb * (norb + 1) // 2
    np.random.seed(np.asarray(12, np.uint64))
    g2e = np.random.rand(npair,npair)
    g2e = g2e + g2e.T
    link = cistring.gen_linkstr_index(range(norb), nelec//2, tril=True)
    na = link.shape[0]
    cp.random.seed(np.asarray(11, np.uint64))
    ci0 = cp.random.rand(na)
    ci0 = cp.einsum('i,j->ij', ci0, ci0)
    ci0 *= 1/cp.linalg.norm(ci0)

    ci1 = contract_2e(g2e, ci0, norb, nelec, (link, link))

    ci0 = ci0.get()
    ref = direct_spin1.contract_2e(g2e, ci0, norb, nelec, (link, link))
    assert abs(ci1 - ref).max() < 1e-12
