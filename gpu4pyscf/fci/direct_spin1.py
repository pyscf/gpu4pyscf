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

import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci import direct_spin1

from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import get_avail_mem


TILE = 32

code = r'''
#define TILE 32
extern "C" {
__global__
void _build_t1(const double *ci0, double *t1,
    long long strb0, long long na, long long nb, long long nnorb,
    const unsigned int *addra, const unsigned int *addrb,
    const signed char *signa, const signed char *signb)
{
    long long vec = blockIdx.z;
    ci0 += vec * na * nb;
    t1 += vec * nnorb * na * TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    long long stra0 = (long long)blockIdx.y * blockDim.y;
    long long stra = stra0 + ty;
    long long strb = strb0 + tx;
    __shared__ unsigned int _addra[TILE*TILE];
    __shared__ unsigned int _addrb[TILE*TILE];
    __shared__ signed char _signa[TILE*TILE];
    __shared__ signed char _signb[TILE*TILE];

    for (long long j0 = 0; j0 < nnorb; j0 += TILE) {
        long long ja = j0 + ty;
        long long ia = stra0 + tx;
        long long jb = j0 + ty;
        long long ib = strb0 + tx;
        int tile_idx = ty * TILE + tx;
        if (ja < nnorb && ia < na) {
            long long link_idx = ja * na + ia;
            _addra[tile_idx] = addra[link_idx];
            _signa[tile_idx] = signa[link_idx];
        } else {
            _addra[tile_idx] = 0;
            _signa[tile_idx] = 0;
        }
        if (jb < nnorb && ib < nb) {
            long long link_idx = jb * nb + ib;
            _addrb[tile_idx] = addrb[link_idx];
            _signb[tile_idx] = signb[link_idx];
        } else {
            _addrb[tile_idx] = 0;
            _signb[tile_idx] = 0;
        }
        __syncthreads();

        if (stra < na) {
            int dj = min((long long)TILE, nnorb - j0);
            for (int j = 0; j < dj; j++) {
                double val = 0.;
                if (strb < nb) {
                    int sign = _signa[j*TILE+ty];
                    unsigned int str1 = _addra[j*TILE+ty];
                    if (sign != 0) {
                        val = sign * ci0[(long long)str1 * nb + strb];
                    }

                    sign = _signb[j*TILE+tx];
                    str1 = _addrb[j*TILE+tx];
                    if (sign != 0) {
                        val += sign * ci0[stra * nb + str1];
                    }
                }
                long long t1_idx = ((j0+j) * na + stra) * TILE + tx;
                t1[t1_idx] = val;
            }
        }
        __syncthreads();
    }
}

__global__
void _gather(const double *t1, double *out,
    long long strb0, long long na, long long nb, long long nnorb,
    const unsigned int *addra, const unsigned int *addrb,
    const signed char *signa, const signed char *signb)
{
    long long vec = blockIdx.z;
    t1 += vec * nnorb * na * TILE;
    out += vec * na * nb;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    long long stra0 = (long long)blockIdx.y * blockDim.y;
    long long stra = stra0 + ty;
    long long strb = strb0 + tx;
    __shared__ unsigned int _addra[TILE*TILE];
    __shared__ unsigned int _addrb[TILE*TILE];
    __shared__ signed char _signa[TILE*TILE];
    __shared__ signed char _signb[TILE*TILE];
    double val = 0.;

    for (long long j0 = 0; j0 < nnorb; j0 += TILE) {
        long long ja = j0 + ty;
        long long ia = stra0 + tx;
        long long jb = j0 + ty;
        long long ib = strb0 + tx;
        int tile_idx = ty * TILE + tx;
        if (ja < nnorb && ia < na) {
            long long link_idx = ja * na + ia;
            _addra[tile_idx] = addra[link_idx];
            _signa[tile_idx] = signa[link_idx];
        } else {
            _addra[tile_idx] = 0;
            _signa[tile_idx] = 0;
        }
        if (jb < nnorb && ib < nb) {
            long long link_idx = jb * nb + ib;
            _addrb[tile_idx] = addrb[link_idx];
            _signb[tile_idx] = signb[link_idx];
        } else {
            _addrb[tile_idx] = 0;
            _signb[tile_idx] = 0;
        }
        __syncthreads();

        if (stra < na && strb < nb) {
            int dj = min((long long)TILE, nnorb - j0);
            for (int j = 0; j < dj; j++) {
                int sign = _signa[j*TILE+ty];
                unsigned int str1 = _addra[j*TILE+ty];
                if (sign != 0) {
                    val += sign * t1[((j0+j) * na + str1) * TILE + tx];
                }

                sign = _signb[j*TILE+tx];
                str1 = _addrb[j*TILE+tx];
                if (sign != 0) {
                    atomicAdd(out + stra * nb + str1,
                              sign * t1[((j0+j) * na + stra) * TILE + tx]);
                }
            }
        }
        __syncthreads();
    }
    if (stra < na && strb < nb) {
        atomicAdd(out + stra * nb + strb, val);
    }
}
}'''

_contract_2e_spin1 = cp.RawModule(code=code)
_build_t1 = _contract_2e_spin1.get_function('_build_t1')
_gather = _contract_2e_spin1.get_function('_gather')


def _link_index_to_addrs(link_index, nnorb):
    nstr = link_index.shape[0]
    pair = link_index[:, :, 0].T
    addr = np.zeros((nnorb, nstr), dtype=np.uint32)
    sign = np.zeros((nnorb, nstr), dtype=np.int8)
    idx = np.arange(nstr)
    addr[pair, idx] = link_index[:, :, 2].T
    sign[pair, idx] = link_index[:, :, 3].T
    return cp.asarray(addr), cp.asarray(sign)


def _prepare_link_index(link_index, norb):
    if len(link_index) == 4 and isinstance(link_index[0], cp.ndarray):
        return link_index

    nnorb = norb * (norb + 1) // 2
    link_indexa, link_indexb = link_index
    addra, signa = _link_index_to_addrs(link_indexa, nnorb)
    if link_indexa is link_indexb:
        addrb, signb = addra, signa
    else:
        addrb, signb = _link_index_to_addrs(link_indexb, nnorb)
    return addra, signa, addrb, signb


def _prepare_rdm_link_index(link_index, norb, nelec):
    if link_index is None:
        neleca, nelecb = direct_spin1._unpack_nelec(nelec)
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        if neleca == nelecb:
            link_indexb = link_indexa
        else:
            link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index

    nnorb = norb * norb

    def convert(link):
        full_link = link.copy()
        full_link[:, :, 0] = link[:, :, 1] * norb + link[:, :, 0]
        return _link_index_to_addrs(full_link, nnorb)

    addra, signa = convert(link_indexa)
    if link_indexa is link_indexb:
        addrb, signb = addra, signa
    else:
        addrb, signb = convert(link_indexb)
    return addra, signa, addrb, signb


def contract_2e(eri, ci0, norb, nelec, link_index=None):
    nelec = direct_spin1._unpack_nelec(nelec)
    if link_index is None:
        link_index = direct_spin1._unpack(norb, nelec, None)
    gpu_link_index = _prepare_link_index(link_index, norb)
    addra, signa, addrb, signb = gpu_link_index

    neleca, nelecb = nelec
    na = lib.comb(norb, neleca)
    nb = lib.comb(norb, nelecb)
    ci0 = cp.asarray(ci0, dtype=cp.float64, order='C')
    input_shape = ci0.shape
    single_vector = ci0.ndim == 1 or ci0.shape == (na, nb)
    if single_vector:
        nvec = 1
    elif ci0.ndim in (2, 3):
        nvec = ci0.shape[0]
    else:
        raise ValueError(f'Invalid CI shape {ci0.shape}')
    if ci0.size != nvec * na * nb:
        raise ValueError(
            f'Invalid CI size {ci0.size}; expected {nvec * na * nb}')
    ci0 = ci0.reshape(nvec, na, nb)

    nnorb = norb * (norb + 1) // 2
    eri = cp.asarray(eri, dtype=cp.float64, order='C')
    if eri.shape != (nnorb, nnorb):
        raise ValueError(f'Invalid ERI shape {eri.shape}; expected {(nnorb, nnorb)}')

    out = cp.zeros_like(ci0)
    t1 = cp.empty((nvec, nnorb, na * TILE), dtype=cp.float64)
    gt1 = cp.empty_like(t1)
    threads = (TILE, TILE)
    blocks = (1, (na + threads[1] - 1) // threads[1], nvec)
    rest_args = (na, nb, nnorb, addra, addrb, signa, signb)
    for strb0 in range(0, nb, TILE):
        _build_t1(blocks, threads, (ci0, t1, strb0) + rest_args)
        cp.matmul(eri, t1, out=gt1)
        _gather(blocks, threads, (gt1, out, strb0) + rest_args)
    return out[0] if single_vector else out.reshape(input_shape)


def make_rdm12(fcivec, norb, nelec, link_index=None, reorder=True):
    nelec = direct_spin1._unpack_nelec(nelec)
    neleca, nelecb = nelec
    na = lib.comb(norb, neleca)
    nb = lib.comb(norb, nelecb)
    fcivec = cp.asarray(fcivec, dtype=cp.float64, order='C')
    if fcivec.size != na * nb:
        raise ValueError(f'Invalid CI size {fcivec.size}; expected {na * nb}')
    fcivec = fcivec.reshape(na, nb)

    gpu_link_index = _prepare_rdm_link_index(link_index, norb, nelec)
    addra, signa, addrb, signb = gpu_link_index
    nnorb = norb * norb
    dm1 = cp.zeros(nnorb, dtype=cp.float64)
    dm2 = cp.zeros((nnorb, nnorb), dtype=cp.float64)
    t1 = cp.empty((nnorb, na * TILE), dtype=cp.float64)
    ci_tile = cp.zeros((na, TILE), dtype=cp.float64)
    threads = (TILE, TILE)
    blocks = (1, (na + threads[1] - 1) // threads[1])
    rest_args = (na, nb, nnorb, addra, addrb, signa, signb)

    for strb0 in range(0, nb, TILE):
        _build_t1(blocks, threads, (fcivec, t1, strb0) + rest_args)
        blen = min(TILE, nb - strb0)
        ci_tile.fill(0.)
        ci_tile[:, :blen] = fcivec[:, strb0:strb0 + blen]
        dm1 += cp.dot(t1, ci_tile.ravel())
        dm2 += cp.dot(t1, t1.T)

    dm1 = dm1.reshape(norb, norb).T
    dm2 = dm2.reshape(norb, norb, norb, norb).transpose(1, 0, 2, 3)
    if reorder:
        for k in range(norb):
            dm2[:, k, k, :] -= dm1.T
        dm2 = dm2.reshape(nnorb, nnorb)
        dm2 = (dm2 + dm2.T) * .5
        dm2 = dm2.reshape(norb, norb, norb, norb)
    return dm1, dm2


def _qr(vectors, lindep):
    vectors = cp.asarray(vectors).copy()
    nvec = 0
    for vector in vectors:
        for _ in range(2):
            if nvec:
                vector -= vectors[:nvec].conj().dot(vector).dot(
                    vectors[:nvec])
        norm = cp.linalg.norm(vector)
        if norm**2 > lindep:
            vectors[nvec] = vector / norm
            nvec += 1
    return vectors[:nvec]


def davidson1(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
              lindep=1e-14, max_memory=4000, nroots=1, pick=None,
              verbose=logger.WARN, tol_residual=None):
    """In-core GPU Davidson solver. Host-memory and disk spill are unsupported."""
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(verbose=verbose)

    if tol_residual is None:
        tol_residual = np.sqrt(tol)

    if isinstance(x0, (list, tuple)):
        x0 = cp.stack([cp.asarray(x) for x in x0])
    else:
        x0 = cp.asarray(x0)
    if x0.ndim == 1:
        x0 = x0[None]
    vector_size = x0.shape[1]
    if not 0 < nroots <= vector_size:
        raise ValueError(f'Invalid nroots {nroots} for vector size {vector_size}')

    vector_bytes = vector_size * x0.dtype.itemsize
    max_space = min(max_space + (nroots - 1) * 4, vector_size)
    work_vectors = max(5 * nroots, len(x0))
    required_memory = ((2 * max_space + work_vectors) * vector_bytes +
                       max_space**2 * x0.dtype.itemsize)
    available_memory = get_avail_mem()
    memory_limit = min(max_memory * 1e6, available_memory)
    if required_memory > memory_limit:
        raise MemoryError(
            f'GPU Davidson subspace requires '
            f'{required_memory / 1e6:.0f} MB in-core; '
            f'max_memory={max_memory:.0f} MB, '
            f'available GPU memory={available_memory / 1e6:.0f} MB')
    log.debug('Davidson max_space %d, in-core memory %.0f MB', max_space,
              required_memory / 1e6)

    try:
        xs = cp.empty((max_space, vector_size), dtype=x0.dtype)
        ax = cp.empty_like(xs)
        heff = cp.empty((max_space, max_space), dtype=x0.dtype)
    except cp.cuda.memory.OutOfMemoryError as err:
        raise MemoryError(
            f'Failed to allocate GPU Davidson subspace of size {max_space}') \
            from err

    converged = cp.zeros(nroots, dtype=bool)
    energy = previous_energy = None
    ritz = None
    trial = _qr(x0, lindep)
    x0 = None
    space = 0

    for cycle in range(max(1, max_cycle)):
        if len(trial) == 0:
            break
        if space + len(trial) > max_space:
            trial = trial[:max_space-space]

        try:
            atrial = aop(trial)
        except cp.cuda.memory.OutOfMemoryError as err:
            raise MemoryError(
                'Insufficient GPU memory for the Davidson Hamiltonian-vector '
                'product') from err
        old_space = space
        space += len(trial)
        xs[old_space:space] = trial
        ax[old_space:space] = atrial

        hsub = trial.conj().dot(ax[:space].T)
        heff[old_space:space, :space] = hsub
        heff[:old_space, old_space:space] = hsub[:, :old_space].T.conj()
        block = hsub[:, old_space:space]
        heff[old_space:space, old_space:space] = (
            block + block.T.conj()) * .5

        w, v = cp.linalg.eigh(heff[:space, :space])
        if pick is not None:
            w, v, _ = pick(w, v, nroots, locals())
        if len(w) < nroots:
            raise RuntimeError('Not enough eigenvalues')

        previous_energy, energy = energy, w[:nroots]
        coeff = v[:, :nroots].T
        ritz = coeff.dot(xs[:space])
        aritz = coeff.dot(ax[:space])
        residual = aritz - energy[:, None] * ritz
        residual_norm = cp.linalg.norm(residual, axis=1)
        if previous_energy is None:
            de = cp.full_like(energy, cp.inf)
        else:
            de = energy - previous_energy
        converged = ((cp.abs(de) < tol) &
                     (residual_norm < tol_residual))

        max_residual = residual_norm.max()
        max_de = cp.abs(de).max()
        if bool(cp.all(converged).get()):
            log.debug('converged %d %d  |r|= %.3g  e= %s  max|de|= %.3g',
                      cycle, space, max_residual, energy, max_de)
            break

        active = cp.where(
            ~converged & (residual_norm**2 > lindep))[0]
        if len(active) == 0:
            converged = residual_norm < tol_residual
            break

        trial = precond(residual[active], energy[active])
        for _ in range(2):
            trial -= xs[:space].conj().dot(trial.T).T.dot(xs[:space])
        trial = _qr(trial, lindep)
        log.debug('davidson %d %d  |r|= %.3g  e= %s  max|de|= %.3g',
                  cycle, space, max_residual, energy, max_de)
        if len(trial) == 0:
            converged = residual_norm < tol_residual
            break

        if space + len(trial) > max_space:
            trial = _qr(ritz, lindep)
            space = 0

    if ritz is None:
        raise RuntimeError('Davidson failed to build a trial subspace')
    return converged, cp.asnumpy(energy), ritz


class FCISolver(direct_spin1.FCISolver):
    from gpu4pyscf.lib.utils import to_cpu, to_gpu, device

    contract_2e = staticmethod(contract_2e)

    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, max_memory=None, verbose=None, ecore=None, **kwargs):
        if nroots is None:
            nroots = self.nroots
        if tol is None:
            tol = self.conv_tol
        if lindep is None:
            lindep = self.lindep
        if max_cycle is None:
            max_cycle = self.max_cycle
        if max_space is None:
            max_space = self.max_space
        if ecore is None:
            ecore = 0
        if isinstance(verbose, lib.logger.Logger):
            log = logger.Logger(verbose.stdout, verbose.verbose)
        else:
            log = logger.new_logger(self, verbose)
        fci_t0 = log.init_timer()
        nelec = direct_spin1._unpack_nelec(nelec, self.spin)
        h1e = cp.asnumpy(h1e) if isinstance(h1e, cp.ndarray) else np.asarray(h1e)
        eri = cp.asnumpy(eri) if isinstance(eri, cp.ndarray) else np.asarray(eri)
        hdiag = self.make_hdiag(h1e, eri, norb, nelec, compress=False).ravel()
        link_index = direct_spin1._unpack(norb, nelec, None)
        gpu_link_index = _prepare_link_index(link_index, norb)
        h2e = cp.asarray(self.absorb_h1e(h1e, eri, norb, nelec, .5))
        hdiag_gpu = cp.asarray(hdiag)

        if ci0 is None:
            ci0 = self.get_init_guess(norb, nelec, nroots, hdiag)
        elif callable(ci0):
            ci0 = ci0()

        hop_calls = 0
        hop_vectors = 0

        def hop(cis):
            nonlocal hop_calls, hop_vectors
            t0 = log.init_timer()
            out = contract_2e(
                h2e, cis, norb, nelec, gpu_link_index).reshape(len(cis), -1)
            hop_calls += 1
            hop_vectors += len(cis)
            log.timer_debug1(
                f'contract_2e for {len(cis)} CI vectors', *t0)
            return out

        def precond(residual, energy):
            denominator = hdiag_gpu - (
                energy[:, None] - self.level_shift)
            denominator = cp.where(
                cp.abs(denominator) < 1e-8, 1e-8, denominator)
            return residual / denominator

        tol_residual = getattr(self, 'conv_tol_residual', None)
        if tol_residual is None:
            tol_residual = np.sqrt(tol)
        if isinstance(ci0, (list, tuple)):
            ci0 = cp.stack([cp.asarray(x, dtype=cp.float64) for x in ci0])
        else:
            ci0 = cp.asarray(ci0, dtype=cp.float64)
        ci0 = ci0.reshape(-1, hdiag.size)
        if max_memory is None:
            max_memory = self.max_memory

        setup_t1 = log.timer('FCI setup', *fci_t0)
        setup_wall = setup_t1[1] - fci_t0[1]
        davidson_t0 = log.init_timer()
        converged, energies, ci = davidson1(
            hop, ci0, precond, tol=tol, tol_residual=tol_residual,
            lindep=lindep, nroots=nroots, max_cycle=max_cycle,
            max_space=max_space,
            max_memory=max_memory, verbose=log)
        davidson_t1 = log.timer('FCI Davidson', *davidson_t0)
        davidson_wall = davidson_t1[1] - davidson_t0[1]
        total_t1 = log.timer('GPU FCI solver', *fci_t0)
        total_wall = total_t1[1] - fci_t0[1]
        self.timing = {
            'total_wall': total_wall,
            'setup_wall': setup_wall,
            'davidson_wall': davidson_wall,
            'davidson_iterations': hop_calls,
            'davidson_avg_wall': davidson_wall / max(1, hop_calls),
            'contract_2e_calls': hop_calls,
            'contract_2e_vectors': hop_vectors,
        }
        log.debug('GPU FCI timing: total %.3f s; setup %.3f s; Davidson '
                  '%.3f s in %d iterations (%.3f s/iteration)',
                  total_wall, setup_wall, davidson_wall, hop_calls,
                  self.timing['davidson_avg_wall'])

        neleca, nelecb = nelec
        na = lib.comb(norb, neleca)
        nb = lib.comb(norb, nelecb)
        self.norb = norb
        self.nelec = nelec
        if nroots == 1:
            self.converged = bool(converged[0].get())
            self.eci = float(energies[0]) + ecore
            self.ci = ci[0].reshape(na, nb)
        else:
            self.converged = cp.asnumpy(converged)
            self.eci = energies + ecore
            self.ci = [root.reshape(na, nb) for root in ci[:nroots]]
        return self.eci, self.ci

    def energy(self, h1e, eri, fcivec, norb, nelec, link_index=None):
        h2e = self.absorb_h1e(h1e, eri, norb, nelec, .5)
        ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index)
        return cp.vdot(cp.asarray(fcivec).ravel(), ci1.ravel()).real

    def spin_square(self, fcivec, norb, nelec):
        return super().spin_square(cp.asnumpy(fcivec), norb, nelec)

    def make_rdm1s(self, fcivec, norb, nelec, link_index=None):
        return super().make_rdm1s(cp.asnumpy(fcivec), norb, nelec, link_index)

    def make_rdm1(self, fcivec, norb, nelec, link_index=None):
        return super().make_rdm1(cp.asnumpy(fcivec), norb, nelec, link_index)

    def make_rdm12(self, fcivec, norb, nelec, link_index=None, reorder=True):
        dm1, dm2 = make_rdm12(fcivec, norb, nelec, link_index, reorder)
        return cp.asnumpy(dm1), cp.asnumpy(dm2)


FCI = FCISolver
