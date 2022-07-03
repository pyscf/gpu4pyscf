import time
import copy
import ctypes
import contextlib
import numpy as np
import scipy.linalg
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.scf import hf, jk, _vhf
from gpu4pyscf.lib.utils import patch_cpu_kernel

libgint = lib.load_library('libgint')
libgint.GINTbuild_jk.restype = ctypes.c_int
LMAX_ON_GPU = 3


def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None,
           verbose=None):
    '''Compute J, K matrices with CPU-GPU hybrid algorithm
    '''
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, verbose)
    if hermi != 1:
        raise NotImplementedError('JK-builder only supports hermitian density matrix')

    if vhfopt is None:
        vhfopt = _VHFOpt(mol, 'int2e').build()

    pmol = vhfopt.mol
    coeff = vhfopt.coeff
    dm0 = dm
    nao = dm0.shape[-1]
    dms = dm0.reshape(-1,nao,nao)
    dms = np.asarray(lib.einsum('nij,pi,qj->npq', dms, coeff, coeff), order='C')

    scripts = []
    vj = vk = None
    if with_j:
        vj = np.zeros(dms.shape).transpose(0, 2, 1)
        scripts.append('ji->s2kl')
    if with_k:
        vk = np.zeros(dms.shape).transpose(0, 2, 1)
        scripts.append('jk->s1il')

    if hermi == 1:
        scripts = [s.replace('s1', 's2') for s in scripts]

    dm_ctr_cond = np.max(
        [lib.condense('absmax', x, vhfopt.l_ctr_offsets) for x in dms], axis=0)
    if hermi != 1:
        dm_ctr_cond = (dm_ctr_cond + dm_ctr_cond.T) * .5
    ncptype = len(vhfopt.uniq_l_ctr)
    cp_idx, cp_jdx = np.tril_indices(ncptype)

    l_symb = lib.param.ANGULAR
    pair2bra, pair2ket = vhfopt.bas_pair2shls
    bas_pairs_locs = vhfopt.bas_pairs_locs
    log_qs = vhfopt.log_qs

    # adjust nbins according to the size of the system
    pairs_max = (bas_pairs_locs[1:] - bas_pairs_locs[:-1]).max()
    nbins = max(10, int(pairs_max//200000))
    if nbins > 10:
        log.debug('Set the number of buckets for s_index to %d', nbins)

    fn = libgint.GINTbuild_jk
    with _jkmatrix_cache(mol, dms, vhfopt, with_j, with_k) as jkcache:
        for cp_ij_id, log_q_ij in enumerate(log_qs):
            cpi = cp_idx[cp_ij_id]
            cpj = cp_jdx[cp_ij_id]
            li = vhfopt.uniq_l_ctr[cpi,0]
            lj = vhfopt.uniq_l_ctr[cpj,0]
            if li > LMAX_ON_GPU or lj > LMAX_ON_GPU:
                continue

            for cp_kl_id, log_q_kl in enumerate(log_qs[:cp_ij_id+1]):
                cpk = cp_idx[cp_kl_id]
                cpl = cp_jdx[cp_kl_id]
                lk = vhfopt.uniq_l_ctr[cpk,0]
                ll = vhfopt.uniq_l_ctr[cpl,0]
                if lk > LMAX_ON_GPU and ll > LMAX_ON_GPU:
                    continue

                t0 = time.perf_counter()
                # TODO: determine cutoff based on the relevant maximum value of dm blocks?
                cutoff = vhfopt.direct_scf_tol / max(
                    dm_ctr_cond[cpi,cpj], dm_ctr_cond[cpk,cpl],
                    dm_ctr_cond[cpi,cpk], dm_ctr_cond[cpj,cpk],
                    dm_ctr_cond[cpi,cpl], dm_ctr_cond[cpj,cpl])
                bins_locs_ij = _make_s_index_offsets(log_q_ij, nbins, cutoff)
                bins_locs_kl = _make_s_index_offsets(log_q_kl, nbins, cutoff)

                err = fn(vhfopt.bpcache, jkcache,
                         bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                         bins_locs_kl.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(nbins), ctypes.c_int(cp_ij_id), ctypes.c_int(cp_kl_id))
                if err != 0:
                    detail = f'CUDA Error for ({l_symb[li]}{l_symb[lj]}|{l_symb[lk]}{l_symb[ll]})'
                    raise RuntimeError(detail)
                log.debug1('(%s%s|%s%s) on GPU %.3fs',
                           l_symb[li], l_symb[lj], l_symb[lk], l_symb[ll],
                           time.perf_counter() - t0)
        if with_j:
            libgint.GINTfetch_j(vj.ctypes.data_as(ctypes.c_void_p), jkcache)
            # *2 because only the lower triangle part of dm was used in J contraction
            vj *= 2
            vj = vj + vj.transpose(0, 2, 1)
        if with_k:
            libgint.GINTfetch_k(vk.ctypes.data_as(ctypes.c_void_p), jkcache)
            vk = vk + vk.transpose(0, 2, 1)
    cput0 = log.timer_debug1('get_jk pass 1 on gpu', *cput0)

    h_shls = vhfopt.h_shls
    if h_shls:
        log.debug3('Integrals for %s functions on CPU', l_symb[LMAX_ON_GPU+1])
        shls_excludes = [0, h_shls[0]] * 4
        vs_h = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
                                 dms, 1, pmol._atm, pmol._bas, pmol._env,
                                 vhfopt=vhfopt, shls_excludes=shls_excludes)
        if with_j and with_k:
            vj += vs_h[0]
            vk += vs_h[1]
        elif with_j:
            vj += vs_h[0]
        else:
            vk += vs_h[0]
        cput0 = log.timer_debug1('get_jk pass 3 for l>4 basis on cpu', *cput0)

    pnao = dms.shape[-1]
    idx = np.tril_indices(pnao)
    if with_j:
        vj[:, idx[1], idx[0]] = vj[:, idx[0], idx[1]]
        vj = lib.einsum('npq,pi,qj->nij', vj, coeff, coeff)
        vj = vj.reshape(dm0.shape)

    if with_k:
        if hermi:
            vk[:, idx[1], idx[0]] = vk[:, idx[0], idx[1]]
        vk = lib.einsum('npq,pi,qj->nij', vk, coeff, coeff)
        vk = vk.reshape(dm0.shape)
    return vj, vk


def _get_jk(mf, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
            omega=None):
    if omega is not None:
        raise NotImplementedError('Range separated Coulomb integrals')
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mf)
    log.debug3('apply get_jk on gpu')
    if hasattr(mf, '_opt_gpu'):
        vhfopt = mf._opt_gpu
    else:
        vhfopt = _VHFOpt(mol, getattr(mf.opt, '_intor', 'int2e'),
                         getattr(mf.opt, 'prescreen', 'CVHFnrs8_prescreen'),
                         getattr(mf.opt, '_qcondname', 'CVHFsetnr_direct_scf'),
                         getattr(mf.opt, '_dmcondname', 'CVHFsetnr_direct_scf_dm'))
        vhfopt.build(mf.direct_scf_tol)
        mf._opt_gpu = vhfopt
    vj, vk = get_jk(mol, dm, hermi, vhfopt, with_j, with_k, omega,
                    verbose=log)
    log.timer('vj and vk on gpu', *cput0)
    return vj, vk


class RHF(hf.RHF):
    device = 'gpu'
    get_jk = patch_cpu_kernel(hf.RHF.get_jk)(_get_jk)


class _VHFOpt(_vhf.VHFOpt):
    def __init__(self, mol, intor, prescreen='CVHFnoscreen',
                 qcondname='CVHFsetnr_direct_scf', dmcondname=None):
        self.mol, self.coeff = basis_seg_contraction(mol)
        # Note mol._bas will be sorted in .build() method. VHFOpt should be
        # initialized after mol._bas updated.
        self._intor = intor
        self._prescreen = prescreen
        self._qcondname = qcondname
        self._dmcondname = dmcondname

    def build(self, cutoff=1e-13):
        cput0 = (logger.process_clock(), logger.perf_counter())
        mol = self.mol
        # Sort basis according to angular momentum and contraction patterns so
        # as to group the basis functions to blocks in GPU kernel.
        l_ctrs = mol._bas[:,[gto.ANG_OF, gto.NPRIM_OF]]
        uniq_l_ctr, uniq_bas_idx, inv_idx, l_ctr_counts = np.unique(
            l_ctrs, return_index=True, return_inverse=True, return_counts=True, axis=0)
        if mol.verbose >= logger.DEBUG:
            logger.debug1(mol, 'Number of shells for each [l, nctr] group')
            for l_ctr, n in zip(uniq_l_ctr, l_ctr_counts):
                logger.debug(mol, '    %s : %s', l_ctr, n)

        sorted_idx = np.argsort(inv_idx).astype(np.int32)
        # Sort contraction coefficients before updating self.mol
        ao_loc = mol.ao_loc_nr(cart=True)
        nao = ao_loc[-1]
        # Some addressing problems in GPU kernel code
        assert nao < 32768
        ao_idx = np.array_split(np.arange(nao), ao_loc[1:-1])
        ao_idx = np.hstack([ao_idx[i] for i in sorted_idx])
        self.coeff = self.coeff[ao_idx]
        # Sort basis inplace
        mol._bas = mol._bas[sorted_idx]

        # Initialize vhfopt after reordering mol._bas
        _vhf.VHFOpt.__init__(self, mol, self._intor, self._prescreen,
                             self._qcondname, self._dmcondname)
        self.direct_scf_tol = cutoff

        lmax = uniq_l_ctr[:,0].max()
        nbas_by_l = [l_ctr_counts[uniq_l_ctr[:,0]==l].sum() for l in range(lmax+1)]
        l_slices = np.append(0, np.cumsum(nbas_by_l))
        if lmax >= LMAX_ON_GPU:
            self.g_shls = l_slices[LMAX_ON_GPU:LMAX_ON_GPU+2].tolist()
        else:
            self.g_shls = []
        if lmax > LMAX_ON_GPU:
            self.h_shls = l_slices[LMAX_ON_GPU+1:].tolist()
        else:
            self.h_shls = []

        # TODO: is it more accurate to filter with overlap_cond (or exp_cond)?
        q_cond = self.get_q_cond()
        cput1 = logger.timer(mol, 'Initialize q_cond', *cput0)
        log_qs = []
        pair2bra = []
        pair2ket = []
        l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        for i, (p0, p1) in enumerate(zip(l_ctr_offsets[:-1], l_ctr_offsets[1:])):
            if uniq_l_ctr[i,0] > LMAX_ON_GPU:
                # no integrals with g functions should be evaluated on GPU
                continue

            for q0, q1 in zip(l_ctr_offsets[:i], l_ctr_offsets[1:i+1]):
                q_sub = q_cond[p0:p1,q0:q1].ravel()
                idx = q_sub.argsort(axis=None)[::-1]
                q_sorted = q_sub[idx]
                mask = q_sorted > cutoff

                idx = idx[mask]
                ishs, jshs = np.unravel_index(idx, (p1-p0, q1-q0))
                ishs += p0
                jshs += q0
                pair2bra.append(ishs)
                pair2ket.append(jshs)

                log_q = np.log(q_sorted[mask])
                log_q[log_q > 0] = 0
                log_qs.append(log_q)

            q_sub = q_cond[p0:p1,p0:p1].ravel()
            idx = q_sub.argsort(axis=None)[::-1]
            q_sorted = q_sub[idx]
            ishs, jshs = np.unravel_index(idx, (p1-p0, p1-p0))
            mask = (ishs >= jshs) & (q_sorted > cutoff)

            ishs = ishs[mask]
            jshs = jshs[mask]
            ishs += p0
            jshs += p0
            pair2bra.append(ishs)
            pair2ket.append(jshs)

            log_q = np.log(q_sorted[mask])
            log_q[log_q > 0] = 0
            log_qs.append(log_q)

        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = l_ctr_offsets
        self.bas_pair2shls = np.hstack(
            pair2bra + pair2ket).astype(np.int32).reshape(2,-1)
        self.bas_pairs_locs = np.append(
            0, np.cumsum([x.size for x in pair2bra])).astype(np.int32)
        self.log_qs = log_qs

        ao_loc = mol.ao_loc_nr(cart=True)
        ncptype = len(log_qs)
        self.bpcache = ctypes.POINTER(BasisProdCache)()
        libgint.GINTinit_basis_prod(
            ctypes.byref(self.bpcache),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            self.bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
            self.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ncptype),
            mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            mol._env.ctypes.data_as(ctypes.c_void_p))
        logger.timer(mol, 'Initialize GPU cache', *cput1)
        return self

    def clear(self):
        _vhf.VHFOpt.__del__(self)
        libgint.GINTdel_basis_prod(ctypes.byref(self.bpcache))
        return self

    def __del__(self):
        try:
            self.clear()
        except AttributeError:
            pass

class BasisProdCache(ctypes.Structure):
    pass

class JKMatrixCache(ctypes.Structure):
    pass

@contextlib.contextmanager
def _jkmatrix_cache(mol, dms, vhfopt, with_j, with_k):
    try:
        jkcache = ctypes.POINTER(JKMatrixCache)()
        libgint.GINTinit_jkmatrix_cache(
            ctypes.byref(jkcache), vhfopt.bpcache,
            dms.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(dms.shape[0]), ctypes.c_int(dms.shape[1]),
            ctypes.c_int(with_j), ctypes.c_int(with_k))
        yield jkcache
    finally:
        libgint.GINTdel_jkmatrix_cache(ctypes.byref(jkcache))

def basis_seg_contraction(mol, allow_replica=False):
    '''transform generally contracted basis to segment contracted basis

    Kwargs:
        allow_replica:
            transform the generally contracted basis to replicated
            segment-contracted basis
    '''
    bas_templates = {}
    _bas = []
    _env = mol._env.copy()
    contr_coeff = []

    aoslices = mol.aoslice_by_atom()
    for ia, (ib0, ib1) in enumerate(aoslices[:,:2]):
        key = tuple(mol._bas[ib0:ib1,gto.PTR_EXP])
        if key in bas_templates:
            bas_of_ia, coeff = bas_templates[key]
            bas_of_ia = bas_of_ia.copy()
            bas_of_ia[:,gto.ATOM_OF] = ia
        else:
            # Generate the template for decontracted basis
            coeff = []
            bas_of_ia = []
            for shell in mol._bas[ib0:ib1]:
                l = shell[gto.ANG_OF]
                nf = (l + 1) * (l + 2) // 2
                nctr = shell[gto.NCTR_OF]
                if nctr == 1:
                    bas_of_ia.append(shell)
                    coeff.append(np.eye(nf))
                    continue

                # Only basis with nctr > 1 needs to be decontracted
                nprim = shell[gto.NPRIM_OF]
                pcoeff = shell[gto.PTR_COEFF]
                if allow_replica:
                    coeff.extend([np.eye(nf)] * nctr)
                    bs = np.repeat(shell[np.newaxis], nctr, axis=0)
                    bs[:,gto.NCTR_OF] = 1
                    bs[:,gto.PTR_COEFF] = np.arange(pcoeff, pcoeff+nprim*nctr, nprim)
                    bas_of_ia.append(bs)
                else:
                    pexp = shell[gto.PTR_EXP]
                    exps = _env[pexp:pexp+nprim]
                    norm = gto.gto_norm(l, exps)
                    # remove normalization from contraction coefficients
                    c = _env[pcoeff:pcoeff+nprim*nctr].reshape(nctr,nprim)
                    c = np.einsum('ip,p,ef->iepf', c, 1/norm, np.eye(nf))
                    coeff.append(c.reshape(nf*nctr, nf*nprim).T)

                    _env[pcoeff:pcoeff+nprim] = norm
                    bs = np.repeat(shell[np.newaxis], nprim, axis=0)
                    bs[:,gto.NPRIM_OF] = 1
                    bs[:,gto.NCTR_OF] = 1
                    bs[:,gto.PTR_EXP] = np.arange(pexp, pexp+nprim)
                    bs[:,gto.PTR_COEFF] = np.arange(pcoeff, pcoeff+nprim)
                    bas_of_ia.append(bs)

            bas_of_ia = np.vstack(bas_of_ia)
            bas_templates[key] = (bas_of_ia, coeff)

        _bas.append(bas_of_ia)
        contr_coeff.extend(coeff)

    pmol = copy.copy(mol)
    pmol.cart = True
    pmol._bas = np.asarray(np.vstack(_bas), dtype=np.int32)
    pmol._env = _env
    contr_coeff = scipy.linalg.block_diag(*contr_coeff)
    if not mol.cart:
        contr_coeff = contr_coeff.dot(mol.cart2sph_coeff())
    return pmol, contr_coeff

def _make_s_index_offsets(log_q, nbins=10, cutoff=1e-12):
    '''Divides the shell pairs to "nbins" collections down to "cutoff"'''
    scale = nbins / np.log(min(cutoff, .1))
    s_index = np.ceil(scale * log_q).astype(np.int32)
    bins = np.bincount(s_index)
    assert bins.max() < 65536 * 8
    if bins.size < nbins:
        bins = np.append(bins, np.zeros(nbins-bins.size, dtype=np.int32))
    else:
        bins = bins[:nbins]
    return np.append(0, np.cumsum(bins)).astype(np.int32)

del patch_cpu_kernel
