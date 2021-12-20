import ctypes
import numpy as np
import scipy.linalg
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.scf import hf, jk, _vhf
from pyscf.scf.hf import SCF, RHF

libgint = lib.load_library('libgint')
BUILDJ = 0b01
BUILDK = 0b10
TILESIZE = 8
LMAX_ON_GPU = 4
BLKSIZE_BY_L = [72, 72, 72, 80, 120]
MAX_BLKSIZE = 480


def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None):
    '''Compute J, K matrices with CPU-GPU hybrid algorithm
    '''
    cput0 = (logger.process_clock(), logger.perf_counter())
    if vhfopt is None:
        vhfopt = _VHFOpt(mol, 'int2e', 'CVHFnrs8_prescreen',
                         'CVHFsetnr_direct_scf', 'CVHFsetnr_direct_scf_dm')

    pmol = vhfopt.pmol
    slices = vhfopt.slices
    g_shls = list(vhfopt.g_shls)
    h_shls = list(vhfopt.h_shls)
    coeff = vhfopt.coeff

    dm0 = dm
    nao = dm0.shape[-1]
    dms = dm0.reshape(-1,nao,nao)
    n_dm = dms.shape[0]
    dms = np.asarray(lib.einsum('nij,pi,qj->npq', dms, coeff, coeff), order='C')

    if with_j and with_k:
        jk_type = BUILDJ | BUILDK
        vj, vk = vs = np.zeros((2,) + dms.shape).transpose(0, 1, 3, 2)
        scripts = ['ji->s2kl', 'jk->s1il']
    elif with_j:
        jk_type = BUILDJ
        vj = vs = np.zeros(dms.shape).transpose(0, 2, 1)
        vk = None
        scripts = ['ji->s2kl']
    elif with_k:
        jk_type = BUILDK
        vk = vs = np.zeros(dms.shape).transpose(0, 2, 1)
        vj = None
        scripts = ['jk->s1il']
    if hermi == 1:
        scripts = [s.replace('s1', 's2') for s in scripts]

    vhfopt.set_dm(dms, pmol._atm, pmol._bas, pmol._env)
    q_cond = vhfopt.get_q_cond()
    dm_cond = vhfopt.get_dm_cond()
    ao_loc = pmol.ao_loc_nr()

    fn = libgint.GPUbuild_jk
    n_slices = len(slices) - 1  # ~= lmax+1
    for li in range(n_slices):
        for lj in range(li+1):
            for lk in range(li+1):
                for ll in range(lk+1):
                    shls_slice = slices[li:li+2] + slices[lj:lj+2] + slices[lk:lk+2] + slices[ll:ll+2]
                    if (pmol.bas_angular(slices[li]) == LMAX_ON_GPU and
                        pmol.bas_angular(slices[lj]) == LMAX_ON_GPU and
                        pmol.bas_angular(slices[lk]) == LMAX_ON_GPU and
                        pmol.bas_angular(slices[ll]) == LMAX_ON_GPU):
                        continue
                    fn(vs.ctypes.data_as(ctypes.c_void_p),
                       dms.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n_dm),
                       ctypes.c_int(jk_type), ctypes.c_int(hermi),
                       (ctypes.c_int*8)(*shls_slice),
                       ao_loc.ctypes.data_as(ctypes.c_void_p),
                       (ctypes.c_int*5)(*BLKSIZE_BY_L),
                       q_cond.ctypes.data_as(ctypes.c_void_p),
                       dm_cond.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_double(vhfopt.direct_scf_tol),
                       pmol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pmol.natm),
                       pmol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pmol.nbas),
                       pmol._env.ctypes.data_as(ctypes.c_void_p))
    cput0 = logger.timer_debug1(mol, 'get_jk pass 1 on gpu', *cput0)

    # For gggg and l >= 5
    if g_shls and pmol.bas_angular(g_shls[0]) == LMAX_ON_GPU:
        logger.debug3(mol, 'Integrals (%s%s|%s%s) on CPU',
                     *[lib.param.ANGULAR[LMAX_ON_GPU]] * 4)
        shls_slice = g_shls + g_shls + g_shls + g_shls
        p0, p1 = ao_loc[g_shls]
        # Setting _dmcondname=None to temporarily disable vhfopt.set_dm
        # Otherwise jk.get_jk calls set_dm for dms_blk with improper dimensions
        with lib.temporary_env(vhfopt, _dmcondname=None):
            vs_g = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
                                     dms[:,p0:p1,p0:p1], 1,
                                     pmol._atm, pmol._bas, pmol._env,
                                     vhfopt=vhfopt, shls_slice=shls_slice)
        if with_j and with_k:
            vj[:,p0:p1,p0:p1] += vs_g[0]
            vk[:,p0:p1,p0:p1] += vs_g[1]
        else:
            vs[:,p0:p1,p0:p1] += vs_g[0]
        cput0 = logger.timer_debug1(mol, 'get_jk pass 2 for l=4 basis on cpu', *cput0)

    if h_shls:
        logger.debug3(mol, 'Integrals for %s functions on CPU',
                      lib.param.ANGULAR[LMAX_ON_GPU+1])
        nbas = pmol.nbas
        shls_excludes = [0, h_shls[0]] * 4
        vs_h = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
                                 dms, 1, pmol._atm, pmol._bas, pmol._env,
                                 vhfopt=vhfopt, shls_excludes=shls_excludes)
        if with_j and with_k:
            vj += vs_h[0]
            vk += vs_h[1]
        else:
            vs += vs_h[0]
        cput0 = logger.timer_debug1(mol, 'get_jk pass 3 for l>4 basis on cpu', *cput0)

    pnao = ao_loc[-1]
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


def __apply_gpu(cpu_kernel):
    def _get_jk(mf, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
                omega=None):
        if getattr(mf, 'device', 'cpu') == 'gpu':
            if omega is not None:
                raise NotImplementedError('Range separated Coulomb integrals')
            cput0 = (logger.process_clock(), logger.perf_counter())
            logger.debug3(mf, 'apply get_jk on gpu')
            if hasattr(mf, '_opt_gpu'):
                vhfopt = mf._opt_gpu
            else:
                vhfopt = _VHFOpt(mol, getattr(mf.opt, '_intor', 'int2e'),
                                 getattr(mf.opt, 'prescreen', 'CVHFnrs8_prescreen'),
                                 getattr(mf.opt, '_qcondname', 'CVHFsetnr_direct_scf'),
                                 getattr(mf.opt, '_dmcondname', 'CVHFsetnr_direct_scf_dm'))
                mf._opt_gpu = vhfopt
            vj, vk = get_jk(mol, dm, hermi, vhfopt, with_j, with_k, omega)
            logger.timer(mf, 'vj and vk on gpu', *cput0)
            return vj, vk

        logger.debug3(mf, 'apply get_jk on cpu')
        return cpu_kernel(mf, mol, dm, hermi, with_j, with_k, omega)
    return _get_jk

SCF.device = 'gpu'
SCF.get_jk = __apply_gpu(SCF.get_jk)
RHF.get_jk = __apply_gpu(RHF.get_jk)


class _VHFOpt(_vhf.VHFOpt):
    def __init__(self, mol, intor,
                 prescreen='CVHFnoscreen', qcondname=None, dmcondname=None):
        self.pmol, self.slices, self.g_shls, self.h_shls, self.coeff = \
                self._groupby_l(mol, intor, qcondname)
        super().__init__(self.pmol, intor, prescreen, qcondname, dmcondname)

    def _groupby_l(self, mol, intor, qcondname):
        '''Group basis by angular momentum'''
        pmol, coeff = mol.decontract_basis(to_cart=True)

        ls = pmol._bas[:,gto.ANG_OF]
        basis_order = np.argsort(ls, kind='mergesort')
        ao_order = pmol.get_ao_indices(basis_order)
        pmol._bas = np.asarray(pmol._bas[basis_order], dtype=np.int32)
        l_slices, g_shls, h_shls = _group_shells_to_slices(pmol)

        qcond = pmol.condense_to_shell(pmol.intor_symmetric('int1e_ovlp'), 'absmax')
        # qcond index indicates the number of non-zero integrals in the qcond
        # matrix for each shell
        qcond[np.diag_indices_from(qcond)] = 0
        q_index = np.count_nonzero(qcond > 1e-7, axis=1)

        ao_loc = pmol.ao_loc
        lmax = len(l_slices) - 1
        len_cart = (lmax + 1) * (lmax + 2) // 2
        slices = []
        basis_order = []
        for l, (i0, i1) in enumerate(zip(l_slices[:-1], l_slices[1:])):
            if i0 == i1:
                logger.debug1(mol, 'No basis found for l = %d', l)
                continue

            len_cart = (l + 1) * (l + 2) // 2
            shells_limit = int(MAX_BLKSIZE / TILESIZE / len_cart) * TILESIZE
            slices.extend(range(i0, i1, shells_limit))

            # Sort basis according to the compactness of basis. Put compact basis
            # together so that more blocks in gpu kernel can be dropped.
            q_order = q_index[i0:i1].argsort()
            basis_order.append(i0 + q_order)
            p0, p1 = ao_loc[i0], ao_loc[i1]
            ao_order[p0:p1] = ao_order[p0:p1].reshape(-1, len_cart)[q_order].ravel()

        slices.append(l_slices[-1])
        nbas_upto_g = l_slices[-1]
        pmol._bas[:nbas_upto_g] = pmol._bas[np.hstack(basis_order)]

        coeff = scipy.linalg.block_diag(*coeff)
        coeff = coeff[ao_order]
        logger.debug1(mol, 'slices = %s', slices)
        return pmol, slices, g_shls, h_shls, coeff

def _group_shells_to_slices(mol):
    nbas_by_l = np.bincount(mol._bas[:,gto.ANG_OF])
    logger.debug(mol, 'Number of basis for each l %s', nbas_by_l)
    l_slices = np.append(0, np.cumsum(nbas_by_l))

    # find g shells and h_shls for all high angular momentum basis
    g_shls = []
    h_shls = []
    ls = mol._bas[:,gto.ANG_OF]
    lmax = ls.max()
    if lmax >= LMAX_ON_GPU:
        g_shls = l_slices[LMAX_ON_GPU:LMAX_ON_GPU+2].tolist()
    if lmax > LMAX_ON_GPU:
        h_shls = l_slices[LMAX_ON_GPU+1:].tolist()
        l_slices = l_slices[:LMAX_ON_GPU+2].tolist()
        lmax = LMAX_ON_GPU
    return l_slices, g_shls, h_shls
