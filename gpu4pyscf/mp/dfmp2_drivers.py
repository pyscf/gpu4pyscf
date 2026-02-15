import gpu4pyscf.mp.dfmp2_addons as dfmp2_addons

import pyscf
import gpu4pyscf
import numpy as np
import cupy as cp
import cupy
import cupyx


def get_int3c2e_opt(mol, aux, device_list=None, fac=0.2, log=None):
    if log is None:
        log = pyscf.lib.logger.new_logger(mol, verbose=mol.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()

    gpu_mem_list = dfmp2_addons.get_avail_mem_devices(device_list=device_list)
    gpu_mem_avail = min(gpu_mem_list)
    nbytes = 8  # int3c2e is always FP64 in this program
    nao = mol.nao
    nbatch_aux_guess = fac * gpu_mem_avail / (nbytes * nao * nao)
    nbatch_aux = int(max(nbatch_aux_guess, dfmp2_addons.MIN_BATCH_AUX_GPU))
    log.debug(f'in get_int3c2e_opt, nbatch_aux: {nbatch_aux}')

    intopt = gpu4pyscf.df.int3c2e.VHFOpt(mol, aux, 'int2e')
    intopt.build(diag_block_with_triu=True, aosym=True, group_size_aux=nbatch_aux)
    t0 = log.timer('in get_int3c2e_opt, build intopt', *t0)
    return intopt


def dfmp2_kernel_one_gpu(
    mol, aux, occ_coeff, vir_coeff, occ_energy, vir_energy, j3c_backend='bdiv', dtype_cderi=np.float32, t2=None, j2c_decomp_alg='cd', log=None
):
    assert j3c_backend in ['bdiv', 'vhfopt']
    if log is None:
        log = pyscf.lib.logger.new_logger(mol, verbose=mol.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()
    t1 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()

    nocc = occ_energy.shape[0]
    nvir = vir_energy.shape[0]
    naux = aux.nao
    naux_cart = aux.nao_cart()
    naux_alloc = naux_cart if j3c_backend == 'bdiv' else naux

    idx_device = cupy.cuda.get_device_id()

    if j3c_backend == 'bdiv':
        intopt = gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt(mol, aux)
    else:
        intopt = get_int3c2e_opt(mol, aux, device_list=[idx_device], log=log)

    t1 = log.timer('in dfmp2_kernel_one_gpu, build intopt', *t1)

    # j2c
    if j3c_backend == 'bdiv':
        j2c = dfmp2_addons.get_j2c_bdiv(intopt)
    else:
        j2c = dfmp2_addons.get_j2c_vhfopt(intopt)
    j2c_decomp = dfmp2_addons.get_j2c_decomp_gpu(aux, j2c=j2c, alg=j2c_decomp_alg)
    cupy.cuda.get_current_stream().synchronize()
    t1 = log.timer('in dfmp2_kernel_one_gpu, build j2c and decompose', *t1)

    # cderi_ovl_gpu
    get_j3c_ovl = dfmp2_addons.get_j3c_ovl_gpu_bdiv if j3c_backend == 'bdiv' else dfmp2_addons.get_j3c_ovl_gpu_vhfopt
    cderi_ovl_gpu = cp.empty([nocc, nvir, naux_alloc], dtype=dtype_cderi)
    cderi_ovl_gpu = get_j3c_ovl(mol, intopt, [occ_coeff], [vir_coeff], [cderi_ovl_gpu], log=log)[0]
    dfmp2_addons.decompose_j3c_gpu(mol, j2c_decomp, [cderi_ovl_gpu])
    cupy.cuda.get_current_stream().synchronize()
    t1 = log.timer('in dfmp2_kernel_one_gpu, build cderi_ovl', *t1)

    # computation of MP2 correlation energy pair
    e_corr_pair_bi1, e_corr_pair_bi2 = dfmp2_addons.get_dfmp2_energy_pair_intra(mol, cderi_ovl_gpu, occ_energy, vir_energy, t2=t2, log=log)
    t1 = log.timer('in dfmp2_kernel_one_gpu, mp2 occ pair corr energy', *t1)

    # finalize
    e_corr_bi1 = e_corr_pair_bi1.sum()
    e_corr_bi2 = e_corr_pair_bi2.sum()
    e_corr_os = e_corr_bi1
    e_corr_ss = e_corr_bi1 - e_corr_bi2
    result = {
        'e_corr_pair_bi1': e_corr_pair_bi1,
        'e_corr_pair_bi2': e_corr_pair_bi2,
        'e_corr_os': e_corr_os,
        'e_corr_ss': e_corr_ss,
    }

    log.timer('dfmp2_kernel_one_gpu', *t0)
    log.info(f'e_corr_os: {e_corr_os}')
    log.info(f'e_corr_ss: {e_corr_ss}')
    return result


def dfump2_kernel_one_gpu(
    mol, aux, occ_coeff, vir_coeff, occ_energy, vir_energy, j3c_backend='bdiv', dtype_cderi=np.float32, t2=None, j2c_decomp_alg='cd', log=None
):
    assert j3c_backend in ['bdiv', 'vhfopt']
    if log is None:
        log = pyscf.lib.logger.new_logger(mol, verbose=mol.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()
    t1 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()

    spins = [0, 1]
    assert len(occ_coeff) == len(vir_coeff) == len(occ_energy) == len(vir_energy) == 2
    nocc = [occ_energy[s].shape[0] for s in spins]
    nvir = [vir_energy[s].shape[0] for s in spins]
    naux = aux.nao
    naux_cart = aux.nao_cart()
    naux_alloc = naux_cart if j3c_backend == 'bdiv' else naux

    idx_device = cupy.cuda.get_device_id()

    if j3c_backend == 'bdiv':
        intopt = gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt(mol, aux)
    else:
        intopt = get_int3c2e_opt(mol, aux, device_list=[idx_device], log=log)

    t1 = log.timer('in dfmp2_kernel_one_gpu, build intopt', *t1)

    # j2c
    if j3c_backend == 'bdiv':
        j2c = dfmp2_addons.get_j2c_bdiv(intopt)
    else:
        j2c = dfmp2_addons.get_j2c_vhfopt(intopt)
    j2c_decomp = dfmp2_addons.get_j2c_decomp_gpu(aux, j2c=j2c, alg=j2c_decomp_alg)
    cupy.cuda.get_current_stream().synchronize()
    t1 = log.timer('in dfmp2_kernel_one_gpu, build j2c and decompose', *t1)

    # cderi_ovl_gpu
    get_j3c_ovl = dfmp2_addons.get_j3c_ovl_gpu_bdiv if j3c_backend == 'bdiv' else dfmp2_addons.get_j3c_ovl_gpu_vhfopt
    cderi_ovl_gpu = [cp.empty([nocc[s], nvir[s], naux_alloc], dtype=dtype_cderi) for s in spins]
    cderi_ovl_gpu = get_j3c_ovl(mol, intopt, occ_coeff, vir_coeff, cderi_ovl_gpu, log=log)
    dfmp2_addons.decompose_j3c_gpu(mol, j2c_decomp, cderi_ovl_gpu)
    cupy.cuda.get_current_stream().synchronize()
    t1 = log.timer('in dfmp2_kernel_one_gpu, build cderi_ovl', *t1)

    # computation of MP2 correlation energy pair
    e_corr_pair_aa = dfmp2_addons.get_dfmp2_energy_pair_intra(mol, cderi_ovl_gpu[0], occ_energy[0], vir_energy[0], ss_only=True, t2=t2[0], log=log)
    e_corr_pair_bb = dfmp2_addons.get_dfmp2_energy_pair_intra(mol, cderi_ovl_gpu[1], occ_energy[1], vir_energy[1], ss_only=True, t2=t2[2], log=log)
    e_corr_pair_ab = dfmp2_addons.get_dfump2_energy_pair_intra(mol, cderi_ovl_gpu, occ_energy, vir_energy, t2=t2[1], log=log)
    t1 = log.timer('in dfmp2_kernel_one_gpu, mp2 occ pair corr energy', *t1)

    # finalize
    e_corr_aa = 0.25 * e_corr_pair_aa.sum()
    e_corr_bb = 0.25 * e_corr_pair_bb.sum()
    e_corr_ab = e_corr_pair_ab.sum()
    e_corr_os = e_corr_ab
    e_corr_ss = e_corr_aa + e_corr_bb
    result = {
        'e_corr_aa': e_corr_aa,
        'e_corr_bb': e_corr_bb,
        'e_corr_ab': e_corr_ab,
        'e_corr_os': e_corr_os,
        'e_corr_ss': e_corr_ss,
    }

    log.timer('dfump2_kernel_one_gpu', *t0)
    log.info(f'e_corr_os: {e_corr_os}')
    log.info(f'e_corr_ss: {e_corr_ss}')
    return result


def dfmp2_kernel_multi_gpu_cderi_cpu(
    mol, aux, occ_coeff, vir_coeff, occ_energy, vir_energy, ndevice=None, j3c_backend='bdiv', dtype_cderi=np.float32, log=None
):
    # default parameters
    assert j3c_backend in ['bdiv', 'vhfopt']
    if log is None:
        log = pyscf.lib.logger.new_logger(mol, verbose=mol.verbose)
    t0 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()

    if ndevice is None:
        ndevice = cupy.cuda.runtime.getDeviceCount()
    if isinstance(ndevice, list):
        device_list = ndevice
        ndevice = len(device_list)
    else:
        assert isinstance(ndevice, int)
        device_list = list(range(ndevice))

    # basic configurations
    nocc = occ_energy.shape[0]
    nvir = vir_energy.shape[0]
    naux = aux.nao
    naux_cart = aux.nao_cart()
    naux_alloc = naux_cart if j3c_backend == 'bdiv' else naux

    # we assume that each occupied batch should be larger than 8
    if nocc < 8 * ndevice:
        log.warn('Number of occupied orbitals should be larger than 8 on each device.')
        if nocc < 8 * 2:
            log.warn('Occupied orbital number too small. Run MP2 on single GPU.')
            # return dfmp2_kernel_one_gpu(mol, aux, occ_coeff, vir_coeff, occ_energy, vir_energy)
        else:
            ndevice = nocc // 8
            device_list = device_list[:ndevice]
            log.warn(f'Lower device count to {ndevice}.')

    # split occupied orbitals by available GPU memory
    # 0.4 * occ(batch) * nvir * naux * nbytes
    gpu_mem_list = dfmp2_addons.get_avail_mem_devices(device_list)
    gpu_mem_avail = min(gpu_mem_list)
    nbytes = 4 if dtype_cderi == np.float32 else 8
    nocc_batch_max = int(np.floor(gpu_mem_avail * 0.4 / (nvir * naux * nbytes)))
    if nocc_batch_max < 8:
        raise RuntimeError(
            f'GPU memory seems insufficient.\nCurrent  mem (in bytes): {gpu_mem_list}.\nRequired mem (in bytes): {8 * 2 * nvir * naux * nbytes}.'
        )
    nbatch = max(int(np.ceil(nocc / nocc_batch_max / ndevice)), 1)
    nsplit = ndevice * nbatch
    occ_balanced_split = dfmp2_addons.balanced_split(nocc, nsplit)
    assert min(occ_balanced_split) >= 4
    occ_coeff_split = []
    occ_energy_split = []
    idx_occ_0, idx_occ_1 = 0, 0
    for nocc_batch in occ_balanced_split:
        idx_occ_1 = idx_occ_0 + nocc_batch
        occ_coeff_split.append(occ_coeff[:, idx_occ_0:idx_occ_1])
        occ_energy_split.append(occ_energy[idx_occ_0:idx_occ_1])
        idx_occ_0 = idx_occ_1

    # intopt
    t1 = pyscf.lib.logger.process_clock(), pyscf.lib.logger.perf_counter()
    if j3c_backend == 'bdiv':
        intopt = gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt(mol, aux)
    else:
        intopt = get_int3c2e_opt(mol, aux, device_list=device_list, log=log)
    t1 = log.timer('in dfmp2_kernel_multi_gpu_cderi_cpu, build intopt', *t1)

    # j2c
    if j3c_backend == 'bdiv':
        j2c = dfmp2_addons.get_j2c_bdiv(intopt)
    else:
        j2c = dfmp2_addons.get_j2c_vhfopt(intopt)
    j2c_decomp = dfmp2_addons.get_j2c_decomp_gpu(aux, j2c=j2c)
    j2c_decomp_cpu = dict()
    for key, val in j2c_decomp.items():
        if isinstance(val, cp.ndarray):
            j2c_decomp_cpu[key] = j2c_decomp[key].get()
        else:
            j2c_decomp_cpu[key] = j2c_decomp[key]
    j2c = j2c_decomp = None
    cupy.cuda.get_current_stream().synchronize()
    t1 = log.timer('in dfmp2_kernel_multi_gpu_cderi_cpu, build j2c and decompose', *t1)

    # handle cderi_gpu
    cderi_ovl_gpu_list = [None] * nsplit
    cderi_ovl_cpu_list = [None] * nsplit

    def exec_device_cderi(idx_device):
        # intopt of bdiv kernel is device-dependent, need to be regenerated on target device.
        if j3c_backend == 'bdiv':
            intopt_device = gpu4pyscf.df.int3c2e_bdiv.Int3c2eOpt(mol, aux)
        else:
            intopt_device = intopt

        cderi_ovl_gpu = None
        j2c_decomp_device = dict()
        for key, val in j2c_decomp_cpu.items():
            if isinstance(val, np.ndarray):
                j2c_decomp_device[key] = cp.asarray(j2c_decomp_cpu[key])
            else:
                j2c_decomp_device[key] = j2c_decomp_cpu[key]
        occ_coeff_batch_list = []
        vir_coeff_batch_list = []
        cderi_ovl_cpu_batch = []
        cderi_ovl_gpu_batch = None
        for idx_batch in range(nbatch):
            idx_split = idx_device + idx_batch * ndevice
            occ_coeff_device = cp.asarray(occ_coeff_split[idx_split])
            vir_coeff_device = cp.asarray(vir_coeff)
            occ_coeff_batch_list.append(occ_coeff_device)
            vir_coeff_batch_list.append(vir_coeff_device)
            nocc_batch = occ_coeff_device.shape[-1]
            if nbatch == 1:
                # for bdiv kernel, CPU stores only the usual auxbasis (does not use naux_alloc)
                cderi_ovl_cpu = cupyx.empty_pinned([nocc_batch, nvir, naux], dtype=dtype_cderi)
                cderi_ovl_cpu_batch.append(cderi_ovl_cpu)
                cderi_ovl_gpu = cp.empty([nocc_batch, nvir, naux_alloc], dtype=dtype_cderi)
                cderi_ovl_gpu_batch = [cderi_ovl_gpu]
            else:
                # for bdiv kernel, CPU stores the sorted cartesian auxbasis (use naux_alloc)
                cderi_ovl_cpu = cupyx.empty_pinned([nocc_batch, nvir, naux_alloc], dtype=dtype_cderi)
                cderi_ovl_cpu_batch.append(cderi_ovl_cpu)

        # build cderi_ovl
        get_j3c_ovl = dfmp2_addons.get_j3c_ovl_gpu_bdiv if j3c_backend == 'bdiv' else dfmp2_addons.get_j3c_ovl_gpu_vhfopt
        if cderi_ovl_gpu is None:
            cderi_ovl_cpu_batch = get_j3c_ovl(mol, intopt_device, occ_coeff_batch_list, vir_coeff_batch_list, cderi_ovl_cpu_batch, log=log)
            dfmp2_addons.decompose_j3c_gpu(mol, j2c_decomp_device, cderi_ovl_cpu_batch, log=log)
        else:
            cderi_ovl_gpu_batch = get_j3c_ovl(mol, intopt_device, occ_coeff_batch_list, vir_coeff_batch_list, cderi_ovl_gpu_batch, log=log)
            dfmp2_addons.decompose_j3c_gpu(mol, j2c_decomp_device, cderi_ovl_gpu_batch, log=log)
            for cderi_gpu, cderi_cpu in zip(cderi_ovl_gpu_batch, cderi_ovl_cpu_batch):
                cderi_gpu.get(out=cderi_cpu, blocking=False)

        # write back to global list
        for idx_batch in range(nbatch):
            idx_split = idx_device + idx_batch * ndevice
            cderi_ovl_cpu_list[idx_split] = cderi_ovl_cpu_batch[idx_batch]
            if cderi_ovl_gpu is not None:
                cderi_ovl_gpu_list[idx_split] = cderi_ovl_gpu_batch[idx_batch]

    gpu4pyscf.lib.multi_gpu.map(exec_device_cderi, device_list)
    t1 = log.timer('in dfmp2_kernel_multi_gpu_cderi_cpu, build cderi_ovl', *t1)

    result_intra = []
    result_inter = []

    # if cderi not available for all GPU memory, clear GPU tensor list
    if nbatch != 1:
        cderi_ovl_gpu_list = None

    # MP2 computation from cderi on multiple GPUs
    for idx_batch in range(nbatch):

        def exec_device_energy_pair_intra(idx_device):
            idx_split = idx_device + idx_batch * ndevice
            cderi_ovl_to_submit = cderi_ovl_gpu_list[idx_split] if cderi_ovl_gpu_list is not None else cderi_ovl_cpu_list[idx_split]
            return dfmp2_addons.get_dfmp2_energy_pair_intra(
                mol,
                cderi_ovl_to_submit,
                occ_energy_split[idx_split],
                vir_energy,
            )

        def exec_device_energy_pair_inter(idx_device):
            idx_split = idx_device + idx_batch * ndevice
            eval_mode_list = []
            for i in range(nsplit):
                if i == idx_split:
                    eval_mode_list.append(None)
                else:
                    eval_mode_list.append(i < idx_split)
            cderi_ovl_to_submit = cderi_ovl_gpu_list[idx_split] if cderi_ovl_gpu_list is not None else cderi_ovl_cpu_list[idx_split]
            return dfmp2_addons.get_dfmp2_energy_pair_inter(
                mol,
                cderi_ovl_to_submit,
                occ_energy_split[idx_split],
                vir_energy,
                cderi_ovl_cpu_list,
                occ_energy_split,
                eval_mode_list,
            )

        result_intra.extend(gpu4pyscf.lib.multi_gpu.map(exec_device_energy_pair_intra, device_list))
        result_inter.extend(gpu4pyscf.lib.multi_gpu.map(exec_device_energy_pair_inter, device_list))
    t1 = log.timer('in dfmp2_kernel_multi_gpu_cderi_cpu, mp2 occ pair corr energy', *t1)

    # finalize (collect results from separate parts)
    e_corr_pair_bi1 = np.zeros([nocc, nocc])
    e_corr_pair_bi2 = np.zeros([nocc, nocc])
    idx_occ_0, idx_occ_1 = 0, 0
    for idx_split, nocc_batch in enumerate(occ_balanced_split):
        idx_occ_1 = idx_occ_0 + nocc_batch
        slc = slice(idx_occ_0, idx_occ_1)
        e_corr_pair_bi1[slc, slc] = result_intra[idx_split][0]
        e_corr_pair_bi2[slc, slc] = result_intra[idx_split][1]
        e_corr_pair_bi1[slc, :] += result_inter[idx_split][0]
        e_corr_pair_bi1[:, slc] += result_inter[idx_split][0].T
        e_corr_pair_bi2[slc, :] += result_inter[idx_split][1]
        e_corr_pair_bi2[:, slc] += result_inter[idx_split][1].T
        idx_occ_0 = idx_occ_1

    e_corr_bi1 = e_corr_pair_bi1.sum()
    e_corr_bi2 = e_corr_pair_bi2.sum()
    e_corr_os = e_corr_bi1
    e_corr_ss = e_corr_bi1 - e_corr_bi2
    result = {
        'e_corr_pair_bi1': e_corr_pair_bi1,
        'e_corr_pair_bi2': e_corr_pair_bi2,
        'e_corr_os': e_corr_os,
        'e_corr_ss': e_corr_ss,
    }

    log.timer('dfmp2_kernel_multi_gpu_cderi_cpu', *t0)
    log.info(f'e_corr_os: {e_corr_os}')
    log.info(f'e_corr_ss: {e_corr_ss}')
    return result
