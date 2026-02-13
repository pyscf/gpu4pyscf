# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import cupy as cp
import numpy as np
import sys, gc, json
import scipy.linalg


from gpu4pyscf.tdscf import math_helper
from gpu4pyscf.tdscf.math_helper import gpu_mem_info, release_memory, get_avail_gpumem, get_avail_cpumem
from gpu4pyscf.lib.cupy_helper import asarray as cuasarray


from gpu4pyscf.lib import logger
from functools import partial
from pyscf.lib.misc import current_memory
from pyscf.data.nist import HARTREE2EV


RIS_PRECOND_CITATION_INFO = '''
Please cite the TDDFT-ris preconditioning method if you are happy with the fast convergence:

    1.  Zhou, Zehao, and Shane M. Parker.
        Converging Time-Dependent Density Functional Theory Calculations in Five Iterations
        with Minimal Auxiliary Preconditioning. Journal of Chemical Theory and Computation
        20, no. 15 (2024): 6738-6746.

    2.  Zhou, Zehao, Fabio Della Sala, and Shane M. Parker.
        Minimal auxiliary basis set approach for the electronic excitation spectra
        of organic molecules. The Journal of Physical Chemistry Letters
        14, no. 7 (2023): 1968-1976.

    3.  Zhou, Zehao, and Shane M. Parker.
        Accelerating molecular property calculations with
        semiempirical preconditioning. The Journal of Chemical Physics 155, no. 20 (2021).


'''




def _time_add(log, t_total, t_start):
    ''' t_total: list
        t_start: tuple

        In-place revise t_total, add the time elapsed since t_start
    '''
    current_t = log.timer_silent(*t_start)
    for i, val in enumerate(current_t):
        t_total[i] += val

def _time_profiling(log, t_mvp, t_subgen, t_solve_sub, t_sub2full, t_precond, t_fill_holder, t_total):
    '''
    This function prints out the time and percentage of each submodule

    Args:
    t_xxxx: 3-element list, [<cpu time>, <wall time>, <gpu time>]
            each t_xxxx is a timer, the time profiling for each submodule in krylov_solver
            for example, t_mvp is the time profiling for matrix vector product

    example output:

    Timing breakdown:
                            CPU(sec)  wall(sec)    GPU(ms) | Percentage
    mat vec product            3.61       3.67    8035.68    42.9   42.7   93.6
    proj subspace              0.00       0.00       3.88     0.0    0.0    0.0
    solve subspace             0.00       0.00       3.24     0.0    0.0    0.0
    proj fullspace             0.00       0.00       1.56     0.0    0.0    0.0
    precondition               0.37       0.43     428.91     4.4    5.0    5.0
    fill holder                0.01       0.01       9.50     0.1    0.1    0.1
    Sum                        4.00       4.11    8482.78    47.5   47.9   98.8
    Total                      8.42       8.59    8587.89   100.0  100.0  100.0
    '''
    time_labels = ["CPU(sec)", "wall(sec)", "GPU(ms)"]
    labels = time_labels[:len(t_total)]

    log.info("Timing breakdown:")
    header_time = " ".join(f"{label:>10}" for label in labels)
    log.info(f"{'':<20}  {header_time} | Percentage ")

    t_sum = [t_mvp[i] + t_subgen[i] + t_solve_sub[i] + t_sub2full[i] + t_precond[i] + t_fill_holder[i] for i in range(len(t_total))]

    ''' also calculate the time percentage for each timer '''
    timers = {
        'mat vec product':t_mvp,
        'proj subspace':  t_subgen,
        'solve subspace': t_solve_sub,
        'proj fullspace': t_sub2full,
        'precondition':   t_precond,
        'fill holder':    t_fill_holder,
        'Sum':            t_sum,
        'Total':          t_total
    }
    for entry, cost in timers.items():
        time_str = " ".join(f"{x:>10.2f}" for x in cost)
        percent_str = " ".join(f"{(x/y*100 if y != .0 else 100):>6.1f}" for x, y in zip(cost, t_total))
        log.info(f"{entry:<20} {time_str}  {percent_str}")



def eigenvalue_diagonal(**kwargs):
    '''solve
        DX=XΩ
        D is diagonal matrix
    '''
    n_states = kwargs['n_states']
    hdiag = kwargs['hdiag']

    hdiag = hdiag.reshape(-1,)
    A_size = hdiag.shape[0]
    Dsort = hdiag.argsort()[:n_states].reshape(-1,)

    X = cp.zeros((n_states, A_size),dtype=hdiag.dtype)
    for i in range(n_states):
        X[i, Dsort[i]] = hdiag.dtype.type(1.0)
    _converged, _energies = True, None
    # print('X norm', cp.linalg.norm(X, axis=1))
    return _converged, _energies, X

def linear_diagonal(**kwargs):
    ''' solve  DX=rhs,
        where D is a diagonal matrix'''
    hdiag = kwargs['hdiag']
    rhs = kwargs['rhs']

    _converged = True
    return _converged, rhs / hdiag

def shifted_linear_diagonal(**kwargs):
    '''
    solve shifted linear system, where D is a diagonal matrix
    DX - XΩ = rhs
    X = r/(D-Ω)
    Args:
        rhs: 2D array
            right hand side of the linear system
        hdiag: 1D array
            diagonal of the Hamiltonian matrix
        omega: 1D array
            diagonal of the shift matrix
    return X (X is in-place modified rhs)
    '''

    rhs = kwargs['rhs']
    hdiag = kwargs['hdiag']
    omega = kwargs['omega_shift']

    rhs = rhs.astype(dtype=hdiag.dtype, copy=False)
    omega = omega.astype(dtype=hdiag.dtype, copy=False)

    n_states = rhs.shape[0]
    assert n_states == len(omega)
    t = hdiag.dtype.type(1e-14)

    # omega = omega.reshape(-1,1)
    # D = cp.repeat(hdiag.reshape(1,-1), n_states, axis=0) - omega
    # '''
    # force all small values not in [-t,t]
    # '''
    # D = cp.where( abs(D) < t, cp.sign(D)*t, D)
    # X = rhs/D
    # del rhs, D

    X = cp.empty_like(rhs)
    for i in range(n_states):
        Di = hdiag - omega[i]                    # 1D: len(hdiag)

        # Replace |Di| < t with sign(Di)*t  (avoid cp.abs to save memory)
        mask = (Di > -t) & (Di < t)              # Boolean mask for near-zero
        Di = cp.where(mask, cp.sign(Di) * t, Di)  # In-place friendly

        X[i] = rhs[i] / Di                       # Element-wise division
        # rhs[i] /= Di # danger of modifying rhs in-place!!!

        # Optional: clean small intermediates early
        del Di, mask
    # X = rhs
    release_memory()
    _converged = True
    return _converged, X

def shifted_linear_diagonal_inplace(**kwargs):
    '''
    solve shifted linear system, where D is a diagonal matrix
    DX - XΩ = rhs
    X = r/(D-Ω)
    Args:
        rhs: 2D array
            right hand side of the linear system
        hdiag: 1D array
            diagonal of the Hamiltonian matrix
        omega: 1D array
            diagonal of the shift matrix
    return X (X is in-place modified rhs)
    '''

    rhs = kwargs['rhs']
    hdiag = kwargs['hdiag']
    omega = kwargs['omega_shift']

    rhs = rhs.astype(dtype=hdiag.dtype, copy=False)
    omega = omega.astype(dtype=hdiag.dtype, copy=False)

    n_states = rhs.shape[0]
    assert n_states == len(omega)
    t = hdiag.dtype.type(1e-14)

    # X = cp.empty_like(rhs)
    for i in range(n_states):
        Di = hdiag - omega[i]                    # 1D: len(hdiag)

        # Replace |Di| < t with sign(Di)*t  (avoid cp.abs to save memory)
        mask = (Di > -t) & (Di < t)
        force = cp.sign(Di) * t            # Boolean mask for near-zero
        Di = cp.where(mask, force, Di)  # In-place friendly

        # X[i] = rhs[i] / Di                       # Element-wise division
        rhs[i] /= Di # danger of modifying rhs in-place!!!

        # Optional: clean small intermediates early
        del Di, force, mask
        release_memory()
    X = rhs
    release_memory()
    _converged = True
    return _converged, X


'''for each problem type, setup diagonal initial guess and preconitioner '''

'''eigenvalue problem'''
_eigenvalue_diagonal_initguess = eigenvalue_diagonal
# _eigenvalue_diagonal_precond  = shifted_linear_diagonal
_eigenvalue_diagonal_precond  = shifted_linear_diagonal_inplace


'''linear problem'''
_linear_diagonal_initguess = linear_diagonal
_linear_diagonal_precond   = linear_diagonal


'''shifted linear problem'''
_shifted_linear_diagonal_initguess = shifted_linear_diagonal
_shifted_linear_diagonal_precond   = shifted_linear_diagonal

def krylov_solver(matrix_vector_product, hdiag, problem_type='eigenvalue',
                  initguess_fn=None, precond_fn=None, rhs=None,
                  omega_shift=None, n_states=20,conv_tol=1e-5,conv_tol_scaling=0.1,
                  max_iter=35, extra_init=8, gs_initial=False, gram_schmidt=True,
                  restart_subspace=None, single=False, in_ram=False,
                  verbose=logger.NOTE):
    '''
        This solver is used to solve the following problems:
        (1) Eigenvalue problem, return Ω and X
                    AX = XΩ

        (2) Linear system, return X
                    AX = rhs.
            e.g. CPKS problem (A+B)Z = R in TDDFT gradient calculation

        (3) Shifted linear system , return X
                 AX - XΩ = rhs, where Ω is a diagonal matrix.
            e.g. preconditioning,  Green's function

    Theory:
    (1) Eigenvalue problem
           AX = XΩ
        A(Vx) = (Vx)Ω
        V.TAV x = V.TV xΩ
        ax = sxΩ,
        whehre basis overlap s=V.TV, W=AV
        residual r = AX - XΩ = Wx - XΩ

    (2) Linear system
          AX = P
        A(Vx) = P
        V.TAV x = V.TP
        ax = p,
        where p = V.TP (but note that P != Vp)
        residual r = AX - P = Wx - P

    (3) Shifted linear system
        AX - XΩ = P   (P denotes rhs)
        A(Vx) - (Vx) Ω = P
        V.TAV x - V.TV xΩ = V.TP
        ax - sxΩ = p
        residual r = AX - XΩ - P = Wx - XΩ - P

    Args:
        matrix_vector_product: function
            matrix vector product
            e.g. def mvp(X):
                    return A.dot(X)
        hdiag: 1D array
            diagonal of the Hamiltonian matrix
        problem_type: str
            'eigenvalue', 'linear' or 'shifted_linear'
        initguess_fn: function
            function to generate initial guess
        precond_fn: function
            function to apply preconditioner

        -- for eigenvalue problem:
            n_states: int
                number of states to be solved, required, default 20

        -- for linear and shifted_linear problem:
            rhs: 2D array
                right hand side of the linear system, required

        -- for shifted_linear problem:
            omega_shift: 1D array
                diagonal of the shift matrix, required

        conv_tol: float
            convergence tolerance
        conv_tol_scaling: float
            (.0,1.0)
            tolerance set to conv_tol_scaling*conv_tol when selecting which states to precond.
            allow to keep preconditioning the converged states to help trailing sates
            Defaults to 1
            to prevent trailing
        max_iter: int
            maximum iterations
        extra_init: int
            extra number of states to be initialized
        restart_subspace: int or None
            restart the Krylov solver periodically if the subspace size is larger than this value.
            Default None, no restart.
        gs_initial: bool
            apply gram_schmidt procedure on the initial guess,
            only in the case of gram_schmidt = True, but given wired initial guess
        gram_schmidt: bool
            use Gram-Schmidt orthogonalization
        single: bool
            use single precision
        verbose: logger.Logger
            logger object

    Returns:
        converged: the index of converged states/vectors
        omega: 1D array
            eigenvalues
        X: 2D array  (in c-order, each row is a solution vector)
            eigenvectors or solution vectors
    '''

    if problem_type not in ['eigenvalue', 'linear', 'shifted_linear']:
        raise ValueError('Invalid problem type, please choose either eigenvalue, linear or shifted_linear.')

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if single:
        log.info('Using single precision')
        # assert hdiag.dtype == cp.float32
        hdiag = hdiag.astype(cp.float32, copy=False)
    else:
        log.info('Using double precision')
        # assert hdiag.dtype == cp.float64
        hdiag = hdiag.astype(cp.float64, copy=False)



    log.info(f'====== {problem_type.capitalize()} Krylov Solver Starts ======')
    logger.TIMER_LEVEL = 4

    log.info(f'n_states={n_states}, conv_tol={conv_tol}, max_iter={max_iter}, extra_init={extra_init}')

    ''' detailed timing for each sub module
        cpu0 = (cpu time, wall time, gpu time)'''
    cpu0 = log.init_timer()
    t_mvp         = [.0] * len(cpu0)
    t_subgen      = [.0] * len(cpu0)
    t_solve_sub   = [.0] * len(cpu0)
    t_sub2full    = [.0] * len(cpu0)
    t_precond     = [.0] * len(cpu0)
    t_fill_holder = [.0] * len(cpu0)
    t_total       = [.0] * len(cpu0)

    A_size = hdiag.shape[0]
    log.info(f'Size of A matrix = {A_size}')
    if problem_type == 'eigenvalue':
        if extra_init is None:
            extra_init = 8
            size_new = min([n_states + extra_init, 2 * n_states, A_size])
        else:
            size_new = min([n_states + extra_init, A_size])

    elif problem_type in ['linear','shifted_linear']:
        if rhs is None:
            raise ValueError('rhs is required for linear or shifted_linear problem.')

        size_new = rhs.shape[0]
        n_states = rhs.shape[0]

    # record the number of extra initial vectors
    n_extra_init = size_new - n_states

    log.info(f'single trial vector X_ia size: {A_size*hdiag.itemsize/1024**2:.2f} MB')
    if restart_subspace is None:
        ''' calculate the maximum number of vectors allowed by the memory'''
        if in_ram:
            available_mem = get_avail_cpumem()
        else:
            available_mem = get_avail_gpumem()

        restart_subspace = int((available_mem * 0.9) // (2*A_size*hdiag.itemsize))
        log.info(f'the maximum number of vectors allowed by the memory is {restart_subspace}')
    else:
        log.info(f'user specified the maximum number of vectors is {restart_subspace}')

    max_N_mv = min(size_new + max_iter * n_states, restart_subspace)
    log.info(f'the maximum number of vectors in V_holder and W_holder is {max_N_mv}')

    # Initialize arrays
    V_holder_mem = max_N_mv*A_size*hdiag.itemsize/1024**3

    if in_ram:
        rss = current_memory()[0] / 1024 # current memory usage in GB
        log.info(f'the maximum CPU memory usage throughout the Krylov solver is around {2*V_holder_mem + rss:.2f} GB')
    else:
        free_mem, total_mem = cp.cuda.Device().mem_info
        used_mem = (total_mem - free_mem)/1024**3
        log.info(f'the maximum GPU memory usage throughout the Krylov solver is around {2*V_holder_mem + used_mem:.2f} GB')

    xp = np if in_ram else cp
    log.info(f'xp {xp}')
    V_holder = xp.empty((max_N_mv, A_size), dtype=hdiag.dtype)
    W_holder = xp.empty_like(V_holder)

    log.info(f'V_holder {V_holder_mem:.2f} GB')
    log.info(f'W_holder {V_holder_mem:.2f} GB')
    log.info(f'dtype of V_holder & W_holder {V_holder.dtype}')


    sub_A_holder = cp.empty((max_N_mv, max_N_mv), dtype=hdiag.dtype)

    if problem_type in ['linear','shifted_linear']:
        '''Normalize RHS for linear system'''
        rhs_norm = cp.linalg.norm(rhs, axis=1, keepdims=True)
        rhs = rhs/rhs_norm
        sub_rhs_holder = cp.empty((max_N_mv, rhs.shape[0]), dtype=hdiag.dtype)


    # Setup basis projection method
    if gram_schmidt:
        log.info('Using Gram-Schmidt orthogonalization')
        fill_holder = partial(math_helper.Gram_Schmidt_fill_holder, double=True)

    else:
        log.info('Using non-orthogonalized Krylov subspace (nKs) method.')
        nks_citation = '''
        Furche, Filipp, Brandon T. Krull, Brian D. Nguyen, and Jake Kwon.
        Accelerating molecular property calculations with nonorthonormal Krylov space methods.
        The Journal of Chemical Physics 144, no. 17 (2016).
        '''
        log.info(nks_citation)
        fill_holder = math_helper.nKs_fill_holder
        s_holder = cp.empty_like(sub_A_holder)

    if initguess_fn and callable(initguess_fn):
        log.info(' use user-specified function to generate initial guess.')
    else:
        log.info(' use hdiag to generate initial guess.')

        initguess_functions = {
            'eigenvalue':     _eigenvalue_diagonal_initguess,
            'linear':         _linear_diagonal_initguess,
            'shifted_linear': _shifted_linear_diagonal_initguess,
        }
        initguess_fn = initguess_functions[problem_type]


    ''' Generate initial guess '''
    log.info('generating initial guess')
    cpu0 = log.init_timer()

    if problem_type == 'eigenvalue':
        _converged, _energies, init_guess_X = initguess_fn(n_states=size_new, hdiag=hdiag)

    elif problem_type == 'linear':
        _converged, init_guess_X = initguess_fn(hdiag=hdiag, rhs=rhs)

    elif problem_type =='shifted_linear':
        omega_shift = cuasarray(omega_shift, dtype=hdiag.dtype)
        _converged, init_guess_X = initguess_fn(hdiag=hdiag, rhs=rhs, omega_shift=omega_shift)
    log.timer(f' {problem_type.capitalize()} initguess_fn cost', *cpu0)


    cpu0 = log.init_timer()

    # size_old = 0
    # if gs_initial:
    #     '''initial guess were already orthonormalized'''
    #     log.info(' initial guess were already orthonormalized, no need Gram_Schmidt here')
    #     size_new = math_helper.nKs_fill_holder(V_holder, size_old, init_guess_X)
    # else:
    #     log.info(' put initial guess into V_holder')
    #     size_new = fill_holder(V_holder, size_old, init_guess_X)
    log.info(gpu_mem_info('before put initial guess into V_holder'))
    size_old = 0
    if gs_initial:
        '''initial guess were already orthonormalized'''
        log.info(' initial guess were already orthonormalized, no need Gram_Schmidt here')
        # size_new = math_helper.nKs_fill_holder(V_holder, 0, init_guess_X[n_states:, :])# first fill extra_init vectors
        # size_new = math_helper.nKs_fill_holder(V_holder, size_new, init_guess_X[:n_states, :]) # n_states vectors
        extra_init_X = init_guess_X[n_states:, :]
        if in_ram:
            extra_init_X = extra_init_X.get()
        V_holder[:n_extra_init, :] = extra_init_X
        del extra_init_X

        n_states_X = init_guess_X[:n_states, :]
        if in_ram:
            n_states_X = n_states_X.get()
        V_holder[n_extra_init:n_extra_init+n_states, :] = n_states_X
        del n_states_X
        size_new = init_guess_X.shape[0]

    else:
        log.info(' put initial guess into V_holder')
        # size_new = fill_holder(V_holder, size_old, init_guess_X)
        if n_extra_init > 0:
            size_new = fill_holder(V_holder, 0, init_guess_X[n_states:, :])# first fill extra_init vectors
            size_new = fill_holder(V_holder, size_new, init_guess_X[:n_states, :]) # n_states vectors
        else:
            size_new = fill_holder(V_holder, size_old, init_guess_X)
    # print('type(size_new)', type(size_new))
    # print('size_new.shape', size_new.shape)
    # print('size_new',size_new)
    # print('size_old', size_old)

    del init_guess_X
    release_memory()

    log.timer(f' {problem_type.capitalize()} init_guess_X fill_holder cost', *cpu0)
    log.info('initial guess done')
    log.info(gpu_mem_info('after put initial guess into V_holder'))

    if precond_fn and callable(precond_fn):
        log.info(' use user-specified function for preconditioning.')
    else:
        log.info(' use hdiag for preconditioning.')
        precond_functions = {
            'eigenvalue':     _eigenvalue_diagonal_precond,
            'linear':         _linear_diagonal_precond,
            'shifted_linear': _shifted_linear_diagonal_precond,
        }
        precond_fn = precond_functions[problem_type]
        precond_fn = partial(precond_fn, hdiag=hdiag)

    eigenvalue_record = []
    residual_record = []
    n_mvp_record = []
    ''' Davidson iteration starts!
    '''
    for ii in range(max_iter):
        release_memory()
        gc.collect()
        ''' Matrix-vector product '''
        t0 = log.init_timer()
        # log.info( f'V_holder type {type(V_holder)}')
        # log.info( f'W_holder type {type(W_holder)}')
        log.info(gpu_mem_info(f' ▶ ------- iter {ii+1:<3d} MVP starts, {size_new-size_old} vectors'))
        if in_ram:
            X = V_holder[size_old:size_new, :]
        else:
            X = cuasarray(V_holder[size_old:size_new, :])

        log.info(f'     X {X.shape} {X.nbytes//1024**2} MB')
        log.info(f'     V_holder[:size_new, :] memory usage {V_holder[:size_new, :].nbytes/1024**3:.2f} GB')
        log.info(f'     subspace size / maximum subspace size: {size_new} / {max_N_mv}')

        mvp = matrix_vector_product(X)
        del X
        release_memory()
        cp.cuda.Stream.null.synchronize()

        log.info(gpu_mem_info('     after MVP'))

        if in_ram:
            mvp = mvp.get()
            release_memory()
        W_holder[size_old:size_new, :] = mvp
        del mvp
        gc.collect()
        release_memory()

        n_mvp_record.append(size_new - size_old) # number of vectors in MPV, major cost in Krylov solver

        log.info(gpu_mem_info('     MVP stored in W_holder'))

        _time_add(log, t_mvp, t0)
        log.timer('  MVP total cost', *t0)

        ''' Project into Krylov subspace '''
        t0 = log.init_timer()
        sub_A_holder = math_helper.gen_VW(sub_A_holder, V_holder, W_holder, size_old, size_new, symmetry=True)
        log.info(gpu_mem_info('     sub_A_holder updated'))

        sub_A = sub_A_holder[:size_new, :size_new]
        if problem_type in ['linear','shifted_linear']:
            sub_rhs_holder = math_helper.gen_VP(sub_rhs_holder, V_holder, rhs, size_old, size_new)
            sub_rhs = sub_rhs_holder[:size_new, :]

        _time_add(log, t_subgen, t0)

        ''' solve subsapce problem
            solution x is column-wise vectors
            each vetcor contains elements of linear combination coefficient of projection basis
        '''
        t0 = log.init_timer()
        if not gram_schmidt:
            ''' no Gram Schidmit procedure, need the overlap matrix of projection basis'''
            math_helper.gen_VW(s_holder, V_holder, V_holder, size_old, size_new, symmetry=True)
            overlap_s = s_holder[:size_new, :size_new]
            log.info(gpu_mem_info('     overlap_s calculated'))

        if problem_type == 'eigenvalue':
            if gram_schmidt:
                ''' solve ax=xΩ '''
                # sub_A = sub_A.astype(cp.float64)
                omega, x = cp.linalg.eigh(sub_A)
                # omega = omega.astype(hdiag.dtype)
                # x = x.astype(hdiag.dtype)
            else:
                ''' solve ax=sxΩ '''
                try:
                    omega, x = math_helper.solve_AX_SX(sub_A, overlap_s)
                except:
                    # preconditioned solver: d^-1/2 s d^-1/2'''
                    # sub_A = sub_A.astype(cp.float64)
                    # overlap_s = overlap_s.astype(cp.float64)
                    omega_cpu, x_cpu = scipy.linalg.eigh(sub_A.get(), overlap_s.get())
                    omega = cuasarray(omega_cpu)
                    x = cuasarray(x_cpu)
                    # omega, x = cusolver.eigh(sub_A, overlap_s)
                    # omega = omega.astype(hdiag.dtype)
                    # x = x.astype(hdiag.dtype)

            omega = omega[:n_states]
            x = x[:, :n_states]
            log.info(f' Energies (eV): {[round(e,3) for e in (omega*HARTREE2EV).tolist()]}')

        elif problem_type == 'linear':
            x = cp.linalg.solve(sub_A, sub_rhs)

        elif problem_type == 'shifted_linear':
            if gram_schmidt:
                ''' solve ax - xΩ = sub_rhs '''
                x = math_helper.solve_AX_Xla_B(sub_A, omega_shift, sub_rhs)
                # # alternative solver
                # x = scipy.linalg.solve_sylvester(sub_A.get(), -cp.diag(omega_shift).get(), sub_rhs.get())
                # e, u = cp.linalg.eigh(sub_A)
                # print('e ', e)
                # print('omega_shift', omega_shift)
                # for shift in omega_shift:
                #     print(cp.min(abs(e - shift)))
                # x = cuasarray(x)
            else:
                ''' solve ax - s xΩ = sub_rhs
                    => s^-1 ax - xΩ = s^-1 sub_rhs
                TODO need precondition step: s/d first'''
                s_inv = cp.linalg.inv(overlap_s)
                x = scipy.linalg.solve_sylvester(s_inv.dot(sub_A).get(), -cp.diag(omega_shift).get(), s_inv.dot(sub_rhs).get())
                x = cuasarray(x)

        _time_add(log, t_solve_sub, t0)
        log.info(gpu_mem_info('     after solving subspace'))
        t0 = log.init_timer()

        ''' compute the residual
            full_X is current guess solution
            AX is A.dot(full_X)'''
        # AX = cuasarray(W_holder[:size_new, :])
        # AX = cp.dot(x.T, AX)
        # del AX
        # release_memory()


        xT = x.T
        # print('xT.dtype', xT.dtype)
        # AX = AVx = Wx
        # initial data holder, residual := AX
        residual = cp.zeros((n_states,A_size), dtype=hdiag.dtype)
        log.info(gpu_mem_info('     before AX = xTW'))
        residual = math_helper.dot_product_xchunk_V(xT, W_holder[:size_new,:], out=residual)
        log.info(gpu_mem_info('     after AX = xTW'))

        if problem_type == 'eigenvalue':
            ''' r = AX - XΩ
                  = AVx - VxΩ
                  = Wx - VxΩ '''
            # X = cuasarray(V_holder[:size_new, :])
            # full_X = cp.dot(x.T, X)
            residual = math_helper.dot_product_xchunk_V(omega[:,None] * xT, V_holder[:size_new,:], alpha=-1.0, beta=1.0, out=residual)

        elif problem_type == 'linear':
            ''' r = AX - rhs '''
            residual -= rhs

        elif problem_type == 'shifted_linear':
            ''' r = AX - X omega_shift - rhs '''
            # X = cuasarray(V_holder[:size_new, :])
            # full_X = cp.dot(x.T, X)
            # print('omega_shift.dtype', omega_shift.dtype)
            residual = math_helper.dot_product_xchunk_V(omega_shift[:,None] * xT, V_holder[:size_new,:], alpha=-1.0, beta=1.0, out=residual)
            residual -= rhs

        log.info(gpu_mem_info('     residual computed'))
        release_memory()

        _time_add(log, t_sub2full, t0)

        ''' Check convergence '''
        r_norms = cp.linalg.norm(residual, axis=1)

        if problem_type == 'eigenvalue':
             eigenvalue_record.append((omega*HARTREE2EV).tolist())
        residual_record.append(r_norms.tolist())

        if log.verbose >= 5:
            data = {
                "A_size": A_size,
                "n_states": n_states,
                "n_extra_init": n_extra_init,
                "conv_tol": conv_tol,
                "max_iter": max_iter,
                "restart_subspace": restart_subspace,
                "max_N_mv": max_N_mv,
                "in_ram": in_ram,
                "problem_type": problem_type,
                "n_iterations": ii+1,
                "eigenvalue_history": eigenvalue_record if problem_type == 'eigenvalue' else None,
                "residual_norms_history":residual_record,
                "n_mvp_history": n_mvp_record,
            }

            with open('iter_record.json', 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                log.info('iter_record.json saved')

        max_idx = cp.argmax(r_norms)
        log.info('              state :  ||R||2  unconverged')
        for state in range(len(r_norms)):
            if r_norms[state] < conv_tol:
                log.info(f'              {state+1:>5d} {r_norms[state]:.2e}')
            else:
                log.info(f'              {state+1:>5d} {r_norms[state]:.2e} *')

        max_norm = cp.max(r_norms)
        log.info(f'              max|R|: {max_norm:>12.2e}, state {max_idx+1}')

        if max_norm < conv_tol or ii == (max_iter - 1):
            break

        else:

            unconverged_idx = cp.where(r_norms.ravel() > conv_tol_scaling * conv_tol)[0]
            log.info(f'              number of unconverged states: {unconverged_idx.size}')


            if size_new + unconverged_idx.size > max_N_mv:
                log.info(f'     !!! restart subspace (subspace {size_new+unconverged_idx.size} > {max_N_mv})')
                ''' fill N_state solution into the V_holder, but keep the extra initial guess vectors
                    W_holder is also restarted to fully remove the numerical noise
                '''
                del residual
                current_X = math_helper.dot_product_xchunk_V(x.T, V_holder[:size_new,:])
                size_old = n_extra_init
                size_new = fill_holder(V_holder, size_old, current_X)

                del current_X
                release_memory()

            else:
                ''' Preconditioning step '''
                # index_bool = r_norms > conv_tol
                t0 = log.init_timer()
                log.info(gpu_mem_info('     ▸ Preconditioning starts'))

                # residual_unconv = residual[index_bool, :] with boolean indexing creates a copy, which costs extra memory
                # instead, manually move the unconverged residual vectors forehead, use residual[:unconverged_idx.size, :] to save memory

                pos = 0
                for idx in unconverged_idx:
                    if idx != pos:
                        residual[pos,:] = residual[idx,:]
                    pos += 1

                residual_unconv = residual[:unconverged_idx.size, :]

                if problem_type == 'eigenvalue':
                    _converged, X_new = precond_fn(rhs=residual_unconv, omega_shift=omega[unconverged_idx])
                elif problem_type == 'linear':
                    _converged, X_new = precond_fn(rhs=residual_unconv)
                elif problem_type =='shifted_linear':
                    _converged, X_new = precond_fn(rhs=residual_unconv, omega_shift=omega_shift[unconverged_idx])
                log.timer('          preconditioning', *t0)
                del residual_unconv
                release_memory()

                _time_add(log, t_precond, t0)

                ''' put the new guess XY into the holder '''
                t0 = log.init_timer()
                log.info(gpu_mem_info('     ▸ Preconditioning ends'))

                log.info('        putting new guesses into the holder')

                size_old = size_new
                size_new = fill_holder(V_holder, size_old, X_new)
                log.timer('        new guesses put into the holder', *t0)

                del X_new, residual
                release_memory()
                log.info(gpu_mem_info('     ▸ new guesses put into the holder'))

                # if gram_schmidt:
                #     log.info(f'V_holder orthonormality: {math_helper.check_orthonormal(V_holder[:size_new, :])}')
                if size_new == size_old:
                    log.info('All new guesses kicked out during filling holder !!!!!!!')
                    break
                _time_add(log, t_fill_holder, t0)

    if ii == max_iter - 1 and max_norm >= conv_tol:
        log.info(f'=== {problem_type.capitalize()} Krylov Solver not converged below {conv_tol:.2e} due to max iteration limit ! ===')
        log.info(f'Current residual norms: {r_norms.tolist()}')
        log.info(f'max residual norms {cp.max(r_norms)}')

    converged = r_norms <= conv_tol

    log.info(f'Finished in {ii+1} steps')
    log.info(f'Maximum residual norm = {max_norm:.2e}')
    log.info(f'Final subspace size = {sub_A.shape[0]}')

    full_X = math_helper.dot_product_xchunk_V(x.T, V_holder[:size_new,:])

    if problem_type in['linear', 'shifted_linear']:
        full_X *= rhs_norm

    _time_add(log, t_total, cpu0)

    log.timer(f'{problem_type.capitalize()} Krylov Solver total cost', *cpu0)
    _time_profiling(log, t_mvp, t_subgen, t_solve_sub, t_sub2full, t_precond, t_fill_holder, t_total)

    log.info(f'========== {problem_type.capitalize()} Krylov Solver Done ==========')

    del V_holder, W_holder
    release_memory()

    if problem_type == 'eigenvalue':
        return converged, omega, full_X
    elif problem_type in ['linear', 'shifted_linear']:
        return converged, full_X

def nested_krylov_solver(matrix_vector_product, hdiag, problem_type='eigenvalue',
        rhs=None, omega_shift=None, n_states=20, conv_tol=1e-5,
        max_iter=8, gram_schmidt=True, single=False, verbose=logger.INFO,
        init_mvp=None, precond_mvp=None, extra_init=3, extra_init_diag=8,
        init_conv_tol=1e-3, init_max_iter=10,
        precond_conv_tol=1e-2, precond_max_iter=10,
        init_restart_subspace=None, precond_restart_subspace=None, restart_subspace=None,
        in_ram=False):
    '''
    Wrapper for Krylov solver to handle preconditioned eigenvalue, linear, or shifted linear problems.
    requires the non-diagonal approximation of A matrix, i.e., ris approximation.

    Args:
        matrix_vector_product: Callable, computes AX.
        hdiag: 1D cupy array, diagonal of the Hamiltonian matrix.
        problem_type: str, 'eigenvalue', 'linear', 'shifted_linear'.
        rhs: 2D cupy array, right-hand side for linear systems (default: None).
        omega_shift: Diagonal matrix for shifted linear systems (default: None).
        n_states: int, number of eigenvalues or vectors to solve.
        conv_tol: float, convergence tolerance.
        max_iter: int, maximum iterations.
        gram_schmidt: bool, use Gram-Schmidt orthogonalization.
        single: bool, use single precision.
        verbose: logger.Logger or int, logging verbosity.
        init_mvp: Callable, matrix-vector product for initial guess (default: None).
        precond_mvp: Callable, matrix-vector product for preconditioner (default: None).
        init_conv_tol: float, convergence tolerance for initial guess.
        init_max_iter: int, maximum iterations for initial guess.
        precond_conv_tol: float, convergence tolerance for preconditioner.
        precond_max_iter: int, maximum iterations for preconditioner.

    Returns:
        Output of krylov_solver.
    '''

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    dtype = cp.float32 if single else cp.float64
    log.info(f'precision {dtype}')
    if single:
        log.info('Using single precision')
        hdiag = hdiag.astype(cp.float32, copy=False)
    else:
        log.info('Using double precision')
        hdiag = hdiag.astype(cp.float64, copy=False)

    # Validate problem type
    if problem_type not in ['eigenvalue', 'linear', 'shifted_linear']:
        raise ValueError('Invalid problem type, please choose either eigenvalue, linear or shifted_linear.')

    # Define micro_init_precond mapping
    #    the problem_type of
    #    macro problem      intial guess      preconditioner
    micro_init_precond = {
        'eigenvalue':     ['eigenvalue',     'shifted_linear'],
        'linear':         ['linear',         'linear'        ],
        'shifted_linear': ['shifted_linear', 'shifted_linear']
    }

    # Setup initial guess
    if callable(init_mvp):
        log.info('Using iterative initial guess')

        init_problem_type = micro_init_precond[problem_type][0]
        initguess_fn = partial(
            krylov_solver,
            problem_type=init_problem_type, hdiag=hdiag,
            matrix_vector_product=init_mvp,
            conv_tol=init_conv_tol, max_iter=init_max_iter,
            restart_subspace=init_restart_subspace,
            gram_schmidt=gram_schmidt, single=single, verbose=log.verbose-2
        )
    else:
        log.info('Using diagonal initial guess')
        initguess_fn = None

    # Setup preconditioner
    if callable(precond_mvp):
        log.info('Using iterative preconditioner')

        precond_problem_type = micro_init_precond[problem_type][1]
        precond_fn = partial(
            krylov_solver,
            problem_type=precond_problem_type, hdiag=hdiag,
            matrix_vector_product=precond_mvp,
            conv_tol=precond_conv_tol, max_iter=precond_max_iter,
            restart_subspace=precond_restart_subspace,
            gram_schmidt=gram_schmidt, single=single, verbose=log.verbose-1
        )
    else:
        log.info('Using diagonal preconditioner')
        precond_fn = None

    if not init_mvp and not precond_mvp:
        log.warn(f'diagonal initial guess and preconditioner provided, using extra_init={extra_init_diag}')
        extra_init = extra_init_diag

    # Run solver
    output = krylov_solver(
        matrix_vector_product=matrix_vector_product, hdiag=hdiag,
        problem_type=problem_type, n_states=n_states,
        rhs=rhs, omega_shift=omega_shift, extra_init=extra_init,
        initguess_fn=initguess_fn, precond_fn=precond_fn,
        conv_tol=conv_tol, max_iter=max_iter,
        gram_schmidt=gram_schmidt, single=single, verbose=verbose,
        restart_subspace=restart_subspace, in_ram=in_ram
    )
    log.info(RIS_PRECOND_CITATION_INFO)
    return output

'''above is for TDA;
following is for TDDFT'''

def ABBA_eigenvalue_diagonal(**kwargs):
    '''solve
        [ D 0 ] X = [ 1  0 ] X Ω
        [ 0 D ] Y   [ 0 -1 ] Y
        D is diagonal matrix
        DX =  X Ω => D = Ω
        DY = -Y Ω => 2DY = 0 => Y=0
    '''
    _converged, _energies, X = eigenvalue_diagonal(**kwargs)
    Y = cp.zeros_like(X)
    _converged, _energies = True, None
    return _converged, _energies, X, Y

def ABBA_shifted_linear_diagonal_backup(**kwargs):
    '''solve
        [ D 0 ] X - [ 1  0 ] X Ω = [rhs_1]
        [ 0 D ] Y   [ 0 -1 ] Y     [rhs_2]
        D is diagonal matrix, Ω is gieven
        DX - X Ω = rhs_1
        DY + Y Ω = rhs_2
    '''
    rhs_1 = kwargs['rhs_1']
    rhs_2 = kwargs['rhs_2']
    hdiag = kwargs['hdiag']
    omega = kwargs['omega_shift']

    n_states = rhs_1.shape[0]
    t = 1e-8
    omega = omega.reshape(-1,1)

    d = cp.repeat(hdiag.reshape(1,-1), n_states, axis=0)

    D_x = d - omega
    D_x = cp.where(abs(D_x) < t, cp.sign(D_x)*t, D_x)

    D_y = d + omega
    D_y = cp.where(abs(D_y) < t, cp.sign(D_y)*t, D_y)

    X_new = rhs_1/D_x
    Y_new = rhs_2/D_y

    _converged = True
    return _converged, X_new, Y_new

def ABBA_shifted_linear_diagonal(**kwargs):
    """
    Solve shifted linear systems with very low extra memory:
        D X - X Ω = rhs_1
        D Y + Y Ω = rhs_2

    Where D is diagonal (given by hdiag), Ω is diagonal (given by omega_shift)

    This version computes row-by-row to avoid creating full broadcasted matrices.
    Extra memory ~ O(A_size) instead of O(n_states × A_size)
    """
    rhs_1 = kwargs['rhs_1']       # shape: (n_states, A_size)
    rhs_2 = kwargs['rhs_2']
    hdiag = kwargs['hdiag']       # shape: (A_size,)
    omega = kwargs['omega_shift'] # shape: (n_states,)

    n_states, A_size = rhs_1.shape
    assert rhs_2.shape == (n_states, A_size)
    assert len(hdiag) == A_size
    assert len(omega) == n_states

    dtype = hdiag.dtype
    t = dtype.type(1e-12)   # threshold, can be 1e-14 if you prefer stricter

    # Pre-allocate output arrays (can reuse rhs_1/rhs_2 if you allow in-place)
    X = cp.empty_like(rhs_1)
    Y = cp.empty_like(rhs_2)

    # Reuse a single temporary row vector
    Di = cp.empty(A_size, dtype=dtype)

    for i in range(n_states):
        # 1. Compute D - omega_i  for X
        cp.subtract(hdiag, omega[i], out=Di)           # Di = hdiag - omega[i]

        # Avoid division by near-zero
        mask = (Di > -t) & (Di < t)
        Di = cp.where(mask, cp.sign(Di) * t, Di)    # in-place where if possible

        # X[i] = rhs_1[i] / Di
        cp.divide(rhs_1[i], Di, out=X[i])

        # 2. Compute D + omega_i  for Y
        cp.add(hdiag, omega[i], out=Di)                # reuse Di: Di = hdiag + omega[i]

        # Same safeguard
        mask = (Di > -t) & (Di < t)
        Di = cp.where(mask, cp.sign(Di) * t, Di)

        # Y[i] = rhs_2[i] / Di
        cp.divide(rhs_2[i], Di, out=Y[i])

    # Optional: if memory is extremely tight, can del mask here
    del Di, mask
    release_memory()
    _converged = True
    return _converged, X, Y


'''eigenvalue problem'''
_ABBA_eigenvalue_diagonal_initguess = ABBA_eigenvalue_diagonal
_ABBA_eigenvalue_diagonal_precond  = ABBA_shifted_linear_diagonal



'''shifted linear problem'''
_ABBA_shifted_linear_diagonal_initguess = ABBA_shifted_linear_diagonal
_ABBA_shifted_linear_diagonal_precond   = ABBA_shifted_linear_diagonal

def ABBA_krylov_solver(matrix_vector_product, hdiag, problem_type='eigenvalue',
                  initguess_fn=None, precond_fn=None, rhs_1=None, rhs_2=None,
                  omega_shift=None, n_states=20,conv_tol=1e-5, conv_tol_scaling=0.1,
                  max_iter=35, extra_init=8, gram_schmidt=True,
                  restart_subspace=None, in_ram=False, gs_initial=False,
                  single=False, verbose=logger.NOTE):
    '''
        This solver is used to solve the following problems:

        (1) eigenvalue problem, return Ω and [X,Y]
            [ A B ] X = [ 1  0 ] X Ω
            [ B A ] Y   [ 0 -1 ] Y
            e.g. Casida equation

        (2) shifted linear system , return X
            [ A B ] X - [ 1  0 ] Y Ω = [rhs_1]
            [ B A ] Y   [ 0 -1 ] X     [rhs_2]
            where Ω is a diagonal matrix.
            e.g. dynamic polarizability

        Note:
        in the case of linear equation,
            [ A B ] X  = [rhs_1]
            [ B A ] Y  = [rhs_2],
        =>  (A+B)(X+Y) = rhs_1 + rhs_2
        fallback to normal krylov solver above

    Theory:

    (1) Eigenvalue problem:

        [ A B ] X = [ 1  0 ] X Ω
        [ B A ] Y   [ 0 -1 ] Y

        use a linear combination of projection basis V,W to expand X,Y
        [X] = [ V W ] [x]
        [Y] = [ W V ] [y]

        so that

        [ V.T W.T ] [ A B ] [ V W ] [x] = [ V.T W.T ] [ 1  0 ] [ V W ] [x] Ω
        [ W.T V.T ] [ B A ] [ W V ] [y]   [ W.T V.T ] [ 0 -1 ] [ W V ] [y]

        [ a b ] x = [  σ  π ] X Ω
        [ b a ] y   [ -π -σ ] Y

        where
            a = [V.T W.T][A B][V] = [V.T W.T][U1] = VU1 + WU2
                         [B A][W]            [U2]

                where
                    [U1] = [A B][V] = [ AV + BW ]
                    [U2]   [B A][W]   [ AW + BV ]

            similarly,
            b = [W.T V.T][A B][W] = [W.T V.T][U1] = WU1 + VU2
                         [B A][V]            [U2]

            the projection basis overlap matrix is

            σ = [V.T W.T][ V] = V.TV - W.TW
                         [-W]

            π = [V.T W.T][ W] = V.TW - W.TV
                         [-V]

        note:
            σ.T = σ,   σ != 1
            π.T = -π,  π != 0

        residual:
        r_1 = U1x + U2y - X_full*omega
        r_2 = U2x + U1y + Y_full*omega
        X_full = Vx + Wy
        Y_full = Wx + Vy


    (2) Shifted linear system:
            [ A B ] X - [ 1  0 ] Y Ω = [P]
            [ B A ] Y   [ 0 -1 ] X     [Q]
        P, Q denotes rhs_1, rhs_2

        [ a b ] x - [  σ  π ] X Ω  = [p]
        [ b a ] y   [ -π -σ ] Y    = [q]

        [p] = [ V.T W.T ][P]
        [q]   [ W.T V.T ][Q]

    Args:
        matrix_vector_product: function
            matrix vector product
            e.g.
            def matrix_vector_product(X, Y):
                U1 = X.dot(A) + Y.dot(B)
                U2 = Y.dot(A) + X.dot(B)
                return U1, U2

        hdiag: 1D array
            diagonal of the A matrix
        problem_type: str
            'eigenvalue' or 'shifted_linear'
        initguess_fn: function
            function to generate initial guess
        precond_fn: function
            function to apply preconditioner

        -- for eigenvalue problem:
            n_states: int
                number of states to be solved, required, default 20

        -- for shifted_linear problem:
            omega_shift: 1D array
                diagonal of the shift matrix, required

        conv_tol: float
            convergence tolerance
        max_iter: int
            maximum iterations
        extra_init: int
            extra number of states to be initialized
        gram_schmidt: bool
            use Gram-Schmidt orthogonalization
        single: bool
            use single precision
        verbose: logger.Logger
            logger object

    Returns:
        omega: 1D array
             eigenvalues
        X_full, Y_full: 2D array  (in c-order, each row is a solution vector)
            eigenvectors or solution vectors

    '''
    if problem_type not in ['eigenvalue',  'shifted_linear']:
        raise ValueError('Invalid problem type, please choose either eigenvalue or shifted_linear.')


    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if single:
        log.info('Using single precision')
        # assert hdiag.dtype == cp.float32
        hdiag = hdiag.astype(cp.float32, copy=False)
    else:
        log.info('Using double precision')
        # assert hdiag.dtype == cp.float64
        hdiag = hdiag.astype(cp.float64, copy=False)



    log.info(f'====== {problem_type.capitalize()} ABBA Krylov Solver Starts ======')
    logger.TIMER_LEVEL = 4

    ''' detailed timing for each sub module
        cpu0 = (cpu time, wall time, gpu time)'''
    cpu0 = log.init_timer()
    t_mvp         = [.0] * len(cpu0)
    t_subgen      = [.0] * len(cpu0)
    t_solve_sub   = [.0] * len(cpu0)
    t_sub2full    = [.0] * len(cpu0)
    t_precond     = [.0] * len(cpu0)
    t_fill_holder = [.0] * len(cpu0)
    t_total       = [.0] * len(cpu0)

    A_size = hdiag.shape[0]
    log.info(f'size of A matrix = {A_size}')

    size_old = 0
    if problem_type == 'eigenvalue':
        if extra_init is None:
            extra_init = 8
            size_new = min([n_states + extra_init, 2 * n_states, A_size])
        else:
            size_new = min([n_states + extra_init, A_size])

    elif problem_type == 'shifted_linear':
        if rhs_1 is None or rhs_2 is None:
            raise ValueError('rhs_1 and rhs_2 is required for shifted_linear problem.')

        size_new = rhs_1.shape[0]
        n_states = rhs_1.shape[0]

    # record the number of extra initial vectors
    n_extra_init = size_new - n_states

    log.info(f'single trial vector X_ia and Y_ia size: 2 *{A_size*hdiag.itemsize/1024**2:.2f} MB')

    if restart_subspace is None:
        ''' calculate the maximum number of vectors allowed by the memory'''
        if in_ram:
            available_mem = get_avail_cpumem()
        else:
            available_mem = get_avail_gpumem()

        restart_subspace = int((available_mem * 0.9) // (4*A_size*hdiag.itemsize))
        log.info(f'the maximum number of vectors allowed by the memory is {restart_subspace}')
    else:
        log.info(f'user specified the maximum number of vectors is {restart_subspace}')

    max_N_mv = min(size_new + max_iter * n_states, restart_subspace)
    log.info(f'size_new + max_iter * n_states = {size_new + max_iter * n_states}')
    log.info(f'the maximum number of vectors in V W U1 U2holder is {max_N_mv}')

    holder_mem = 4 * max_N_mv * A_size * hdiag.itemsize/(1024**2)
    log.info(f'  V W U1 U2 holder use {holder_mem:.2f} MB memory')

    xp = np if in_ram else cp
    log.info(f'xp {xp}')


    V_holder = xp.zeros((max_N_mv, A_size), dtype=hdiag.dtype)
    W_holder = xp.zeros_like(V_holder)

    U1_holder = xp.empty_like(V_holder)
    U2_holder = xp.empty_like(V_holder)

    VU1_holder = cp.empty((max_N_mv,max_N_mv), dtype=hdiag.dtype)
    VU2_holder = cp.empty_like(VU1_holder)
    WU1_holder = cp.empty_like(VU1_holder)
    WU2_holder = cp.empty_like(VU1_holder)

    VV_holder = cp.empty_like(VU1_holder)
    VW_holder = cp.empty_like(VU1_holder)
    WW_holder = cp.empty_like(VU1_holder)

    '''
    set up initial guess, V= TDA initial guess, W=0
    '''

    if problem_type == 'shifted_linear':
        rhs = cp.hstack((rhs_1,rhs_2))
        rhs_norm = cp.linalg.norm(rhs, axis=1, keepdims = True)
        rhs_1 = rhs_1/rhs_norm
        rhs_2 = rhs_2/rhs_norm
        VP_holder = cp.empty((max_N_mv,rhs.shape[0]), dtype=hdiag.dtype)
        VQ_holder = cp.empty_like(VP_holder)
        WP_holder = cp.empty_like(VP_holder)
        WQ_holder = cp.empty_like(VP_holder)


    if gram_schmidt:
        log.info('Using Gram-Schmidt orthogonalization')
        fill_holder = partial(math_helper.VW_Gram_Schmidt_fill_holder, double=True)

    else:
        log.info('Using non-orthogonalized Krylov subspace (nKs) method.')
        nks_citation = '''
        Furche, Filipp, Brandon T. Krull, Brian D. Nguyen, and Jake Kwon.
        Accelerating molecular property calculations with nonorthonormal Krylov space methods.
        The Journal of Chemical Physics 144, no. 17 (2016).
        '''
        log.info(nks_citation)
        fill_holder = math_helper.VW_nKs_fill_holder
        '''Unlike the standard Krylov solver for symmatric matrix,
           in the case of ABBA, the overalp matrix σ π is always needed (non-identity),
           no matter whether use Gram_Schmidt or nKs '''

    if initguess_fn and callable(initguess_fn):
        log.info(' use user-specified function to generate initial guess.')
    else:
        log.info(' use hdiag to generate initial guess.')

        initguess_functions = {
            'eigenvalue':     _ABBA_eigenvalue_diagonal_initguess,
            'shifted_linear': _ABBA_shifted_linear_diagonal_initguess,
        }
        initguess_fn = initguess_functions[problem_type]


    ''' Generate initial guess '''
    log.info('generating initial guess')
    cpu0 = log.init_timer()

    if problem_type == 'eigenvalue':
        _converged, _energies, init_guess_X, init_guess_Y = initguess_fn(n_states=size_new, hdiag=hdiag)

    elif problem_type =='shifted_linear':
        _converged, init_guess_X, init_guess_Y = initguess_fn(hdiag=hdiag, rhs_1=rhs_1, rhs_2=rhs_2, omega_shift=omega_shift)
    log.timer(f' {problem_type.capitalize()} initguess_fn cost', *cpu0)

    cpu0 = log.init_timer()
    log.info(gpu_mem_info('before put initial guess into V_holder and W_holder'))


    if gs_initial:
        extra_init_X = init_guess_X[n_states:, :]
        extra_init_Y = init_guess_Y[n_states:, :]
        if in_ram:
            extra_init_X = extra_init_X.get()
            extra_init_Y = extra_init_Y.get()
        V_holder[:n_extra_init, :] = extra_init_X
        W_holder[:n_extra_init, :] = extra_init_Y

        del extra_init_X
        del extra_init_Y


        n_states_X = init_guess_X[:n_states, :]
        n_states_Y = init_guess_Y[:n_states, :]
        if in_ram:
            n_states_X = n_states_X.get()
            n_states_Y = n_states_Y.get()

        V_holder[n_extra_init:n_extra_init+n_states, :] = n_states_X
        W_holder[n_extra_init:n_extra_init+n_states, :] = n_states_Y
        del n_states_X, n_states_Y
        size_new = init_guess_X.shape[0]

    else:
        V_holder, W_holder, size_new = fill_holder(V_holder=V_holder,
                                                W_holder=W_holder,
                                                count=size_old,
                                                X_new=init_guess_X,
                                                Y_new=init_guess_Y)
    del init_guess_X, init_guess_Y
    release_memory()

    log.timer(f' {problem_type.capitalize()} init_guess_X fill_holder cost', *cpu0)
    log.info('initial guess done')
    log.info(gpu_mem_info('after put initial guess into V_holder'))


    if precond_fn and callable(precond_fn):
        log.info(' use user-specified function for preconditioning.')
    else:
        log.info(' use hdiag for preconditioning.')
        precond_functions = {
            'eigenvalue':     _ABBA_eigenvalue_diagonal_precond,
            'shifted_linear': _ABBA_shifted_linear_diagonal_precond,
        }
        precond_fn = precond_functions[problem_type]
        precond_fn = partial(precond_fn, hdiag=hdiag)

    eigenvalue_record = []
    residual_record = []
    n_mvp_record = []

    for ii in range(max_iter):
        log.info(gpu_mem_info(f' ▶ ------- iter {ii+1:<3d} MVP starts, {size_new-size_old} vectors'))

        ''' Matrix-vector product '''
        t0 = log.init_timer()
        X = V_holder[size_old:size_new, :]
        Y = W_holder[size_old:size_new, :]
        if in_ram:
            X = cuasarray(X)
            Y = cuasarray(Y)
            release_memory()

        log.info(f'     X {X.shape} {X.nbytes//1024**2} MB')
        log.info(f'     Y {Y.shape} {Y.nbytes//1024**2} MB')

        log.info(f'     V_holder[:size_new, :] memory usage {V_holder[:size_new, :].nbytes/1024**3:.2f} GB')
        log.info(f'     W_holder[:size_new, :] memory usage {W_holder[:size_new, :].nbytes/1024**3:.2f} GB')

        log.info(f'     subspace size / maximum subspace size: {size_new} / {max_N_mv}')

        U1_mvp, U2_mvp = matrix_vector_product(X=X,Y=Y)
        del X, Y
        release_memory()
        cp.cuda.Stream.null.synchronize()

        log.info(gpu_mem_info('     after MVP'))

        if in_ram:
            U1_mvp = U1_mvp.get()
            U2_mvp = U2_mvp.get()
            release_memory()
        U1_holder[size_old:size_new, :] = U1_mvp
        U2_holder[size_old:size_new, :] = U2_mvp
        del U1_mvp, U2_mvp
        gc.collect()
        release_memory()

        n_mvp_record.append(size_new - size_old)
        log.info(gpu_mem_info('     MVP stored in U1_holder and U2_holder'))

        _time_add(log, t_mvp, t0)
        log.timer('  MVP total cost', *t0)

        ''' Project into Krylov subspace '''
        t0 = log.init_timer()
        (sub_A, sub_B, sigma, pi,
        VU1_holder, WU2_holder, VU2_holder, WU1_holder,
        VV_holder, WW_holder, VW_holder) = math_helper.gen_sub_ab(V_holder, W_holder, U1_holder, U2_holder,
                                                            VU1_holder, WU2_holder, VU2_holder, WU1_holder,
                                                            VV_holder, WW_holder, VW_holder,
                                                            size_old, size_new)
        if problem_type == 'shifted_linear':
            sub_rhs_1, sub_rhs_2, VP_holder, WQ_holder, WP_holder, VQ_holder = math_helper.gen_sub_pq(
                                                                V_holder, W_holder, rhs_1, rhs_2,
                                                                VP_holder, WQ_holder, WP_holder, VQ_holder,
                                                                size_old, size_new)

        _time_add(log, t_subgen, t0)
        log.timer('  subgen cost', *t0)
        log.info(gpu_mem_info('     after subgen'))
        # norm1, norm2 = math_helper.check_VW_orthogonality(V_holder[:size_new, :], W_holder[:size_new, :])
        # log.info(f'     VVWW norm: {norm1:.2e}, VWTW norm: {norm2:.2e}')

        ''' solve subsapce problem
            solution x,y are column-wise vectors
            each vetcor contains elements of linear combination coefficient of projection basis
        '''
        t0 = log.init_timer()
        if problem_type == 'eigenvalue':
            omega, x, y = math_helper.TDDFT_subspace_eigen_solver2(sub_A, sub_B, sigma, pi, n_states)

            '''solver backup'''
            # solvers = [
            #     math_helper.TDDFT_subspace_eigen_solver2,
            #     math_helper.TDDFT_subspace_eigen_solver,
            #     math_helper.TDDFT_subspace_eigen_solver3,
            # ]

            # omega, x, y = None, None, None

            # for solver in solvers:
            #     try:
            #         omega, x, y = solver(sub_A, sub_B, sigma, pi, n_states)
            #         log.info(f"Solver {solver.__name__} success")
            #         break
            #     except Exception as e:
            #         log.info(f"Solver {solver.__name__} failed: {e}")
            #         continue
            # else:
            #     raise RuntimeError("All available TDDFT subspace eigenvalue solvers failed")

        elif problem_type == 'shifted_linear':
            x,y = math_helper.TDDFT_subspace_linear_solver(sub_A, sub_B, sigma, pi, sub_rhs_1, sub_rhs_2, omega_shift)

        _time_add(log, t_solve_sub, t0)
        log.info(f' Energies (eV): {[round(e,3) for e in (omega*HARTREE2EV).tolist()]}')

        '''
        compute the residual
        X_full, Y_full is current guess solution

        (1) Eigenvalue system:
        r_1 = U1x + U2y - X_full*omega
        r_2 = U2x + U1y + Y_full*omega
        X_full = Vx + Wy
        Y_full = Wx + Vy

        (2) Shifted linear system:
        r_1 = U1x + U2y - X_full*omega_shift - rhs_1
        r_2 = U2x + U1y + Y_full*omega_shift - rhs_2
        X_full = Vx + Wy
        Y_full = Wx + Vy

        '''
        t0 = log.init_timer()

        # V = V_holder[:size_new,:]
        # W = W_holder[:size_new,:]
        # U1 = U1_holder[:size_new, :]
        # U2 = U2_holder[:size_new, :]

        # X_full = cp.dot(x.T, V) + cp.dot(y.T, W)
        # Y_full = cp.dot(x.T, W) + cp.dot(y.T, V)

        # if problem_type == 'eigenvalue':
        #     residual_1 = cp.dot(x.T, U1) + cp.dot(y.T, U2) - omega.reshape(-1, 1) * X_full
        #     residual_2 = cp.dot(x.T, U2) + cp.dot(y.T, U1) + omega.reshape(-1, 1) * Y_full

        # elif problem_type == 'shifted_linear':
        #     residual_1 = cp.dot(x.T, U1) + cp.dot(y.T, U2) - omega_shift.reshape(-1, 1) * X_full - rhs_1
        #     residual_2 = cp.dot(x.T, U2) + cp.dot(y.T, U1) + omega_shift.reshape(-1, 1) * Y_full - rhs_2

        xT = x.T
        yT = y.T

        residual_1 = cp.zeros((n_states,A_size), dtype=hdiag.dtype)
        residual_2 = cp.zeros((n_states,A_size), dtype=hdiag.dtype)


        residual_1 = math_helper.dot_product_xchunk_V(xT, U1_holder[:size_new,:], out=residual_1)
        residual_1 = math_helper.dot_product_xchunk_V(yT, U2_holder[:size_new,:], out=residual_1)

        residual_2 = math_helper.dot_product_xchunk_V(xT, U2_holder[:size_new,:], out=residual_2)
        residual_2 = math_helper.dot_product_xchunk_V(yT, U1_holder[:size_new,:], out=residual_2)


        current_V = V_holder[:size_new,:]
        current_W = W_holder[:size_new,:]

        if problem_type == 'eigenvalue':
            omega_xT = omega[:,None] * xT
            omega_yT = omega[:,None] * yT

            residual_1 = math_helper.dot_product_xchunk_V(omega_xT, current_V, alpha=-1.0, beta=1.0, out=residual_1)
            residual_1 = math_helper.dot_product_xchunk_V(omega_yT, current_W, alpha=-1.0, beta=1.0, out=residual_1)

            residual_2 = math_helper.dot_product_xchunk_V(omega_xT, current_W, alpha=1.0, beta=1.0, out=residual_2)
            residual_2 = math_helper.dot_product_xchunk_V(omega_yT, current_V, alpha=1.0, beta=1.0, out=residual_2)

        elif problem_type == 'shifted_linear':
            omega_shift_xT = omega_shift[:,None] * xT
            omega_shift_yT = omega_shift[:,None] * yT

            residual_1 = math_helper.dot_product_xchunk_V(omega_shift_xT, current_V, alpha=-1.0, beta=1.0, out=residual_1)
            residual_1 = math_helper.dot_product_xchunk_V(omega_shift_yT, current_W, alpha=-1.0, beta=1.0, out=residual_1)

            residual_2 = math_helper.dot_product_xchunk_V(omega_shift_xT, current_W, alpha=1.0, beta=1.0, out=residual_2)
            residual_2 = math_helper.dot_product_xchunk_V(omega_shift_yT, current_V, alpha=1.0, beta=1.0, out=residual_2)

        _time_add(log, t_sub2full, t0)
        log.timer('  sub2full cost', *t0)
        log.info(gpu_mem_info('     after sub2full compute residual'))

        ''' Check convergence '''
        residual = cp.hstack((residual_1, residual_2))

        r_norms = cp.linalg.norm(residual, axis=1)

        del residual
        release_memory()

        max_norm = cp.max(r_norms)

        eigenvalue_record.append((omega*HARTREE2EV).tolist())
        residual_record.append(r_norms.tolist())

        if log.verbose >= 5:
            data = {
                "A_size": A_size,
                "n_states": n_states,
                "n_extra_init": n_extra_init,
                "conv_tol": conv_tol,
                "max_iter": max_iter,
                "restart_subspace": restart_subspace,
                "max_N_mv": max_N_mv,
                "in_ram": in_ram,
                "problem_type": problem_type,
                "n_iterations": len(eigenvalue_record),
                "eigenvalue_history": eigenvalue_record,
                "residual_norms_history":residual_record,
                "n_mvp_history": n_mvp_record,
            }

            with open('iter_record.json', 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                log.info('iter_record.json saved')

        max_idx = cp.argmax(r_norms)
        log.info('              state :  ||R||2  unconverged')
        for state in range(len(r_norms)):
            if r_norms[state] < conv_tol:
                log.info(f'              {state+1:>5d} {r_norms[state]:.2e}')
            else:
                log.info(f'              {state+1:>5d} {r_norms[state]:.2e} *')

        max_norm = cp.max(r_norms)
        log.info(f'              max|R|: {max_norm:>12.2e}, state {max_idx+1}')

        if max_norm < conv_tol or ii == (max_iter -1):
            break

        else:

            unconverged_idx = np.where(r_norms.ravel() > conv_tol_scaling * conv_tol)[0]
            log.info(f'              number of unconverged states: {unconverged_idx.size}')


            if size_new + unconverged_idx.size > max_N_mv:
                log.info(f'     !!! restart subspace (subspace {size_new+unconverged_idx.size} > {max_N_mv})')
                ''' fill N_state solution into the V_holder, but keep the extra initial guess vectors
                    W_holder is also restarted to fully remove the numerical noise
                '''

                X_full = math_helper.dot_product_xchunk_V(xT, current_V) + math_helper.dot_product_xchunk_V(yT, current_W)
                Y_full = math_helper.dot_product_xchunk_V(xT, current_W) + math_helper.dot_product_xchunk_V(yT, current_V)

                size_old = n_extra_init
                V_holder, W_holder, size_new = fill_holder(V_holder=V_holder,
                                                            W_holder=W_holder,
                                                            X_new=X_full,
                                                            Y_new=Y_full,
                                                            count=size_old)
                del X_full, Y_full
                release_memory()

            else:
                '''  preconditioning step '''
                # index_bool = r_norms > conv_tol
                # residual_1 = residual_1[index_bool,:]
                # residual_2 = residual_2[index_bool,:]

                pos = 0
                for idx in unconverged_idx:
                    if idx != pos:
                        residual_1[pos,:] = residual_1[idx,:]
                        residual_2[pos,:] = residual_2[idx,:]

                    pos += 1

                residual_1_unconv = residual_1[:unconverged_idx.size,:]
                residual_2_unconv = residual_2[:unconverged_idx.size,:]

                t0 = log.init_timer()
                log.info(gpu_mem_info('     before preconditioning'))
                log.info('     Preconditioning starts')
                if problem_type == 'eigenvalue':
                    _converged, X_new, Y_new = precond_fn(rhs_1=residual_1_unconv, rhs_2=residual_2_unconv,
                                                        omega_shift=omega[unconverged_idx])

                elif problem_type =='shifted_linear':
                    _converged, X_new, Y_new = precond_fn(rhs_1=residual_1_unconv, rhs_2=residual_2_unconv ,
                                                        omega_shift=omega_shift[unconverged_idx])
                del residual_1_unconv, residual_2_unconv
                release_memory()

                log.info('     Preconditioning ends')
                _time_add(log, t_precond, t0)
                log.timer('  Preconditioning  cost', *t0)
                log.info(gpu_mem_info('     after preconditioning'))
                ''' put the new guess XY into the holder '''
                t0 = log.init_timer()
                size_old = size_new
                V_holder, W_holder, size_new = fill_holder(V_holder=V_holder,
                                                            W_holder=W_holder,
                                                            X_new=X_new,
                                                            Y_new=Y_new,
                                                            count=size_old)
                del X_new, Y_new, residual_1, residual_2
                release_memory()
                log.info(gpu_mem_info('     after fill holder'))
                if size_new == size_old:
                    log.warn('All new guesses kicked out during filling holder !!!!!!!')
                    break
                _time_add(log, t_fill_holder, t0)
                log.timer('  fill holder  cost', *t0)

    if ii == (max_iter -1) and max_norm >= conv_tol:
        log.warn(f'=== {problem_type.capitalize()} ABBA Krylov Solver eigen solver not converged below {conv_tol:.2e} due to max iteration limit ! ===')
        log.warn(f'Current residual norms: {r_norms.tolist()}')
        log.warn(f'max residual norms {cp.max(r_norms)}')

    converged = r_norms <= conv_tol

    log.info(f'Finished in {ii+1} steps')
    log.info(f'Maximum residual norm = {max_norm:.2e}')
    log.info(f'Final subspace size = {sub_A.shape[0]}')

    # X_full = cp.dot(x.T, V) + cp.dot(y.T, W)
    # Y_full = cp.dot(x.T, W) + cp.dot(y.T, V)
    X_full = math_helper.dot_product_xchunk_V(xT, current_V) + math_helper.dot_product_xchunk_V(yT, current_W)
    Y_full = math_helper.dot_product_xchunk_V(xT, current_W) + math_helper.dot_product_xchunk_V(yT, current_V)

    if problem_type == 'shifted_linear':
        X_full = X_full * rhs_norm
        Y_full = Y_full * rhs_norm

    _time_add(log, t_total, cpu0)

    log.timer(f'{problem_type.capitalize()} ABBA Krylov Solver total cost', *cpu0)

    _time_profiling(log, t_mvp, t_subgen, t_solve_sub, t_sub2full, t_precond, t_fill_holder, t_total)

    log.info(f'========== {problem_type.capitalize()} ABBA Krylov Solver Done ==========')

    if problem_type == 'eigenvalue':
        return converged, omega, X_full, Y_full
    elif problem_type == 'shifted_linear':
        return converged, X_full, Y_full

def nested_ABBA_krylov_solver(matrix_vector_product, hdiag, problem_type='eigenvalue',
        rhs_1=None, rhs_2=None, omega_shift=None, n_states=20, conv_tol=1e-5,
        max_iter=8, gram_schmidt=True, single=False, verbose=logger.INFO,
        init_mvp=None, precond_mvp=None, extra_init=3, extra_init_diag=8,
        init_conv_tol=1e-3, init_max_iter=10,
        precond_conv_tol=1e-2, precond_max_iter=10):
    '''
    Wrapper for Krylov solver to handle preconditioned eigenvalue, linear, or shifted linear problems.
    requires the non-diagonal approximation of A matrix, i.e., ris approximation.

    Args:
        matrix_vector_product: Callable, computes AX+BY, BX+AY.
        hdiag: 1D cupy array, diagonal of the Hamiltonian matrix.
        problem_type: str, 'eigenvalue', 'linear', 'shifted_linear'.
        rhs_1: 2D cupy array, upper part of right-hand side for linear systems (default: None).
        rhs_2: 2D cupy array, lower part of right-hand side for linear systems (default: None).
        omega_shift: Diagonal matrix for shifted linear systems (default: None).
        n_states: int, number of eigenvalues or vectors to solve.
        conv_tol: float, convergence tolerance.
        max_iter: int, maximum iterations.
        gram_schmidt: bool, use Gram-Schmidt orthogonalization.
        single: bool, use single precision.
        verbose: logger.Logger or int, logging verbosity.
        init_mvp: Callable, matrix-vector product for initial guess (default: None).
        precond_mvp: Callable, matrix-vector product for preconditioner (default: None).
        init_conv_tol: float, convergence tolerance for initial guess.
        init_max_iter: int, maximum iterations for initial guess.
        precond_conv_tol: float, convergence tolerance for preconditioner.
        precond_max_iter: int, maximum iterations for preconditioner.

    Returns:
        Output of ABBA_krylov_solver.
    '''

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    dtype = cp.float32 if single else cp.float64
    log.info(f'precision {dtype}')
    if single:
        log.info('Using single precision')
        hdiag = hdiag.astype(cp.float32, copy=False)
    else:
        log.info('Using double precision')
        hdiag = hdiag.astype(cp.float64, copy=False)

    # Validate problem type
    if problem_type not in ['eigenvalue', 'shifted_linear']:
        raise ValueError('Invalid problem type, please choose either eigenvalue or shifted_linear.')

    # Define micro_init_precond mapping
    #    the problem_type of
    #    macro problem      intial guess      preconditioner
    micro_init_precond = {
        'eigenvalue':     ['eigenvalue',     'shifted_linear'],
        'shifted_linear': ['shifted_linear', 'shifted_linear']
    }

    # Setup initial guess
    if callable(init_mvp):
        log.info('Using iterative initial guess')

        init_problem_type = micro_init_precond[problem_type][0]
        initguess_fn = partial(
            ABBA_krylov_solver,
            problem_type=init_problem_type, hdiag=hdiag,
            matrix_vector_product=init_mvp,
            conv_tol=init_conv_tol, max_iter=init_max_iter,
            gram_schmidt=gram_schmidt, single=single, verbose=log.verbose-2
        )
    else:
        log.info('Using diagonal initial guess')
        initguess_fn = None

    # Setup preconditioner
    if callable(precond_mvp):
        log.info('Using iterative preconditioner')

        precond_problem_type = micro_init_precond[problem_type][1]
        precond_fn = partial(
            ABBA_krylov_solver,
            problem_type=precond_problem_type, hdiag=hdiag,
            matrix_vector_product=precond_mvp,
            conv_tol=precond_conv_tol, max_iter=precond_max_iter,
            gram_schmidt=gram_schmidt, single=single, verbose=log.verbose-1
        )
    else:
        log.info('Using diagonal preconditioner')
        precond_fn = None

    if not init_mvp and not precond_mvp:
        log.warn(f'diagonal initial guess and preconditioner provided, using extra_init={extra_init_diag}')
        extra_init = extra_init_diag

    # Run solver
    output = ABBA_krylov_solver(
        matrix_vector_product=matrix_vector_product, hdiag=hdiag,
        problem_type=problem_type, n_states=n_states,
        rhs_1=rhs_1, rhs_2=rhs_2, omega_shift=omega_shift, extra_init=extra_init,
        initguess_fn=initguess_fn, precond_fn=precond_fn,
        conv_tol=conv_tol, max_iter=max_iter,
        gram_schmidt=gram_schmidt, single=single, verbose=verbose
    )
    log.info(RIS_PRECOND_CITATION_INFO)
    return output


def example_krylov_solver():

    cp.random.seed(42)
    A_size = 1000
    n_vec = 5
    A = cp.random.rand(A_size,A_size)*0.01
    A = A + A.T
    scaling = 30
    cp.fill_diagonal(A, (cp.random.rand(A_size)+2) * scaling)
    omega_shift = (cp.random.rand(n_vec)+2) * scaling
    rhs = cp.random.rand(n_vec, A_size) * scaling

    def matrix_vector_product(x):
        return x.dot(A)

    hdiag = cp.diag(A)

    _converged, eigenvalues, eigenvecters = krylov_solver(matrix_vector_product=matrix_vector_product, hdiag=hdiag,
                            problem_type='eigenvalue', n_states=5,
                            conv_tol=1e-5, max_iter=35,gram_schmidt=True, verbose=5, single=False)

    _converged, solution_vectors = krylov_solver(matrix_vector_product=matrix_vector_product, hdiag=hdiag,
                            problem_type='linear', rhs=rhs,
                            conv_tol=1e-5, max_iter=35,gram_schmidt=True, verbose=5, single=False)

    _converged, solution_vectors_shifted = krylov_solver(matrix_vector_product=matrix_vector_product, hdiag=hdiag,
                            problem_type='shifted_linear', rhs=rhs, omega_shift=omega_shift,
                            conv_tol=1e-5, max_iter=35,gram_schmidt=True, verbose=5, single=False)

    return eigenvalues, eigenvecters, solution_vectors, solution_vectors_shifted

def example_ABBA_krylov_solver():

    cp.random.seed(42)
    A_size = 1000
    n_vec = 5
    A = cp.random.rand(A_size,A_size)*0.01
    B = cp.random.rand(A_size,A_size)*0.005

    A = A + A.T
    B = B + B.T

    scaling = 30
    cp.fill_diagonal(A, (cp.random.rand(A_size)+2) * scaling)
    omega_shift = (cp.random.rand(n_vec)+0.5) * scaling
    rhs_1 = cp.random.rand(n_vec, A_size) * scaling
    # rhs_2 = cp.random.rand(n_vec, A_size) * scaling
    rhs_2 = rhs_1

    def matrix_vector_product(X, Y):
        U1 = X.dot(A) + Y.dot(B)
        U2 = Y.dot(A) + X.dot(B)
        return U1, U2

    hdiag = cp.diag(A)

    _converged, eigenvalues, X, Y = ABBA_krylov_solver(matrix_vector_product=matrix_vector_product, hdiag=hdiag,
                            problem_type='eigenvalue', n_states=5,
                            conv_tol=1e-5, max_iter=35,gram_schmidt=True, verbose=5, single=False)


    _converged, X_shifted, Y_shifted = ABBA_krylov_solver(matrix_vector_product=matrix_vector_product, hdiag=hdiag,
                            problem_type='shifted_linear', rhs_1=rhs_1, rhs_2=rhs_2, omega_shift=omega_shift,
                            conv_tol=1e-5, max_iter=35,gram_schmidt=True, verbose=5, single=False)

    return eigenvalues, X, Y, X_shifted, Y_shifted

