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


from gpu4pyscf.tdscf import math_helper, _krylov_tools
from gpu4pyscf.tdscf._krylov_tools import eigenvalue_diagonal, _time_add, _time_profiling, RIS_PRECOND_CITATION_INFO
from gpu4pyscf.tdscf.math_helper import gpu_mem_info, release_memory, get_avail_gpumem, get_avail_cpumem
from gpu4pyscf.lib.cupy_helper import asarray as cuasarray

from gpu4pyscf.lib import logger
from functools import partial
from pyscf.lib.misc import current_memory
from pyscf.data.nist import HARTREE2EV

def ABBA_eigenvalue_diagonal(**kwargs):
    '''solve
        [ D 0 ] X = [ 1  0 ] X Ω
        [ 0 D ] Y   [ 0 -1 ] Y
        D is diagonal matrix
        DX =  X Ω => D = Ω
        DY = -Y Ω => 2DY = 0 => Y=0

        X_p_Y := X
        X_m_Y := X
    '''
    _converged, _energies, X = eigenvalue_diagonal(**kwargs)
    _converged, _energies = True, None
    return _converged, _energies, X, X

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
                  max_iter=35, extra_init=None, gram_schmidt=True,
                  restart_subspace=None, in_ram=False, gs_initial=False,
                  single=False, verbose=logger.NOTE):
    '''
        This solver is used to solve the following problems:

        (1) eigenvalue problem, return Ω and [X,Y]
            [ A B ] X = [ 1  0 ] X Ω
            [ B A ] Y   [ 0 -1 ] Y
            e.g. Casida equation

            practival implementation:
            (A+B)(X+Y) = (X-Y)Ω
            (A-B)(X-Y) = (X+Y)Ω

        (2) shifted linear system , return X
            [ A B ] X - [ 1  0 ] Y Ω = [rhs_1]
            [ B A ] Y   [ 0 -1 ] X     [rhs_2]
            where Ω is a diagonal matrix.
            e.g. dynamic polarizability

            practival implementation:
            (A+B)(X+Y) - (X-Y)Ω = rhs_1 + rhs_2
            (A-B)(X-Y) - (X+Y)Ω = rhs_1 - rhs_2


        Note:
        in the case of linear equation,
            [ A B ] X  = [rhs_1]
            [ B A ] Y  = [rhs_2],
        =>  (A+B)(X+Y) = rhs_1 + rhs_2
        fallback to simple krylov solver

    Theory:

    (1) Eigenvalue problem:

        (A+B)(X+Y) = (X-Y)Ω
        (A-B)(X-Y) = (X+Y)Ω

        use a linear combination of projection basis V+W and V-W to expand X+Y and X-Y
        X+Y = (V+W)(x+y)
        X-Y = (V-W)(x-y)

        =>

        (A+B)(V+W)(x+y) = (V-W)(x-y)Ω
        (A-B)(V-W)(x-y) = (V+W)(x+y)Ω

        =>
        (V+W).T(A+B)(V+W)(x+y) = (V+W).T(V-W)(x-y)Ω
        (V-W).T(A-B)(V-W)(x-y) = (V-W).T(V+W)(x+y)Ω

        note that
        (V+W).T(A+B)(V+W) = a+b
        (V-W).T(A-B)(V-W) = a-b
        (V+W).T(V-W) = σ - π
        (V-W).T(V+W) = σ + π,  and (σ + π).T = σ - π


        so that the equation becomes:
        (a+b)(x+y) = (σ - π)(x-y)Ω
        (a-b)(x-y) = (σ + π)(x+y)Ω

        => in matrix form:
        [ a b ] x = [  σ  π ] X Ω
        [ b a ] y   [ -π -σ ] Y

        where

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
    log.info(f'the maximum number of vectors in holder is {max_N_mv}')

    holder_mem = 4 * max_N_mv * A_size * hdiag.itemsize/(1024**2)
    log.info(f'  vector holder totoal use {holder_mem:.2f} MB memory')

    xp = np if in_ram else cp
    log.info(f'xp {xp}')


    V_p_W_holder = xp.empty((max_N_mv, A_size), dtype=hdiag.dtype)
    V_m_W_holder = xp.empty_like(V_p_W_holder)

    U_VpW_holder = xp.empty_like(V_p_W_holder) # for (A+B)(V+W)
    U_VmW_holder = xp.empty_like(V_m_W_holder) # for (A-B)(V-W)

    sub_a_p_b_holder = cp.empty((max_N_mv,max_N_mv), dtype=hdiag.dtype)
    sub_a_m_b_holder = cp.empty_like(sub_a_p_b_holder)
    sub_sigma_p_pi_holder = cp.empty_like(sub_a_p_b_holder)

    '''
    set up initial guess, V= TDA initial guess, W=0
    '''

    if problem_type == 'shifted_linear':
        # TODO !!!
        pass
        # rhs = cp.hstack((rhs_1,rhs_2))
        # rhs_norm = cp.linalg.norm(rhs, axis=1, keepdims = True)
        # rhs_1 = rhs_1/rhs_norm
        # rhs_2 = rhs_2/rhs_norm
        # VP_holder = cp.empty((max_N_mv,rhs.shape[0]), dtype=hdiag.dtype)
        # VQ_holder = cp.empty_like(VP_holder)
        # WP_holder = cp.empty_like(VP_holder)
        # WQ_holder = cp.empty_like(VP_holder)


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
        _converged, _energies, init_guess_XpY, init_guess_XmY = initguess_fn(n_states=size_new, hdiag=hdiag)

    elif problem_type =='shifted_linear':
        _converged, init_guess_XpY, init_guess_XmY = initguess_fn(hdiag=hdiag, rhs_1=rhs_1, rhs_2=rhs_2, omega_shift=omega_shift)
    log.timer(f' {problem_type.capitalize()} initguess_fn cost', *cpu0)

    cpu0 = log.init_timer()
    log.info(gpu_mem_info('before put initial guess into V_holder and W_holder'))


    if gs_initial:
        extra_init_XpY = init_guess_XpY[n_states:, :]
        extra_init_XmY = init_guess_XmY[n_states:, :]
        if in_ram:
            extra_init_XpY = extra_init_XpY.get()
            extra_init_XmY = extra_init_XmY.get()
        V_p_W_holder[:n_extra_init, :] = extra_init_XpY
        V_m_W_holder[:n_extra_init, :] = extra_init_XmY

        del extra_init_XpY
        del extra_init_XmY


        n_states_XpY = init_guess_XpY[:n_states, :]
        n_states_XmY = init_guess_XmY[:n_states, :]
        if in_ram:
            n_states_XpY = n_states_XpY.get()
            n_states_XmY = n_states_XmY.get()

        V_p_W_holder[n_extra_init:n_extra_init+n_states, :] = n_states_XpY
        V_m_W_holder[n_extra_init:n_extra_init+n_states, :] = n_states_XmY
        del n_states_XpY, n_states_XmY
        size_new = init_guess_XpY.shape[0]

    else:
        if n_extra_init > 0:
            size_new = fill_holder(V_p_W_holder, 0, init_guess_XpY[n_states:, :])# first fill extra_init vectors
            size_new = fill_holder(V_p_W_holder, size_new, init_guess_XpY[:n_states, :]) # n_states vectors

            size_new = fill_holder(V_m_W_holder, 0, init_guess_XmY[n_states:, :])# first fill extra_init vectors
            size_new = fill_holder(V_m_W_holder, size_new, init_guess_XmY[:n_states, :]) # n_states vectors
        else:
            size_new = fill_holder(V_p_W_holder, 0, init_guess_XpY)
            size_new = fill_holder(V_m_W_holder, 0, init_guess_XmY)

    del init_guess_XpY, init_guess_XmY
    release_memory()

    log.timer(f' {problem_type.capitalize()} init_guess_XpY and init_guess_XmY fill_holder cost', *cpu0)
    log.info('initial guess done')
    log.info(gpu_mem_info('after put initial guess into V_p_W_holder and V_m_W_holder'))


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
        X_p_Y = V_p_W_holder[size_old:size_new, :]
        X_m_Y = V_m_W_holder[size_old:size_new, :]

        if in_ram:
            X_p_Y = cuasarray(X_p_Y)
            X_m_Y = cuasarray(X_m_Y)
            release_memory()

        log.info(f'     X_p_Y {X_p_Y.shape} {X_p_Y.nbytes//1024**2} MB')
        log.info(f'     X_m_Y {X_m_Y.shape} {X_m_Y.nbytes//1024**2} MB')



        ApB_XpY, AmB_XmY = matrix_vector_product(X_p_Y, X_m_Y)
        del X_p_Y, X_m_Y
        release_memory()
        cp.cuda.Stream.null.synchronize()

        log.info(gpu_mem_info('     after MVP'))

        if in_ram:
            ApB_XpY = ApB_XpY.get()
            AmB_XmY = AmB_XmY.get()
            release_memory()
        U_VpW_holder[size_old:size_new, :] = ApB_XpY
        U_VmW_holder[size_old:size_new, :] = AmB_XmY
        del ApB_XpY, AmB_XmY
        gc.collect()
        release_memory()

        log.info(f'     holder memory usage: 4 * {V_p_W_holder[:size_new, :].nbytes/1024**3:.2f} GB')
        log.info(f'     subspace size / maximum subspace size: {size_new} / {max_N_mv}')

        n_mvp_record.append(size_new - size_old)
        log.info(gpu_mem_info('     MVP stored in holder'))

        _time_add(log, t_mvp, t0)
        log.timer('  MVP total cost', *t0)

        ''' Project into Krylov subspace '''
        t0 = log.init_timer()
        sub_a_p_b_holder = math_helper.gen_VW(sub_a_p_b_holder, V_p_W_holder, U_VpW_holder,
                                              size_old, size_new, symmetry=True)
        sub_a_m_b_holder = math_helper.gen_VW(sub_a_m_b_holder, V_m_W_holder, U_VmW_holder,
                                              size_old, size_new, symmetry=True)

        sub_sigma_p_pi_holder = math_helper.gen_VW(sub_sigma_p_pi_holder, V_m_W_holder, V_p_W_holder,
                                              size_old, size_new, symmetry=False)

        a_p_b = sub_a_p_b_holder[:size_new, :size_new]
        a_m_b = sub_a_m_b_holder[:size_new, :size_new]
        sigma_p_pi = sub_sigma_p_pi_holder[:size_new, :size_new]

        if problem_type == 'shifted_linear':
            pass

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
            omega, x_p_y, x_m_y = math_helper.TDDFT_subspace_eigen_solver4(a_p_b, a_m_b, sigma_p_pi, n_states)
        elif problem_type == 'shifted_linear':
            pass
            # x,y = math_helper.TDDFT_subspace_linear_solver(sub_A, sub_B, sigma, pi, sub_rhs_1, sub_rhs_2, omega_shift)

        _time_add(log, t_solve_sub, t0)
        log.info(f' Energies (eV): {[round(e,3) for e in (omega*HARTREE2EV).tolist()]}')

        '''
        compute the residual
        X_full, Y_full is current guess solution

        (1) Eigenvalue system:
        r_xpy = (U1 + U2)(x+y) - (V-W)(x-y)*omega
        r_xmy = (U1 - U2)(x-y) - (V+W)(x+y)*omega

        (2) Shifted linear system:
        r_xpy = (U1 + U2)(x+y) - (V-W)(x-y)*omega_shift - (rhs_1 - rhs_2)
        r_xmy = (U1 - U2)(x-y) - (V+W)(x+y)*omega_shift + (rhs_1 + rhs_2)
        '''
        t0 = log.init_timer()

        x_p_yT = x_p_y.T
        x_m_yT = x_m_y.T

        residual_xpy = math_helper.dot_product_xchunk_V(x_p_yT, U_VpW_holder[:size_new,:])
        residual_xmy = math_helper.dot_product_xchunk_V(x_m_yT, U_VmW_holder[:size_new,:])


        if problem_type == 'eigenvalue':
            omega_x_p_yT = omega[:,None] * x_p_yT
            omega_x_m_yT = omega[:,None] * x_m_yT

            residual_xpy = math_helper.dot_product_xchunk_V(omega_x_m_yT, V_m_W_holder[:size_new,:], alpha=-1.0, out=residual_xpy)
            residual_xmy = math_helper.dot_product_xchunk_V(omega_x_p_yT, V_p_W_holder[:size_new,:], alpha=-1.0, out=residual_xmy)
            del omega_x_m_yT, omega_x_p_yT

        elif problem_type == 'shifted_linear':
            pass
            # TODO !!!
            # residual_xpy -= rhs_1
            # residual_xmy += rhs_2

        _time_add(log, t_sub2full, t0)
        log.timer('  compute residual cost', *t0)
        log.info(gpu_mem_info('     after sub2full compute residual'))

        ''' Check convergence
            residual_1 R_x
            residual_2 R_y

           residual_xpy = R_x + R_y
           residual_xmy = R_x - R_y

           residual_xpy - residual_xmy = 2*R_y
           residual_xmy + R_y = R_x
        '''

        residual_1 = (residual_xpy + residual_xmy) * 0.5
        residual_2 = (residual_xpy - residual_xmy) * 0.5
        del residual_xpy, residual_xmy
        release_memory()
        # residual = cp.hstack((residual_1, residual_2))
        # r_norms = cp.linalg.norm(residual, axis=1)
        r_norms = cp.sqrt((residual_1**2).sum(axis=1) + (residual_2**2).sum(axis=1))


        # residual_xpy -= residual_xmy
        # residual_xpy *= 0.5

        # residual_2 = residual_xpy

        # residual_xmy += residual_2

        # residual_1 = residual_xmy

        # r_norms = cp.sqrt((residual_1**2).sum(axis=1) + (residual_2**2).sum(axis=1))

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

                X_p_Y_full = math_helper.dot_product_xchunk_V(x_p_yT, V_p_W_holder[:size_new,:])
                X_m_Y_full = math_helper.dot_product_xchunk_V(x_m_yT, V_m_W_holder[:size_new,:])

                size_old = n_extra_init
                size_new = fill_holder(V_p_W_holder, n_extra_init, X_p_Y_full)
                size_new = fill_holder(V_m_W_holder, n_extra_init, X_m_Y_full)

                del X_p_Y_full, X_m_Y_full
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

                # residual_1_unconv /= cp.linalg.norm(residual_1_unconv, axis=1, keepdims=True)
                # residual_2_unconv /= cp.linalg.norm(residual_2_unconv, axis=1, keepdims=True)

                t0 = log.init_timer()
                log.info(gpu_mem_info('     before preconditioning'))
                log.info('     Preconditioning starts')
                if problem_type == 'eigenvalue':
                    _converged, X_new, Y_new = precond_fn(rhs_1=residual_1_unconv, rhs_2=residual_2_unconv,
                                                        omega_shift=omega[unconverged_idx])

                elif problem_type =='shifted_linear':
                    pass
                    # _converged, X_new, Y_new = precond_fn(rhs_1=residual_1_unconv, rhs_2=residual_2_unconv ,
                    #                                     omega_shift=omega_shift[unconverged_idx])
                del residual_1_unconv, residual_2_unconv, residual_1, residual_2

                '''
                X_p_Y_new = X_new + Y_new
                X_m_Y_new = 2*X_new - (X_new + Y_new)
                '''

                X_p_Y_new = (X_new + Y_new)
                X_m_Y_new = (X_new - Y_new)


                # Y_new += X_new
                # X_p_Y_new = Y_new   # X+Y

                # X_new *= 2          # 2X
                # X_new -= X_p_Y_new  # 2X - (X+Y) = X-Y

                # X_m_Y_new = X_new

                release_memory()

                log.info('     Preconditioning ends')
                _time_add(log, t_precond, t0)
                log.timer('  Preconditioning  cost', *t0)
                log.info(gpu_mem_info('     after preconditioning'))
                ''' put the new guess XY into the holder '''
                t0 = log.init_timer()
                size_old = size_new
                size_old1 = size_new
                size_old2 = size_new

                size_new1 = fill_holder(V_p_W_holder, size_old1, X_p_Y_new)
                size_new2= fill_holder(V_m_W_holder, size_old2, X_m_Y_new)

                assert size_new1 == size_new2, 'size_new1 and size_new2 are not equal'

                size_new = size_new1

                del X_p_Y_new, X_m_Y_new
                release_memory()

                # if gram_schmidt:
                #     log.debug(f'V_p_W_holder orthonormality: {math_helper.check_orthonormal(V_p_W_holder[:size_new, :])}')
                #     log.debug(f'V_m_W_holder orthonormality: {math_helper.check_orthonormal(V_m_W_holder[:size_new, :])}')

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
    log.info(f'Final subspace size = {size_new}')

    X_p_Y_full = math_helper.dot_product_xchunk_V(x_p_yT, V_p_W_holder[:size_new,:])
    X_m_Y_full = math_helper.dot_product_xchunk_V(x_m_yT, V_m_W_holder[:size_new,:])

    if problem_type == 'shifted_linear':
        pass
        # X_full = X_full * rhs_norm
        # Y_full = Y_full * rhs_norm

    X_p_Y_full -= X_m_Y_full
    X_p_Y_full *= 0.5 # Y_full

    Y_full = X_p_Y_full

    X_m_Y_full += Y_full # X_full
    X_full = X_m_Y_full

    normality_error = cp.linalg.norm( (cp.dot(X_full, X_full.T) - cp.dot(Y_full, Y_full.T)) - cp.eye(n_states) )
    log.debug(f'check normality of X.TX - Y.TY - I = {normality_error:.2e}')

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

