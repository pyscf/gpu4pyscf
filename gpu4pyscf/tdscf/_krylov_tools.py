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
import sys
import scipy.linalg

from gpu4pyscf.tdscf import math_helper
from gpu4pyscf.lib import logger, cusolver
from functools import partial
from pyscf.data.nist import HARTREE2EV


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
    """solve
        DX=XΩ 
        D is diagonal matrix
    """
    n_states = kwargs['n_states']
    hdiag = kwargs['hdiag']

    hdiag = hdiag.reshape(-1,)
    A_size = hdiag.shape[0]
    Dsort = hdiag.argsort()[:n_states]
    X = cp.zeros((n_states, A_size))
    X[cp.arange(n_states), Dsort] = 1.0
    return X
   
def linear_diagonal(**kwargs):
    ''' solve  DX=rhs,
        where D is a diagonal matrix'''
    hdiag = kwargs['hdiag']
    rhs = kwargs['rhs']
    return rhs / hdiag

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
    return X
    '''

    rhs = kwargs['rhs']
    hdiag = kwargs['hdiag']
    omega = kwargs['omega_shift']

    n_states = cp.shape(rhs)[0]
    assert n_states == len(omega)
    t = 1e-14

    omega = omega.reshape(-1,1)
    D = cp.repeat(hdiag.reshape(1,-1), n_states, axis=0) - omega
    '''
    force all small values not in [-t,t]
    '''
    D = cp.where( abs(D) < t, cp.sign(D)*t, D)
    X = rhs/D
    return X


'''for each problem type, setup diagonal initial guess and preconitioner '''

'''eigenvalue problem'''
_eigenvalue_diagonal_initguess = eigenvalue_diagonal
_eigenvalue_diagonal_precond  = shifted_linear_diagonal


'''linear problem'''
_linear_diagonal_initguess = linear_diagonal
_linear_diagonal_precond   = linear_diagonal


'''shifted linear problem'''
_shifted_linear_diagonal_initguess = shifted_linear_diagonal 
_shifted_linear_diagonal_precond   = shifted_linear_diagonal


def krylov_solver(matrix_vector_product, hdiag, problem_type='eigenvalue', 
                  initguess_fn=None, precond_fn=None, rhs=None, 
                  omega_shift=None, n_states=20,conv_tol=1e-5, 
                  max_iter=35, extra_init=8, gram_schmidt=True, 
                  single=False, verbose=logger.NOTE):
    '''
        Solve the eigenvalue problem, return Ω and X 
                    AX = XΩ 
        or linear system, return X
                    AX = rhs.
        or shifted linear system (Green's function), return X
                 AX - XΩ = rhs, where Ω is a diagonal matrix. 

    Theory:
           AX = XΩ 
        A(Vx) = (Vx)Ω
        V.TAV x = V.TV xΩ   
        ax = sxΩ, whehre basis overlap s=V.TV


          AX = P
        A(Vx) = P
        V.TAV x = V.TP
        ax = p, where p = V.TP   note: P != Vp

        AX - XΩ = P
        A(Vx) - (Vx) Ω = P
        V.TAV x - V.TV xΩ = V.TP
        ax - sxΩ = p

    Args:
        matrix_vector_product: function
            matrix vector product 
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
        hdiag = hdiag.astype(cp.float32)
    else:
        log.info('Using double precision')
        # assert hdiag.dtype == cp.float64
        hdiag = hdiag.astype(cp.float64)


    log.info(f'====== Krylov {problem_type.capitalize()} Solver Starts ======')
    logger.TIMER_LEVEL = 4
    logger.DEBUG1 = 4

    cpu0 = log.init_timer()
    ''' detailed timing for each sub module'''
    t_mvp         = [.0] * len(cpu0)
    t_subgen      = [.0] * len(cpu0)
    t_solve_sub   = [.0] * len(cpu0)
    t_sub2full    = [.0] * len(cpu0)
    t_precond     = [.0] * len(cpu0)
    t_fill_holder = [.0] * len(cpu0)
    t_total       = [.0] * len(cpu0)

    A_size = hdiag.shape[0]
    log.info(f'Size of A matrix = {A_size}')
    size_old = 0
    if problem_type == 'eigenvalue':
        size_new = min([n_states + extra_init, 2 * n_states, A_size])
    elif problem_type in ['linear','shifted_linear']:
        if rhs is None:
            raise ValueError('rhs is required for linear or shifted_linear problem.')
        
        size_new = rhs.shape[0]
        n_states = rhs.shape[0]

    max_N_mv = size_new + max_iter * n_states 

    holder_mem = 2*max_N_mv*A_size*hdiag.itemsize/(1024**2)
    log.info(f'  V and W holder use {holder_mem:.2f} MB memory')

    # Initialize arrays
    V_holder = cp.empty((max_N_mv, A_size))
    W_holder = cp.empty_like(V_holder)
    sub_A_holder = cp.empty((max_N_mv, max_N_mv), dtype=hdiag.dtype)

    if problem_type in ['linear','shifted_linear']:
        '''Normalize RHS for linear system'''
        rhs_norm = cp.linalg.norm(rhs, axis=1, keepdims=True)
        rhs = rhs/rhs_norm
        sub_rhs_holder = cp.empty((max_N_mv, rhs.shape[0]), dtype=hdiag.dtype)


    # Set basis projection method
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
    if problem_type == 'eigenvalue':
        init_guess = initguess_fn(n_states=size_new, hdiag=hdiag)
        if isinstance(init_guess, tuple):
            _energies, init_guess_vec = init_guess
            init_guess = init_guess_vec

    elif problem_type == 'linear':
        init_guess = initguess_fn(hdiag=hdiag, rhs=rhs)

    elif problem_type =='shifted_linear':
        init_guess = initguess_fn(hdiag=hdiag, rhs=rhs, omega_shift=omega_shift)
    
    V_holder, size_new = fill_holder(V_holder, size_old, init_guess)
    log.info('initial guess done')
    

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

    for ii in range(max_iter):

        # Matrix-vector product
        t0 = log.init_timer()
        W_holder[size_old:size_new, :] = matrix_vector_product(V_holder[size_old:size_new, :])
        _time_add(log, t_mvp, t0)

        # Project into Krylov subspace
        t0 = log.init_timer()
        sub_A_holder = math_helper.gen_VW(sub_A_holder, V_holder, W_holder, size_old, size_new, symmetry=True)
        sub_A = sub_A_holder[:size_new, :size_new]
        if problem_type in ['linear','shifted_linear']:
            sub_rhs_holder = math_helper.gen_VP(sub_rhs_holder, V_holder, rhs, size_old, size_new)
            sub_rhs = sub_rhs_holder[:size_new, :]

        _time_add(log, t_subgen, t0)

        # Solve subspace problem
        t0 = log.init_timer()

        ''' solve subsapce problem
            solution x is column-wise vectors
            each vetcor contains elements of linear combination coefficient of projection basis
        '''
        if not gram_schmidt:
            ''' no Gram Schidmit procedure, need the overlap matrix of projection basis'''
            s_holder = math_helper.gen_VW(s_holder, V_holder, V_holder, size_old, size_new, symmetry=False)
            overlap_s = s_holder[:size_new, :size_new]

        if problem_type == 'eigenvalue':
            if gram_schmidt:
                ''' solve ax=xΩ '''
                omega, x = cp.linalg.eigh(sub_A)
            else:
                ''' solve ax=sxΩ 
                # TODO need precondition step: s/d first'''
                omega, x = scipy.linalg.eigh(sub_A.get(), overlap_s.get())
                # omega, x = cusolver.eigh(sub_A, overlap_s)
                omega = cp.asarray(omega)
                x = cp.asarray(x)

            omega = omega[:n_states]
            x = x[:, :n_states]

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
                # x = cp.asarray(x)
            else:
                ''' solve ax - s xΩ = sub_rhs 
                    => s^-1 ax - xΩ = s^-1 sub_rhs
                TODO need precondition step: s/d first'''
                s_inv = cp.linalg.inv(overlap_s)
                x = scipy.linalg.solve_sylvester(s_inv.dot(sub_A).get(), -cp.diag(omega_shift).get(), s_inv.dot(sub_rhs).get())
                x = cp.asarray(x)

        _time_add(log, t_solve_sub, t0)

        t0 = log.init_timer()
        
        
        ''' compute the residual
            full_X is current guess solution 
            AX is Afull_X'''
        AX = cp.dot(x.T, W_holder[:size_new, :])
        if problem_type == 'eigenvalue':
            ''' r = AX - XΩ '''
            full_X = cp.dot(x.T, V_holder[:size_new, :])
            residual = AX - omega.reshape(-1, 1) * full_X

        elif problem_type == 'linear':
            ''' r = AX - rhs '''
            residual = AX - rhs

        elif problem_type == 'shifted_linear':
            ''' r = AX - X omega_shift - rhs '''
            full_X = cp.dot(x.T, V_holder[:size_new, :])
            residual = AX - omega_shift.reshape(-1, 1) * full_X - rhs

        _time_add(log, t_sub2full, t0)

        ''' Check convergence '''
        r_norms = cp.linalg.norm(residual, axis=1)

        max_norm = cp.max(r_norms)
        log.info(f'iter: {ii+1:<3d}   max|R|: {max_norm:<12.2e}  subspace: {sub_A.shape[0]:<8d}')

        if max_norm < conv_tol or ii == (max_iter - 1):
            break

        ''' Preconditioning step '''
        index_bool = r_norms > conv_tol
        t0 = log.init_timer()
        log.debug('     Preconditioning starts')
        residual = residual[index_bool, :]
        if problem_type == 'eigenvalue':
            new_guess = precond_fn(rhs=residual, omega_shift=omega[index_bool])
        elif problem_type == 'linear':
            new_guess = precond_fn(rhs=residual)
        elif problem_type =='shifted_linear':
            new_guess = precond_fn(rhs=residual, omega_shift=omega_shift[index_bool])
        log.debug('     Preconditioning ends')
        _time_add(log, t_precond, t0)

        
        t0 = log.init_timer()
        size_old = size_new
        V_holder, size_new = fill_holder(V_holder, size_old, new_guess)
        # if gram_schmidt:
        #     log.info(f'V_holder orthonormality: {math_helper.check_orthonormal(V_holder[:size_new, :].T)}')
        if size_new == size_old:
            log.warn('All new guesses kicked out during filling holder')
            break
        _time_add(log, t_fill_holder, t0)

    if ii == max_iter - 1 and max_norm >= conv_tol:
        log.warn(f'=== Warning: {problem_type.capitalize()} solver not converged below {conv_tol:.2e} ===')
        log.warn(f'Current residual norms: {r_norms.tolist()}')
    log.info(f'Finished in {ii+1} steps')
    log.info(f'Maximum residual norm = {max_norm:.2e}')
    log.info(f'Final subspace size = {sub_A.shape[0]}')

    # linear problem didn't yet explicitly construct full_X
    if problem_type == 'linear':
        full_X = cp.dot(x.T, V_holder[:size_new, :])
    
    if problem_type in['linear', 'shifted_linear']:
        full_X = full_X * rhs_norm

    _time_add(log, t_total, cpu0)

    log.timer(f'{problem_type.capitalize()} solver total cost', *cpu0)
    _time_profiling(log, t_mvp, t_subgen, t_solve_sub, t_sub2full, t_precond, t_fill_holder, t_total)

    log.info(f'========== {problem_type.capitalize()} Solver Done ==========')

    if problem_type == 'eigenvalue':
        return omega, full_X
    elif problem_type in ['linear', 'shifted_linear']:
        return full_X

def nested_krylov_solver(matrix_vector_product, hdiag, problem_type='eigenvalue',
        rhs=None, omega_shift=None, n_states=20, conv_tol=1e-5, 
        max_iter=8, gram_schmidt=True, single=False, verbose=logger.INFO, 
        init_mvp=None, precond_mvp=None, extra_init=3, extra_init_diag=8,
        init_conv_tol=1e-3, init_max_iter=10,
        precond_conv_tol=1e-2, precond_max_iter=10):
    """
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
    """

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    dtype = cp.float32 if single else cp.float64
    log.info(f'precision {dtype}')
    if single:
        log.info('Using single precision')
        hdiag = hdiag.astype(cp.float32)
    else:
        log.info('Using double precision')
        hdiag = hdiag.astype(cp.float64)

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
        gram_schmidt=gram_schmidt, single=single, verbose=verbose
    )

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

    eigenvalues, eigenvecters = krylov_solver(matrix_vector_product=matrix_vector_product, hdiag=hdiag,
                            problem_type='eigenvalue', n_states=5,
                            conv_tol=1e-5, max_iter=35,gram_schmidt=True, verbose=5, single=False)

    solution_vectors = krylov_solver(matrix_vector_product=matrix_vector_product, hdiag=hdiag,
                            problem_type='linear', rhs=rhs,
                            conv_tol=1e-5, max_iter=35,gram_schmidt=True, verbose=5, single=False)
    
    solution_vectors_shifted = krylov_solver(matrix_vector_product=matrix_vector_product, hdiag=hdiag,
                            problem_type='shifted_linear', rhs=rhs, omega_shift=omega_shift,
                            conv_tol=1e-5, max_iter=35,gram_schmidt=True, verbose=5, single=False)
    
    return eigenvalues, eigenvecters, solution_vectors, solution_vectors_shifted