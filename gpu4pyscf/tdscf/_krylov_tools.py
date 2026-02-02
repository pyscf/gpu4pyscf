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
    Dsort = hdiag.argsort()[:n_states]
    X = cp.zeros((n_states, A_size))
    X[cp.arange(n_states), Dsort] = 1.0
    _converged, _energies = True, None
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
    return X
    '''

    rhs = kwargs['rhs']
    hdiag = kwargs['hdiag']
    omega = kwargs['omega_shift']

    n_states = rhs.shape[0]
    assert n_states == len(omega)
    t = 1e-14

    omega = omega.reshape(-1,1)
    D = cp.repeat(hdiag.reshape(1,-1), n_states, axis=0) - omega
    '''
    force all small values not in [-t,t]
    '''
    D = cp.where( abs(D) < t, cp.sign(D)*t, D)
    X = rhs/D
    _converged = True
    return _converged, X


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
        hdiag = hdiag.astype(cp.float32)
    else:
        log.info('Using double precision')
        # assert hdiag.dtype == cp.float64
        hdiag = hdiag.astype(cp.float64)


    log.info(f'====== {problem_type.capitalize()} Krylov Solver Starts ======')
    logger.TIMER_LEVEL = 4
    logger.DEBUG1 = 4

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
    if problem_type == 'eigenvalue':
        _converged, _energies, init_guess_X = initguess_fn(n_states=size_new, hdiag=hdiag)

    elif problem_type == 'linear':
        _converged, init_guess_X = initguess_fn(hdiag=hdiag, rhs=rhs)

    elif problem_type =='shifted_linear':
        _converged, init_guess_X = initguess_fn(hdiag=hdiag, rhs=rhs, omega_shift=omega_shift)
    
    V_holder, size_new = fill_holder(V_holder, size_old, init_guess_X)
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

        ''' Matrix-vector product '''
        t0 = log.init_timer()
        W_holder[size_old:size_new, :] = matrix_vector_product(V_holder[size_old:size_new, :])
        _time_add(log, t_mvp, t0)

        ''' Project into Krylov subspace '''
        t0 = log.init_timer()
        sub_A_holder = math_helper.gen_VW(sub_A_holder, V_holder, W_holder, size_old, size_new, symmetry=True)
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
            AX is A.dot(full_X)'''
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
            _converged, X_new = precond_fn(rhs=residual, omega_shift=omega[index_bool])
        elif problem_type == 'linear':
            _converged, X_new = precond_fn(rhs=residual)
        elif problem_type =='shifted_linear':
            _converged, X_new = precond_fn(rhs=residual, omega_shift=omega_shift[index_bool])
        log.debug('     Preconditioning ends')
        _time_add(log, t_precond, t0)

        ''' put the new guess XY into the holder '''
        t0 = log.init_timer()
        size_old = size_new
        V_holder, size_new = fill_holder(V_holder, size_old, X_new)
        # if gram_schmidt:
        #     log.info(f'V_holder orthonormality: {math_helper.check_orthonormal(V_holder[:size_new, :].T)}')
        if size_new == size_old:
            log.warn('All new guesses kicked out during filling holder !!!!!!!')
            break
        _time_add(log, t_fill_holder, t0)

    if ii == max_iter - 1 and max_norm >= conv_tol:
        log.warn(f'=== {problem_type.capitalize()} Krylov Solver not converged below {conv_tol:.2e} due to max iteration limit ! ===')
        log.warn(f'Current residual norms: {r_norms.tolist()}')
        log.warn(f'max residual norms {cp.max(r_norms)}')

    converged = r_norms <= conv_tol

    log.info(f'Finished in {ii+1} steps')
    log.info(f'Maximum residual norm = {max_norm:.2e}')
    log.info(f'Final subspace size = {sub_A.shape[0]}')

    # linear problem didn't yet explicitly construct full_X
    if problem_type == 'linear':
        full_X = cp.dot(x.T, V_holder[:size_new, :])
    
    if problem_type in['linear', 'shifted_linear']:
        full_X = full_X * rhs_norm

    _time_add(log, t_total, cpu0)

    log.timer(f'{problem_type.capitalize()} Krylov Solver total cost', *cpu0)
    _time_profiling(log, t_mvp, t_subgen, t_solve_sub, t_sub2full, t_precond, t_fill_holder, t_total)

    log.info(f'========== {problem_type.capitalize()} Krylov Solver Done ==========')

    if problem_type == 'eigenvalue':
        return converged, omega, full_X
    elif problem_type in ['linear', 'shifted_linear']:
        return converged, full_X

def nested_krylov_solver(matrix_vector_product, hdiag, problem_type='eigenvalue',
        rhs=None, omega_shift=None, n_states=20, conv_tol=1e-5, 
        max_iter=8, gram_schmidt=True, single=False, verbose=logger.INFO, 
        init_mvp=None, precond_mvp=None, extra_init=3, extra_init_diag=8,
        init_conv_tol=1e-3, init_max_iter=10,
        precond_conv_tol=1e-2, precond_max_iter=10):
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

def ABBA_shifted_linear_diagonal(**kwargs):
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

    N_states = rhs_1.shape[0]
    t = 1e-8
    omega = omega.reshape(-1,1)

    d = cp.repeat(hdiag.reshape(1,-1), N_states, axis=0)

    D_x = d - omega
    D_x = cp.where(abs(D_x) < t, cp.sign(D_x)*t, D_x)

    D_y = d + omega
    D_y = cp.where(abs(D_y) < t, cp.sign(D_y)*t, D_y)

    X_new = rhs_1/D_x
    Y_new = rhs_2/D_y

    _converged = True
    return _converged, X_new, Y_new


'''eigenvalue problem'''
_ABBA_eigenvalue_diagonal_initguess = ABBA_eigenvalue_diagonal
_ABBA_eigenvalue_diagonal_precond  = ABBA_shifted_linear_diagonal



'''shifted linear problem'''
_ABBA_shifted_linear_diagonal_initguess = ABBA_shifted_linear_diagonal 
_ABBA_shifted_linear_diagonal_precond   = ABBA_shifted_linear_diagonal

def ABBA_krylov_solver(matrix_vector_product, hdiag, problem_type='eigenvalue', 
                  initguess_fn=None, precond_fn=None, rhs_1=None, rhs_2=None,
                  omega_shift=None, n_states=20,conv_tol=1e-5, 
                  max_iter=35, extra_init=8, gram_schmidt=True, 
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


    (1) Shifted linear system: 
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
        hdiag = hdiag.astype(cp.float32)
    else:
        log.info('Using double precision')
        # assert hdiag.dtype == cp.float64
        hdiag = hdiag.astype(cp.float64)



    log.info(f'====== {problem_type.capitalize()} ABBA Krylov Solver Starts ======')
    logger.TIMER_LEVEL = 4
    logger.DEBUG1 = 4

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
        size_new = min([n_states + extra_init, 2 * n_states, A_size])
    elif problem_type == 'shifted_linear':
        if rhs_1 is None or rhs_2 is None:
            raise ValueError('rhs_1 and rhs_2 is required for shifted_linear problem.')
        
        size_new = rhs_1.shape[0]
        n_states = rhs_1.shape[0]

    max_N_mv = size_new + max_iter * n_states 

    holder_mem = 4 * max_N_mv * A_size * hdiag.itemsize/(1024**2)
    log.info(f'  V W U1 U2 holder use {holder_mem:.2f} MB memory')

    V_holder = cp.zeros((max_N_mv, A_size),dtype=hdiag.dtype)
    W_holder = cp.zeros_like(V_holder)

    U1_holder = cp.empty_like(V_holder)
    U2_holder = cp.empty_like(V_holder)

    VU1_holder = cp.empty((max_N_mv,max_N_mv),dtype=hdiag.dtype)
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
    if problem_type == 'eigenvalue':
        _converged, _energies, init_guess_X, init_guess_Y = initguess_fn(n_states=size_new, hdiag=hdiag)

    elif problem_type =='shifted_linear':
        _converged, init_guess_X, init_guess_Y = initguess_fn(hdiag=hdiag, rhs_1=rhs_1, rhs_2=rhs_2, omega_shift=omega_shift)
    
    V_holder, W_holder, size_new = fill_holder(V_holder=V_holder, 
                                               W_holder=W_holder,
                                               m=size_old, 
                                               X_new=init_guess_X, 
                                               Y_new=init_guess_Y)
    log.info('initial guess done')

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

    for ii in range(max_iter):

        ''' Matrix-vector product '''
        t0 = log.init_timer()
        U1_holder[size_old:size_new, :], U2_holder[size_old:size_new, :] = matrix_vector_product(
                                                                            X=V_holder[size_old:size_new, :],
                                                                            Y=W_holder[size_old:size_new, :])
        _time_add(log, t_mvp, t0)

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

        ''' solve subsapce problem
            solution x,y are column-wise vectors
            each vetcor contains elements of linear combination coefficient of projection basis
        '''
        t0 = log.init_timer()
        if problem_type == 'eigenvalue':
            omega, x, y = math_helper.TDDFT_subspace_eigen_solver2(sub_A, sub_B, sigma, pi, n_states)
        elif problem_type == 'shifted_linear':
            x,y = math_helper.TDDFT_subspace_linear_solver(sub_A, sub_B, sigma, pi, sub_rhs_1, sub_rhs_2, omega_shift)

        _time_add(log, t_solve_sub, t0)

        '''
        compute the residual
        X_full, Y_full is current guess solution 
        '''
        t0 = log.init_timer()

        V = V_holder[:size_new,:]
        W = W_holder[:size_new,:]
        U1 = U1_holder[:size_new, :]
        U2 = U2_holder[:size_new, :]

        X_full = cp.dot(x.T, V) + cp.dot(y.T, W)
        Y_full = cp.dot(x.T, W) + cp.dot(y.T, V)

        if problem_type == 'eigenvalue':
            residual_1 = cp.dot(x.T, U1) + cp.dot(y.T, U2) - omega.reshape(-1, 1) * X_full
            residual_2 = cp.dot(x.T, U2) + cp.dot(y.T, U1) + omega.reshape(-1, 1) * Y_full

        elif problem_type == 'shifted_linear':
            residual_1 = cp.dot(x.T, U1) + cp.dot(y.T, U2) - omega_shift.reshape(-1, 1) * X_full - rhs_1
            residual_2 = cp.dot(x.T, U2) + cp.dot(y.T, U1) + omega_shift.reshape(-1, 1) * Y_full - rhs_2

        _time_add(log, t_sub2full, t0)

        ''' Check convergence '''
        residual = cp.hstack((residual_1, residual_2))

        r_norms = cp.linalg.norm(residual, axis=1)
        max_norm = cp.max(r_norms)

        log.info(f'iter: {ii+1:<3d}, max|R|: {max_norm:<10.2e} subspace_size = {sub_A.shape[0]}')

        if max_norm < conv_tol or ii == (max_iter -1):
            break

        '''  preconditioning step '''
        index_bool = r_norms > conv_tol
        residual_1 = residual_1[index_bool,:]
        residual_2 = residual_2[index_bool,:]
        t0 = log.init_timer()
        log.debug('     Preconditioning starts')
        if problem_type == 'eigenvalue':
            _converged, X_new, Y_new = precond_fn(rhs_1=residual_1, rhs_2=residual_2, omega_shift=omega[index_bool])
            
        elif problem_type =='shifted_linear':
            _converged, X_new, Y_new = precond_fn(rhs_1=residual_1, rhs_2=residual_2, omega_shift=omega_shift[index_bool])
        
        log.debug('     Preconditioning ends')
        _time_add(log, t_precond, t0)

        ''' put the new guess XY into the holder '''
        t0 = log.init_timer()
        size_old = size_new
        V_holder, W_holder, size_new = fill_holder(V_holder=V_holder,
                                                    W_holder=W_holder,
                                                    X_new=X_new,
                                                    Y_new=Y_new,
                                                    m=size_old)
        

        if size_new == size_old:
            log.warn('All new guesses kicked out during filling holder !!!!!!!')
            break
        _time_add(log, t_fill_holder, t0)
    
    if ii == (max_iter -1) and max_norm >= conv_tol:
        log.warn(f'=== {problem_type.capitalize()} ABBA Krylov Solver eigen solver not converged below {conv_tol:.2e} due to max iteration limit ! ===')
        log.warn(f'Current residual norms: {r_norms.tolist()}')
        log.warn(f'max residual norms {cp.max(r_norms)}')

    converged = r_norms <= conv_tol

    log.info(f'Finished in {ii+1} steps')
    log.info(f'Maximum residual norm = {max_norm:.2e}')
    log.info(f'Final subspace size = {sub_A.shape[0]}')

    
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
        hdiag = hdiag.astype(cp.float32)
    else:
        log.info('Using double precision')
        hdiag = hdiag.astype(cp.float64)

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

