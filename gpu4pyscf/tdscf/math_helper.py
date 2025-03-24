#!/usr/bin/python

import numpy as np
import cupy as cp
import scipy
import time
# import os, psutil
cp.set_printoptions(linewidth=250, threshold=cp.inf, precision=3)


def TDA_diag_initial_guess(V_holder, N_states, hdiag):
    '''
    N_states is the amount of initial guesses
    sort out the smallest value of hdiag, the corresponding position in the 
    initial guess set as 1.0, everywhere else set as 0.0
    '''
    # print('type(hdiag)', type(hdiag))
    hdiag = hdiag.reshape(-1,)
    Dsort = hdiag.argsort()[:N_states]
    V_holder[cp.arange(N_states), Dsort] = 1.0
    return V_holder

def TDA_diag_preconditioner(residual, omega, hdiag):
    '''
    DX - XΩ = r
    '''

    N_states = cp.shape(residual)[0]
    t = 1e-14

    omega = omega.reshape(-1,1)
    D = cp.repeat(hdiag.reshape(1,-1), N_states, axis=0) - omega
    '''
    force all small values not in [-t,t]
    '''
    D = cp.where( abs(D) < t, cp.sign(D)*t, D)
    new_guess = residual/D

    return new_guess

def TDDFT_diag_preconditioner(R_x, R_y, omega, hdiag):
    '''
    preconditioners for each corresponding residual (state)
    '''
    N_states = R_x.shape[0]
    t = 1e-8
    d = cp.repeat(hdiag.reshape(1,-1), N_states, axis=0)

    omega = omega.reshape(-1,1)
    D_x = d - omega
    D_x = cp.where(abs(D_x) < t, cp.sign(D_x)*t, D_x)
    D_x_inv = D_x**-1

    D_y = d + omega
    D_y = cp.where(abs(D_y) < t, cp.sign(D_y)*t, D_y)
    D_y_inv = D_y**-1

    X_new = R_x*D_x_inv
    Y_new = R_y*D_y_inv
    return X_new, Y_new

# def spolar_diag_initprec(RHS, hdiag=delta_hdiag2, conv_tol=None):
#
#     d = hdiag.reshape(-1,1)
#     RHS = RHS/d
#
#     return RHS
def level_shit_index(eigenvalue):
    for i in range(len(eigenvalue)):
        if eigenvalue[i] > 1e-3:
            return i

            

def commutator(A,B):
    commu = cp.dot(A,B) - cp.dot(B,A)
    return commu

def cond_number(A):
    s,u = cp.linalg.eig(A)
    s = abs(s)
    cond = max(s)/min(s)
    return cond

def matrix_power(S,a, epsilon=None):
    '''X == S^a'''
    s,ket = cp.linalg.eigh(S)
    # s = s**a
    if epsilon:
        valid_indices = s >= epsilon
        if len(valid_indices) != len(s):
            print(f'warning!!! truncating matrix during matrix power {a}: {len(valid_indices)} vs {len(s)}')
        s = s[valid_indices]
        ket = ket[:, valid_indices]
    
    s = s**a
    
    X = cp.dot(ket*s,ket.T)

    return X

def copy_array(A):
    B = cp.zeros_like(A)
    dim = len(B.shape)
    if dim == 1:
        B[:,] = A[:,]
    elif dim == 2:
        B[:,:] = A[:,:]
    elif dim == 3:
        B[:,:,:] = A[:,:,:]
    return B

def Gram_Schmidt_bvec(A, bvec):
    '''orthonormalize vector b against all vectors in A
       b = b - A*(A.T*b)
       suppose A is orthonormalized
    '''
    if A.shape[0] != 0:
        projections_coeff = cp.dot(A, bvec.T)
        bvec -= cp.dot(projections_coeff.T, A)
    return bvec

def VW_Gram_Schmidt(x, y, V, W):
    '''orthonormalize vector |x,y> against all vectors in |V,W>'''

    m = cp.dot(V, x.T) + cp.dot(W, y.T)

    n = cp.dot(W, x.T) + cp.dot(V, y.T)

    x = x - cp.dot(m.T, V) - cp.dot(n.T, W)

    y = y - cp.dot(m.T, W) - cp.dot(n.T, V)

    return x, y

def block_symmetrize(A,m,n):
    A[m:n,:m] = A[:m,m:n].T
    return A

def gen_anisotropy(a):

    # a = 0.5*(a.T + a)
    # tr = (1.0/3.0)*cp.trace(a)
    # xx = a[0,0]
    # yy = a[1,1]
    # zz = a[2,2]

    # xy = a[0,1]
    # xz = a[0,2]
    # yz = a[1,2]
    # anis = (xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2 + 6*(xz**2 + xy**2 + yz**2)
    # anis = 0.5*anis
    # anis = anis**0.5
    a = 0.5*(a.T + a)
    tr = (1.0/3.0)*cp.trace(a)
    # print('type(tr)', type(tr))
    xx = a[0,0]
    # print('type(xx)', type(xx))
    yy = a[1,1]
    zz = a[2,2]

    xy = a[0,1]
    xz = a[0,2]
    yz = a[1,2]

    ssum = xx**2 + yy**2 + zz**2 + 2*(xy**2 + xz**2 + yz**2)
    anis = (1.5 * abs(ssum - 3*tr**2))**0.5
    return float(tr), float(anis)

def utriangle_symmetrize(A):
    upper = cp.triu_indices(n=A.shape[0], k=1)
    lower = (upper[1], upper[0])
    A[lower] = A[upper]
    return A

def anti_block_symmetrize(A,m,n):
    A[m:n,:m] = -A[:m,m:n].T
    return A



def gen_VW(sub_A_holder, V_holder, W_holder, size_old, size_new, symmetry=False):
    '''

    [ V_old ] [W_old.T, W_new.T] = [VW_old,        V_old W_new.T] = [VW_old,   V_old W_new.T  ]
    [ V_new ]                      [V_new W_old.T, V_new W_new.T]   [  V_new  W_current.T     ]

    symmetry: whether sub_A is symmatric 

                        V_holder (W_holder is same set up)

            |------------------------------------------------|
            |      V_old                                     |     
size_old    |------------------------------------------------|  [ V_current ]
            |      V_new                                     |
size_new    |------------------------------------------------|
            |                                                |
            |      empty                                     |
            |                                                |
            |------------------------------------------------|




    sub_A_holder

                            size_old            size_new
                |---------------|-----------------｜-----------------｜
                |               |                 ｜                 ｜
                |               |                 ｜                 ｜
                | V_old W_old.T |  V_old W_new.T  ｜                 ｜
                |  (=sub_A_old) |                 ｜                 ｜
                |               |                 ｜                 ｜
      size_old  |---------------(V_currentW_new.T)｜-----------------｜
                | if symmetry:  |                 ｜                 ｜
                | V_new W_old.T |  V_new W_new.T  ｜                 ｜
                |               |                 ｜                 ｜
                |               |                 ｜                 ｜
                |               |                 ｜                 ｜
      size_new  |---------------------------------｜-----------------｜
                |               |                 ｜                 ｜
                |               |                 ｜                 ｜
                |               |                 ｜                 ｜
                |               |                 ｜                 ｜
                |               |                 ｜                 ｜
                |               |                 ｜                 ｜
                |---------------|-----------------｜-----------------｜
    '''


    V_current = V_holder[:size_new,:]
    W_new = W_holder[size_old:size_new, :]

    sub_A_holder[:size_new, size_old:size_new] = cp.dot(V_current, W_new.T)

    if symmetry:
        # pass
        sub_A_holder[size_old:size_new, :size_old] = sub_A_holder[:size_old, size_old:size_new].T
    else:
        sub_A_holder[size_old:size_new, :size_old] = cp.dot(V_holder[size_old:size_new,:], W_holder[:size_old,:].T)

    return sub_A_holder



def gen_VP(sub_P_holder, V_holder, P, size_old, size_new):
    '''
    [ V_old.T ] [P] = [P_old    ]
    [ V_new.T ]       [V_new.T P]
    '''
    V_new = V_holder[:,size_old:size_new]
    sub_P_holder[size_old:size_new,:] = cp.dot(V_new.T, P)
    return sub_P_holder


def gen_sub_pq(V_holder, W_holder, P, Q, VP_holder, WQ_holder, WP_holder, VQ_holder, size_old, size_new):
    '''
    [ V_old.T ] [P_old] = [P_old]
    [ V_new.T ]           [V_new.T [P_old]]
    '''
    VP_holder = gen_VP(VP_holder, V_holder, P, size_old, size_new)

    VQ_holder = gen_VP(VQ_holder, V_holder, Q, size_old, size_new)

    WP_holder = gen_VP(WP_holder, W_holder, P, size_old, size_new)

    WQ_holder = gen_VP(WQ_holder, W_holder, Q, size_old, size_new)

    p = VP_holder[:size_new,:] + WQ_holder[:size_new,:]
    q = WP_holder[:size_new,:] + VQ_holder[:size_new,:]

    return p, q, VP_holder, WQ_holder, WP_holder, VQ_holder


def gen_sub_ab(V_holder, W_holder, U1_holder, U2_holder,
              VU1_holder, WU2_holder, VU2_holder, WU1_holder,
              VV_holder, WW_holder, VW_holder,
              size_old, size_new):
    '''
    a = V U1.T + W U2.T
    b = V U2.T + W U1.T
    sigma = V V.T - W W.T
    pi = V W.T - V W.T.T

    
    '''

    VU1_holder = gen_VW(VU1_holder, V_holder, U1_holder, size_old, size_new, symmetry=False)
    VU2_holder = gen_VW(VU2_holder, V_holder, U2_holder, size_old, size_new, symmetry=False)
    WU1_holder = gen_VW(WU1_holder, W_holder, U1_holder, size_old, size_new, symmetry=False)
    WU2_holder = gen_VW(WU2_holder, W_holder, U2_holder, size_old, size_new, symmetry=False)

    VV_holder = gen_VW(VV_holder, V_holder, V_holder, size_old, size_new, symmetry=False)
    WW_holder = gen_VW(WW_holder, W_holder, W_holder, size_old, size_new, symmetry=False)
    VW_holder = gen_VW(VW_holder, V_holder, W_holder, size_old, size_new, symmetry=False)

    sub_A = VU1_holder[:size_new, :size_new] + WU2_holder[:size_new, :size_new]
    sub_A = utriangle_symmetrize(sub_A)

    sub_B = VU2_holder[:size_new, :size_new] + WU1_holder[:size_new, :size_new]
    sub_B = utriangle_symmetrize(sub_B)

    sigma = VV_holder[:size_new, :size_new] - WW_holder[:size_new, :size_new]
    sigma = utriangle_symmetrize(sigma)

    pi = VW_holder[:size_new, :size_new] - VW_holder[:size_new, :size_new].T

    return sub_A, sub_B, sigma, pi, VU1_holder, WU2_holder, VU2_holder, WU1_holder, VV_holder, WW_holder, VW_holder


def Gram_Schmidt_fill_holder(V, count, vecs, double = False):
    '''V is a vectors holder
       count is the amount of vectors that already sit in the holder
       nvec is amount of new vectors intended to fill in the V
       count will be final amount of vectors in V
    '''
    nvec = cp.shape(vecs)[0]
    for j in range(nvec):
        vec = vecs[j, :].reshape(1,-1)
        vec = Gram_Schmidt_bvec(V[:count, :], vec)   #single orthonormalize
        if double == True:
            vec = Gram_Schmidt_bvec(V[:count, :], vec)   #double orthonormalize
        norm = cp.linalg.norm(vec)
        if  norm > 1e-14:
            vec = vec/norm
            V[count,:] = vec[0,:]
            count += 1
        # else:
        #     print('zero vector')
    new_count = count
    return V, new_count

def nKs_fill_holder(V, count, vecs, double = True):
    '''V is a vectors holder
       count is the amount of vectors that already sit in the holder
       nvec is amount of new vectors intended to fill in the V
       count will be final amount of vectors in V
    '''
    nvec = cp.shape(vecs)[0]
    for j in range(nvec):
        vec = vecs[j, :].reshape(1,-1)
        norm = cp.linalg.norm(vec)
        if  norm > 1e-14:
            vec = vec/norm
            V[count,:] = vec[0,:]
            count += 1
    new_count = count
    return V, new_count

def S_symmetry_orthogonal(x,y):
    '''symmetrically orthogonalize the vectors |x,y> and |y,x>
       as close to original vectors as possible
    '''
    # print('x y shape', x.shape, y.shape)
    x_p_y = x + y
    x_p_y_norm = cp.linalg.norm(x_p_y)
    # print('x_p_y_norm', x_p_y_norm)

    x_m_y = x - y
    x_m_y_norm = cp.linalg.norm(x_m_y)

    a = x_p_y_norm/x_m_y_norm

    x_p_y = x_p_y/2
    x_m_y = x_m_y * a/2

    new_x = x_p_y + x_m_y
    new_y = x_p_y - x_m_y

    # print('check norm')
    # xy = cp.hstack((new_x,new_y))
    # yx = cp.hstack((new_y,new_x))
    # print(cp.dot(xy, yx.T))

    return new_x, new_y

def symmetrize(A):
    A = (A + A.T)/2
    return A

def anti_symmetrize(A):
    A = (A - A.T)/2
    return A

def check_orthonormal(A):
    '''
    define the orthonormality of a matrix A as the norm of (A.T*A - I)
    '''
    n = cp.shape(A)[1]
    B = cp.dot(A.T, A)
    c = cp.linalg.norm(B - cp.eye(n))
    return c

def check_symmetry(A):
    '''
    define matrix A is symmetric
    '''
    a = cp.linalg.norm(A - A.T)
    return a

def check_anti_symmetry(A):
    '''
    define matrix A is symmetric
    '''
    a = cp.linalg.norm(A + A.T)
    return a

def VW_Gram_Schmidt_fill_holder(V_holder, W_holder, m, X_new, Y_new, double=False):
    '''
    put X_new into V, and Y_new into W
    m: the amount of vectors that already on V or W
    nvec: amount of new vectors intended to put in the V and W
    '''
    VWGSstart = time.time()
    nvec = X_new.shape[0]


    for j in range(0, nvec):
        V = V_holder[:m,:]
        W = W_holder[:m,:]

        x_tmp = X_new[j,:].reshape(1,-1)
        y_tmp = Y_new[j,:].reshape(1,-1)

        x_tmp,y_tmp = VW_Gram_Schmidt(x_tmp, y_tmp, V, W)
        if double == True:
            # print('double')
            x_tmp,y_tmp = VW_Gram_Schmidt(x_tmp, y_tmp, V, W)

        x_tmp,y_tmp = S_symmetry_orthogonal(x_tmp,y_tmp)

        xy_norm = (cp.dot(x_tmp, x_tmp.T) + cp.dot(y_tmp, y_tmp.T))**0.5


        if  xy_norm > 1e-14:
            x_tmp = x_tmp/xy_norm
            y_tmp = y_tmp/xy_norm

            V_holder[m,:] = x_tmp[0,:]
            W_holder[m,:] = y_tmp[0,:]
            m += 1
        else:
            print('vector kicked out during GS orthonormalization')
    new_m = m

    return V_holder, W_holder, new_m


def VW_nKs_fill_holder(V_holder, W_holder, m, X_new, Y_new, double=False):
    nvec = X_new.shape[0]
    for j in range(0, nvec):
        x_tmp = X_new[j,:].reshape(1,-1)
        y_tmp = Y_new[j,:].reshape(1,-1)
        x_tmp,y_tmp = S_symmetry_orthogonal(x_tmp,y_tmp)
        xy_norm = (cp.dot(x_tmp, x_tmp.T) + cp.dot(y_tmp, y_tmp.T))**0.5
        x_tmp = x_tmp/xy_norm
        y_tmp = y_tmp/xy_norm

        if  xy_norm > 1e-18:
            x_tmp = x_tmp/xy_norm
            y_tmp = y_tmp/xy_norm

            V_holder[m,:] = x_tmp[0,:]
            W_holder[m,:] = y_tmp[0,:]
            m += 1
        else:
            print('vector kicked out during GS orthonormalization')

    new_m = m   
    return V_holder, W_holder, new_m


def solve_AX_Xla_B(A, omega, Q):
    '''AX - XΩ  = Q
       A, Ω, Q are known, solve X
    '''
    Qnorm = cp.linalg.norm(Q, axis=0, keepdims = True)
    Q = Q/Qnorm
    N_vectors = len(omega)
    a, u = cp.linalg.eigh(A)
    # print('a =',a)
    # print('omega =',omega)
    ub = cp.dot(u.T, Q)
    ux = cp.zeros_like(Q)
    for k in range(N_vectors):
        ux[:, k] = ub[:, k]/(a - omega[k])
    X = cp.dot(u, ux)
    X *= Qnorm

    return X

def TDDFT_subspace_eigen_solver2(a, b, sigma, pi, k):
    ''' [ a b ] x - [ σ   π] x  Ω = 0 '''
    ''' [ b a ] y   [-π  -σ] y    = 0 '''

    d = abs(cp.diag(sigma))
    d_mh = d**(-0.5)

    s_m_p = d_mh.reshape(-1,1) * (sigma - pi) * d_mh.reshape(1,-1)

    # '''LU = d^−1/2 (σ − π) d^−1/2'''
    # ''' A = PLU '''
    # ''' if A is diagonally dominant, P is identity matrix (in fact not always) '''
    # P_permutation, L, U = scipy.linalg.lu(s_m_p)

    # L = cp.dot(P_permutation, L)

    # L_inv = cp.linalg.inv(L)
    # U_inv = cp.linalg.inv(U)

    L_inv = cp.linalg.cholesky(cp.linalg.inv(s_m_p))
    U_inv = L_inv.T
    ''' a ̃−b ̃= U^-T d^−1/2 (a−b) d^-1/2 U^-1 = GG^T '''
    dambd =  d_mh.reshape(-1,1)*(a-b)*d_mh.reshape(1,-1)
    GGT = cp.linalg.multi_dot([U_inv.T, dambd, U_inv])

    G = scipy.linalg.cholesky(GGT, lower=True)
    G_inv = cp.linalg.inv(G)

    ''' M = G^T L^−1 d^−1/2 (a+b) d^−1/2 L^−T G '''
    dapbd = d_mh.reshape(-1,1)*(a+b)*d_mh.reshape(1,-1)
    M = cp.linalg.multi_dot([G.T, L_inv, dapbd, L_inv.T, G])

    omega2, Z = cp.linalg.eigh(M)
    omega = (omega2**0.5)[:k]
    Z = Z[:,:k]

    ''' It requires Z^T Z = 1/Ω '''
    ''' x+y = d^−1/2 L^−T GZ Ω^-0.5 '''
    ''' x−y = d^−1/2 U^−1 G^−T Z Ω^0.5 '''

    x_p_y = d_mh.reshape(-1,1)\
            *cp.linalg.multi_dot([L_inv.T, G, Z])\
            *(cp.array(omega)**-0.5).reshape(1,-1)

    x_m_y = d_mh.reshape(-1,1)\
            *cp.linalg.multi_dot([U_inv, G_inv.T, Z])\
            *(cp.array(omega)**0.5).reshape(1,-1)

    x = (x_p_y + x_m_y)/2
    y = x_p_y - x

    return omega, x, y

def TDDFT_subspace_eigen_solver3(a, b, sigma, pi, k):
    ''' [ a b ] x - [ σ   π] x  Ω = 0 
        [ b a ] y   [-π  -σ] y    = 0
        AT=BTΩ
        B^-1/2 A B^-1/2 B^1/2 T = B^1/2 T Ω
        MZ = Z Ω
        M = B^-1/2 A B^-1/2
        Z = B^1/2 T
    '''
    half_size = a.shape[0]
    A = cp.zeros((2*half_size,2*half_size))
    print('A size =', A.shape)
    A[:half_size,:half_size] = a[:,:]
    A[:half_size,half_size:] = b[:,:]
    A[half_size:,:half_size] = b[:,:]
    A[half_size:,half_size:] = a[:,:]  

    B = cp.zeros_like(A)
    B[:half_size,:half_size] = sigma[:,:]
    B[:half_size,half_size:] = pi[:,:]
    B[half_size:,:half_size] = -pi[:,:]  
    B[half_size:,half_size:] = -sigma[:,:]
    print(B)
    #B^-1/2
    B_neg_tmp = matrix_power(B, -0.5)
    M = cp.linalg.multi_dot([B_neg_tmp, A, B_neg_tmp])
    omega, Z = cp.linalg.eigh(M)
    print('omega =', omega)
    omega = omega[half_size:k]
    Z = Z[:, half_size:k]
    print('omega =', omega)

    T = cp.dot(B_neg_tmp, Z)
    x = T[:half_size,:]
    y = T[half_size:,:]

    return omega, x, y

def TDDFT_subspace_eigen_solver(a, b, sigma, pi, k):
    ''' [ a b ] x - [ σ   π] x  Ω = 0 
        [ b a ] y   [-π  -σ] y    = 0
        AT=BTΩ
        A^1/2 T = A^-1/2 B A^-1/2 A^1/2 T Ω
        MZ = Z 1/Ω
        M = A^-1/2 B A^-1/2 A^1/2
        Z = A^1/2 T 
        Z is always returned as normlized vectors, which are not what we wanted
        because Z^T Z = [x]^T A^1/2 A^1/2 [x] = [x]^T [ a b ] [x] =  [x]^T [ σ   π] x Ω = Ω
                        [y]               [y]   [y]   [ b a ] [y]    [y]   [-π  -σ] y
        therefore Z=Z*(Ω**0.5)
        k: N_states
    '''
    half_size = a.shape[0]
    A = cp.zeros((2*half_size,2*half_size))
    # print('A size =', A.shape)
    A[:half_size,:half_size] = a[:,:]
    A[:half_size,half_size:] = b[:,:]
    A[half_size:,:half_size] = b[:,:]
    A[half_size:,half_size:] = a[:,:]  
    # print('check_symmetry(A)', check_symmetry(A))
    B = cp.zeros_like(A)
    B[:half_size,:half_size] = sigma[:,:]
    B[:half_size,half_size:] = pi[:,:]
    B[half_size:,:half_size] = -pi[:,:]
    B[half_size:,half_size:] = -sigma[:,:]  
    # print('check_symmetry(B)', check_symmetry(B))
    # print(B)
    #A^-1/2
    A_neg_tmp = matrix_power(A, -0.5)
    # M = cp.linalg.multi_dot([A_neg_tmp, B, A_neg_tmp])
    M = cp.dot(A_neg_tmp, B)
    M = cp.dot(M,A_neg_tmp )

    # print('check_symmetry(M)', check_symmetry(M))
    omega, Z = cp.linalg.eigh(M)
    
    # print('type(omega) ', type(omega))

    omega = 1/omega[-k:][::-1]
    Z = Z[:, -k:][:, ::-1]
    Z = Z*(omega**0.5)

    # print('omega =', omega)

    T = cp.dot(A_neg_tmp, Z)
    x = T[:half_size,:]
    y = T[half_size:,:]

    # xy_norm_check = cp.linalg.norm( (cp.dot(x.T,x) - cp.dot(y.T,y)) -cp.eye(k) )
    # print('check norm of X^TX - Y^YY - I = {:.2e}'.format(xy_norm_check)) 
    
    return omega, x, y

def XmY_2_XY(Z, AmB_sq, omega):
    '''given Z, (A-B)^2, omega
       return X, Y

        X-Y = (A-B)^-1/2 Z
        X+Y = (A-B)^1/2 Z omega^-1 
    '''
    AmB_sq = AmB_sq.reshape(1,-1)

    '''AmB = (A - B)'''
    AmB = AmB_sq**0.5

    XmY = AmB**(-0.5) * Z

    omega = omega.reshape(-1,1)
    XpY = (AmB * XmY)/omega

    X = (XpY + XmY)/2
    Y = (XpY - XmY)/2

    return X, Y

def gen_VW_f_order(sub_A_holder, V_holder, W_holder, size_old, size_new, symmetry = True, up_triangle = False):
    '''
    [ V_old.T ] [W_old, W_new] = [VW_old,        V_old.T W_new] = [VW_old, V_current.T W_new]
    [ V_new.T ]                  [V_new.T W_old, V_new.T W_new]   [               '' ''     ]


    V_holder or W_holder

                size_old     size_new
    |--------------|--------------|------------|
    |              |              |            |
    |   V_old      |    V_new     |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |        [ V_current ]        |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |              |              |            |
    |--------------|--------------|------------|

    sub_A_holder

                            size_old            size_new
                |---------------|-----------------｜-----------------｜
                |               |                 ｜                 ｜
                |    VW_old     |                 ｜                 ｜
                |               |                 ｜                 ｜
      size_old  |---------------|V_current.T W_new｜-----------------｜
                | V_new.T W_old |                 ｜                 ｜
                | or            |                 ｜                 ｜
                | W_new.T V_old |                 ｜                 ｜
      size_new  |---------------|-----------------｜-----------------｜
                |               |                 ｜                 ｜
                |               |                 ｜                 ｜
                |               |                 ｜                 ｜
                |---------------|-----------------｜-----------------｜
    '''

    V_current = V_holder[:,:size_new]
    W_new = W_holder[:,size_old:size_new]
    sub_A_holder[:size_new,size_old:size_new] = cp.dot(V_current.T, W_new)

    if symmetry == True:
        sub_A_holder = block_symmetrize(sub_A_holder,size_old,size_new)
    elif symmetry == False:
        if up_triangle == False:
            '''
            up_triangle == False means also explicitly compute the lower triangle,
                                        either equal upper triangle.T or recompute
            '''
            V_new = V_holder[:,size_old:size_new]
            W_old = W_holder[:,:size_old]
            sub_A_holder[size_old:size_new,:size_old] = cp.dot(V_new.T, W_old)
        elif up_triangle == True:
            '''
            otherwise juts let the lower triangle be zeros
            '''
            pass

    return sub_A_holder

# def show_memory_info(hint):
#     pid = os.getpid()
#     p = psutil.Process(pid)
#     info = p.memory_full_info()
#     memory = info.uss / 1024**3
#     print('{:>50} memory used: {:<.2f} GB'.format(hint, memory))