# Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import cupy as cp
from pyscf.fci import direct_spin1

TILE = 32

code = r'''
#define TILE 32
extern "C" {
__global__
void _build_t1(double *ci0, double *t1,
    long long strb0, long long na, long long nb, long long nnorb,
    unsigned short *addra, unsigned short *addrb, char *signa, char *signb)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int stra0 = blockIdx.y * blockDim.y;
    int strb = strb0 + tx;
    int stra = stra0 + ty;

    int nab = na * TILE;
    int ab_id = stra * TILE + tx;
    __shared__ unsigned short _addra[TILE*TILE];
    __shared__ unsigned short _addrb[TILE*TILE];
    __shared__ char _signa[TILE*TILE];
    __shared__ char _signb[TILE*TILE];
    int sign, str1, j0, j;
    int dj = TILE;
    double val;

    for (j0 = 0; j0 < nnorb; j0+=TILE) {
        _addra[ty*TILE+tx] = addra[(j0+ty)*na+stra0+tx];
        _addrb[ty*TILE+tx] = addrb[(j0+ty)*nb+strb0+tx];
        _signa[ty*TILE+tx] = signa[(j0+ty)*na+stra0+tx];
        _signb[ty*TILE+tx] = signb[(j0+ty)*nb+strb0+tx];
        if (j0 + TILE > nnorb) {
            dj = nnorb - j0;
        }
        __syncthreads();
        if (stra < na && strb < nb) {
            for (j = 0; j < dj; j++) {
                val = 0;
                sign = _signa[j*TILE+ty];
                str1 = _addra[j*TILE+ty];
                if (sign != 0) {
                    val = sign * ci0[str1*nb+strb];
                }

                sign = _signb[j*TILE+tx];
                str1 = _addrb[j*TILE+tx];
                if (sign != 0) {
                    val += sign * ci0[stra*nb+str1];
                }
                t1[(j0+j)*nab + ab_id] = val;
            }
        }
        __syncthreads();
    }
}

__global__
void _gather(double *out, double *t1,
    long long strb0, long long na, long long nb, long long nnorb,
    unsigned short *addra, unsigned short *addrb, char *signa, char *signb)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int stra0 = blockIdx.y * blockDim.y;
    int strb = strb0 + tx;
    int stra = stra0 + ty;
    int nab = na * TILE;
    int ab_id = stra * TILE + tx;
    __shared__ unsigned short _addra[TILE*TILE];
    __shared__ unsigned short _addrb[TILE*TILE];
    __shared__ char _signa[TILE*TILE];
    __shared__ char _signb[TILE*TILE];
    int sign, str1, j0, j;
    int dj = TILE;
    double val = 0.;

    for (j0 = 0; j0 < nnorb; j0+=TILE) {
        _addra[ty*TILE+tx] = addra[(j0+ty)*na+stra0+tx];
        _addrb[ty*TILE+tx] = addrb[(j0+ty)*nb+strb0+tx];
        _signa[ty*TILE+tx] = signa[(j0+ty)*na+stra0+tx];
        _signb[ty*TILE+tx] = signb[(j0+ty)*nb+strb0+tx];
        if (j0 + TILE > nnorb) {
            dj = nnorb - j0;
        }
        __syncthreads();
        if (stra < na && strb < nb) {
            for (j = 0; j < dj; j++) {
                sign = _signa[j*TILE+ty];
                str1 = _addra[j*TILE+ty];
                if (sign != 0) {
                    val += sign * t1[(j0+j)*nab + (str1*TILE+tx)];
                }

                sign = _signb[j*TILE+tx];
                str1 = _addrb[j*TILE+tx];
                if (sign != 0) {
                    out[stra*nb+str1] += sign * t1[(j0+j)*nab + ab_id];
                }
            }
        }
        __syncthreads();
    }
    out[stra*nb+strb] += val;
}
}'''

_contract_2e_spin1 = cp.RawModule(code=code)
_build_t1 = _contract_2e_spin1.get_function('_build_t1')
_gather = _contract_2e_spin1.get_function('_gather')

def _link_index_to_addrs(link_index, nnorb):
    na, nov = link_index.shape[:2]
    ia = link_index[:,:,0].T
    addr = np.zeros((nnorb, na), dtype=np.uint16)
    sign = np.zeros((nnorb, na), dtype=np.int8)
    #:for j in range(nov):
    #:    for a in range(na):
    #:        addr[ia[a,j],a] = link_index[a,j,2]
    #:        sign[ia[a,j],a] = link_index[a,j,3]
    idx = np.arange(na)
    addr[ia,idx] = link_index[:,:,2].T
    sign[ia,idx] = link_index[:,:,3].T
    # Add paddings to avoid illegal address in kernel
    _addr = cp.empty((nnorb+TILE-1, na), dtype=np.uint16)[:nnorb]
    _sign = cp.empty((nnorb+TILE-1, na), dtype=np.int8)[:nnorb]
    _addr.set(addr)
    _sign.set(sign)
    return _addr, _sign

def contract_2e(eri, ci0, norb, nelec, link_index):
    ci0 = cp.asarray(ci0)
    out = cp.zeros_like(ci0)
    na, nb = ci0.shape
    nnorb = norb * (norb + 1) // 2
    assert eri.shape == (nnorb, nnorb)
    eri = cp.asarray(eri)
    link_indexa, link_indexb = link_index
    addra, signa = _link_index_to_addrs(link_indexa, nnorb)
    if link_indexa is link_indexb:
        addrb, signb = addra, signa
    else:
        addrb, signb = _link_index_to_addrs(link_indexb, nnorb)

    threads = (TILE, TILE)
    blocks = (1, (na+TILE-1)//TILE)
    rest_args = (na, nb, nnorb, addra, addrb, signa, signb)
    t1 = cp.empty((nnorb, na*TILE))
    gt1 = cp.empty((nnorb, na*TILE))
    if 0:
        # Pipeline the three operations. Seems no performance improvement
        buf = cp.empty((nnorb, na*TILE))
        sm_t1 = cp.cuda.Stream(non_blocking=True)
        sm_gt1 = cp.cuda.Stream(non_blocking=True)
        for strb0 in range(0, nb, TILE):
            _build_t1(blocks, threads, (ci0, t1, strb0) + rest_args, stream=sm_t1)
            sm_t1.synchronize()
            with sm_gt1:
                eri.dot(t1, out=gt1)
            sm_gt1.synchronize()
            _gather(blocks, threads, (out, gt1, strb0) + rest_args)
            t1, gt1, buf = buf, t1, gt1
    else:
        for strb0 in range(0, nb, TILE):
            _build_t1(blocks, threads, (ci0, t1, strb0) + rest_args)
            eri.dot(t1, out=gt1)
            _gather(blocks, threads, (out, gt1, strb0) + rest_args)
    return out.get()

class FCI(direct_spin1.FCI):
    from gpu4pyscf.lib.utils import to_cpu, to_gpu, device

    contract_2e = staticmethod(contract_2e)
