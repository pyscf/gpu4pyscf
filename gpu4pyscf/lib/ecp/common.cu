/*
 * Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
template <int l> __device__ __host__
Cartesian<(l+1)*(l+2)/2> ang_nuc_l(double rx, double ry, double rz){
    double rxPow[l+1], ryPow[l+1], rzPow[l+1];
    rxPow[0] = ryPow[0] = rzPow[0] = 1.0;
    for (int i = 1; i <= l; i++) {
        rxPow[i] = rxPow[i - 1] * rx;
        ryPow[i] = ryPow[i - 1] * ry;
        rzPow[i] = rzPow[i - 1] * rz;
    }

    double g[(l+1)*(l+2)/2];
    int index = 0;
    for (int i = l; i >= 0; i--) {
        for (int j = l - i; j >= 0; j--) {
            int k = l - i - j;
            g[index++] = rxPow[i] * ryPow[j] * rzPow[k];
        }
    }

    double c[2*l+1];
    cart2sph(c, l, g);
    Cartesian<(l+1)*(l+2)/2> omega;
    sph2cart(omega.data, l, c);
    return omega;
}
*/

__device__
double rad_part(const int ish, const int *ecpbas, const double *env){
    const int npk = ecpbas[ish*BAS_SLOTS+NPRIM_OF];
    const int r_order = ecpbas[ish*BAS_SLOTS+RADI_POWER];
    const int exp_ptr = ecpbas[ish*BAS_SLOTS+PTR_EXP];
    const int coeff_ptr = ecpbas[ish*BAS_SLOTS+PTR_COEFF];

    double u1 = 0.0;
    double r = 0.0;
    if (threadIdx.x < NGAUSS){
        r = r128[threadIdx.x];
    }
    for (int kp = 0; kp < npk; kp++){
        const double ak = env[exp_ptr+kp];
        const double ck = env[coeff_ptr+kp];
        u1 += ck * exp(-ak * r * r);
    }
    double w = 0.0;
    if (threadIdx.x < NGAUSS){
        w = w128[threadIdx.x];
    }
    return u1 * pow(r, r_order) * w;
}

__device__
void cache_fac(double *fx, int LI, double *ri){
    const int LI1 = LI + 1;
    double xx[AO_LMAX_IP+1], yy[AO_LMAX_IP+1], zz[AO_LMAX_IP+1];
    xx[0] = 1; yy[0] = 1; zz[0] = 1;
    for (int i = 1; i <= LI; i++){
        xx[i] = xx[i-1] * ri[0];
        yy[i] = yy[i-1] * ri[1];
        zz[i] = zz[i-1] * ri[2];
    }

    const int nfi = (LI1+1)*LI1/2;
    for (int i = 0; i <= LI; i++){
        const int xoffset = i*(i+1)/2;
        const int yoffset = xoffset + nfi;
        const int zoffset = yoffset + nfi;
        for (int j = 0; j <= i; j++){
            const double bfac = _binom[xoffset+j]; // binom(i,j)
            fx[xoffset+j] = bfac * xx[i-j];
            fx[yoffset+j] = bfac * yy[i-j];
            fx[zoffset+j] = bfac * zz[i-j];
        }
    }
}

template <int LI> __device__
void cache_fac(double *fx, double *ri){
    constexpr int LI1 = LI + 1;
    double xx[LI1], yy[LI1], zz[LI1];
    xx[0] = 1; yy[0] = 1; zz[0] = 1;
    for (int i = 1; i <= LI; i++){
        xx[i] = xx[i-1] * ri[0];
        yy[i] = yy[i-1] * ri[1];
        zz[i] = zz[i-1] * ri[2];
    }

    constexpr int nfi = (LI1+1)*LI1/2;
    for (int i = 0; i <= LI; i++){
        const int xoffset = i*(i+1)/2;
        const int yoffset = xoffset + nfi;
        const int zoffset = yoffset + nfi;
        for (int j = 0; j <= i; j++){
            const double bfac = _binom[xoffset+j]; // binom(i,j)
            fx[xoffset+j] = bfac * xx[i-j];
            fx[yoffset+j] = bfac * yy[i-j];
            fx[zoffset+j] = bfac * zz[i-j];
        }
    }
}

__device__
void block_reduce(double val, double *d_out) {
    __shared__ double sdata[THREADS];
    unsigned int tid = threadIdx.x;

    sdata[tid] = val;
    __syncthreads();

    // Perform reduction in shared memory.
    // Reduce the data until 32 threads remain.
    for (unsigned int s = THREADS / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Unroll the final warp (32 threads) without __syncthreads().
    if (tid < 32) {
        // Use a volatile pointer to ensure memory loads/stores are not optimized away.
        volatile double *vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // The first thread writes the block's final result to global memory.
    if (tid == 0) {
        d_out[0] += sdata[0];
    }
    __syncthreads();
}

__device__ __forceinline__
void set_shared_memory(double *smem, const int size) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        smem[i] = 0.0;
    }
    __syncthreads();
}

__device__
void _li_up(double *out, double *buf, const int li, const int lj){
    const int nfj = (lj+1) * (lj+2) / 2;
    const int nfi = (li+1) * (li+2) / 2;
    const int nfi0 = li * (li+1) / 2;
    double *outx = out;
    double *outy = outx + nfi*nfj;
    double *outz = outy + nfi*nfj;
    const double fac = 1.0 / _ecp_fac[li-1];
    for (int ij = threadIdx.x; ij < nfi0*nfj; ij+=blockDim.x){
        const int i = ij % nfi0;
        const int j = ij / nfi0;
        const double yfac = fac * (_cart_pow_y[i] + 1);
        const double zfac = fac * (_cart_pow_z[i] + 1);
        const double xfac = fac * (li-1 - _cart_pow_y[i] - _cart_pow_z[i] + 1);

        atomicAdd(outx + j*nfi +          i, xfac * buf[j*nfi0 + i]);
        atomicAdd(outy + j*nfi + _y_addr[i], yfac * buf[j*nfi0 + i]);
        atomicAdd(outz + j*nfi + _z_addr[i], zfac * buf[j*nfi0 + i]);
    }
}

__device__
void _li_up_and_write(double *out, double *buf, const int li, const int lj, const int nao){
    const int nfi0 = li * (li+1) / 2;
    const int nfj = (lj+1) * (lj+2) / 2;
    double *outxx = out ;
    double *outxy = out + nao*nao;
    double *outxz = out + 2*nao*nao;
    double *outyx = out + 3*nao*nao;
    double *outyy = out + 4*nao*nao;
    double *outyz = out + 5*nao*nao;
    double *outzx = out + 6*nao*nao;
    double *outzy = out + 7*nao*nao;
    double *outzz = out + 8*nao*nao;
    const double fac = 1.0 / _ecp_fac[li-1];
    for (int ij = threadIdx.x; ij < nfi0*nfj; ij+=blockDim.x){
        const int i = ij % nfi0;
        const int j = ij / nfi0;
        const double yfac = fac * (_cart_pow_y[i] + 1);
        const double zfac = fac * (_cart_pow_z[i] + 1);
        const double xfac = fac * (li-1 - _cart_pow_y[i] - _cart_pow_z[i] + 1);

        const int i_addr[3] = {i, _y_addr[i], _z_addr[i]};
        atomicAdd(outxx + j + i_addr[0]*nao, xfac * buf[j*nfi0 + i]);
        atomicAdd(outxy + j + i_addr[1]*nao, yfac * buf[j*nfi0 + i]);
        atomicAdd(outxz + j + i_addr[2]*nao, zfac * buf[j*nfi0 + i]);

        atomicAdd(outyx + j + i_addr[0]*nao, xfac * buf[j*nfi0 + i + nfi0*nfj]);
        atomicAdd(outyy + j + i_addr[1]*nao, yfac * buf[j*nfi0 + i + nfi0*nfj]);
        atomicAdd(outyz + j + i_addr[2]*nao, zfac * buf[j*nfi0 + i + nfi0*nfj]);

        atomicAdd(outzx + j + i_addr[0]*nao, xfac * buf[j*nfi0 + i + 2*nfi0*nfj]);
        atomicAdd(outzy + j + i_addr[1]*nao, yfac * buf[j*nfi0 + i + 2*nfi0*nfj]);
        atomicAdd(outzz + j + i_addr[2]*nao, zfac * buf[j*nfi0 + i + 2*nfi0*nfj]);
    }
}


__device__
void _li_down(double *out, double *buf, const int li, const int lj){
    const int nfi = (li+1) * (li+2) / 2;
    const int nfj = (lj+1) * (lj+2) / 2;
    const int nfi1= (li+2) * (li+3) / 2;
    double *outx = out;
    double *outy = outx + nfi*nfj;
    double *outz = outy + nfi*nfj;
    const double fac = _ecp_fac[li];

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij % nfi;
        const int j = ij / nfi;
        atomicAdd(outx + j*nfi+i, fac * buf[j*nfi1+i]);
        atomicAdd(outy + j*nfi+i, fac * buf[j*nfi1+_y_addr[i]]);
        atomicAdd(outz + j*nfi+i, fac * buf[j*nfi1+_z_addr[i]]);
    }
}

__device__
void _li_down_and_write(double *out, double *buf, const int li, const int lj, const int nao){
    const int nfi = (li+1) * (li+2) / 2;
    const int nfj = (lj+1) * (lj+2) / 2;
    const int nfi1= (li+2) * (li+3) / 2;
    double *outxx = out ;
    double *outxy = out + nao*nao;
    double *outxz = out + 2*nao*nao;
    double *outyx = out + 3*nao*nao;
    double *outyy = out + 4*nao*nao;
    double *outyz = out + 5*nao*nao;
    double *outzx = out + 6*nao*nao;
    double *outzy = out + 7*nao*nao;
    double *outzz = out + 8*nao*nao;
    const double fac = _ecp_fac[li];

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij % nfi;
        const int j = ij / nfi;
        const int i_addr[3] = {i, _y_addr[i], _z_addr[i]};

        atomicAdd(outxx + j + i*nao, fac * buf[j*nfi1 + i_addr[0]]);
        atomicAdd(outxy + j + i*nao, fac * buf[j*nfi1 + i_addr[1]]);
        atomicAdd(outxz + j + i*nao, fac * buf[j*nfi1 + i_addr[2]]);

        atomicAdd(outyx + j + i*nao, fac * buf[j*nfi1 + i_addr[0] + nfi1*nfj]);
        atomicAdd(outyy + j + i*nao, fac * buf[j*nfi1 + i_addr[1] + nfi1*nfj]);
        atomicAdd(outyz + j + i*nao, fac * buf[j*nfi1 + i_addr[2] + nfi1*nfj]);

        atomicAdd(outzx + j + i*nao, fac * buf[j*nfi1 + i_addr[0] + 2*nfi1*nfj]);
        atomicAdd(outzy + j + i*nao, fac * buf[j*nfi1 + i_addr[1] + 2*nfi1*nfj]);
        atomicAdd(outzz + j + i*nao, fac * buf[j*nfi1 + i_addr[2] + 2*nfi1*nfj]);
    }
}


__device__
void _lj_up_and_write(double *out, double *buf, const int li, const int lj, const int nao){
    const int nfi = (li+1)*(li+2)/2;
    const int nfj0 = lj * (lj+1) / 2;
    double *outxx = out;
    double *outxy = out + nao*nao;
    double *outxz = out + 2*nao*nao;
    double *outyx = out + 3*nao*nao;
    double *outyy = out + 4*nao*nao;
    double *outyz = out + 5*nao*nao;
    double *outzx = out + 6*nao*nao;
    double *outzy = out + 7*nao*nao;
    double *outzz = out + 8*nao*nao;
    const double fac = 1.0 / _ecp_fac[lj-1];
    for (int ij = threadIdx.x; ij < nfi*nfj0; ij+=blockDim.x){
        const int i = ij % nfi;
        const int j = ij / nfi;
        const double yfac = fac * (_cart_pow_y[j] + 1);
        const double zfac = fac * (_cart_pow_z[j] + 1);
        const double xfac = fac * (lj-1 - _cart_pow_y[j] - _cart_pow_z[j] + 1);
        const int j_addr[3] = {j, _y_addr[j], _z_addr[j]};

        atomicAdd(outxx + j_addr[0] + nao*i, xfac * buf[j*nfi + i]);
        atomicAdd(outxy + j_addr[1] + nao*i, yfac * buf[j*nfi + i]);
        atomicAdd(outxz + j_addr[2] + nao*i, zfac * buf[j*nfi + i]);

        atomicAdd(outyx + j_addr[0] + nao*i, xfac * buf[j*nfi + i + nfi*nfj0]);
        atomicAdd(outyy + j_addr[1] + nao*i, yfac * buf[j*nfi + i + nfi*nfj0]);
        atomicAdd(outyz + j_addr[2] + nao*i, zfac * buf[j*nfi + i + nfi*nfj0]);

        atomicAdd(outzx + j_addr[0] + nao*i, xfac * buf[j*nfi + i + 2*nfi*nfj0]);
        atomicAdd(outzy + j_addr[1] + nao*i, yfac * buf[j*nfi + i + 2*nfi*nfj0]);
        atomicAdd(outzz + j_addr[2] + nao*i, zfac * buf[j*nfi + i + 2*nfi*nfj0]);
    }
}

__device__
void _lj_down_and_write(double *out, double *buf, const int li, const int lj, const int nao){
    const int nfi = (li+1) * (li+2) / 2;
    const int nfj = (lj+1) * (lj+2) / 2;
    const int nfj1 = (lj+2) * (lj+3) / 2;
    double *outxx = out ;
    double *outxy = out + nao*nao;
    double *outxz = out + 2*nao*nao;
    double *outyx = out + 3*nao*nao;
    double *outyy = out + 4*nao*nao;
    double *outyz = out + 5*nao*nao;
    double *outzx = out + 6*nao*nao;
    double *outzy = out + 7*nao*nao;
    double *outzz = out + 8*nao*nao;
    const double fac = _ecp_fac[lj];
    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij % nfi;
        const int j = ij / nfi;
        const int j_addr[3] = {j, _y_addr[j], _z_addr[j]};

        atomicAdd(outxx + j + i*nao, fac * buf[j_addr[0]*nfi + i]);
        atomicAdd(outxy + j + i*nao, fac * buf[j_addr[1]*nfi + i]);
        atomicAdd(outxz + j + i*nao, fac * buf[j_addr[2]*nfi + i]);

        atomicAdd(outyx + j + i*nao, fac * buf[j_addr[0]*nfi + i + nfi*nfj1]);
        atomicAdd(outyy + j + i*nao, fac * buf[j_addr[1]*nfi + i + nfi*nfj1]);
        atomicAdd(outyz + j + i*nao, fac * buf[j_addr[2]*nfi + i + nfi*nfj1]);

        atomicAdd(outzx + j + i*nao, fac * buf[j_addr[0]*nfi + i + 2*nfi*nfj1]);
        atomicAdd(outzy + j + i*nao, fac * buf[j_addr[1]*nfi + i + 2*nfi*nfj1]);
        atomicAdd(outzz + j + i*nao, fac * buf[j_addr[2]*nfi + i + 2*nfi*nfj1]);
    }
}

/*
__device__
void _li_up_up(double *out, double *buf, const int li, const int lj){
    const int nfi = (li+1) * (li+2) / 2;
    const int nfj = (lj+1) * (lj+2) / 2;
    const int nfi0 = (li-1) * li / 2;
    double *outxx = out;
    double *outxy = out + nfi*nfj;
    double *outxz = out + 2*nfi*nfj;
    double *outyx = out + 3*nfi*nfj;
    double *outyy = out + 4*nfi*nfj;
    double *outyz = out + 5*nfi*nfj;
    double *outzx = out + 6*nfi*nfj;
    double *outzy = out + 7*nfi*nfj;
    double *outzz = out + 8*nfi*nfj;
    const double fac = 1.0 / _ecp_fac[li-1] / _ecp_fac[lj-2];
    for (int ij = threadIdx.x; ij < nfi0*nfj; ij+=blockDim.x){
        const int i = ij % nfi0;
        const int j = ij / nfi0;
        const double ypow1 = _cart_pow_y[i] + 1;
        const double zpow1 = _cart_pow_z[i] + 1;
        const double xpow1 = li-1 - _cart_pow_y[i] - _cart_pow_z[i] + 1;

        const double ypow2 = _cart_pow_y[i] + 2;
        const double zpow2 = _cart_pow_z[i] + 2;
        const double xpow2 = li-1 - _cart_pow_y[i] - _cart_pow_z[i] + 2;

        const int idx = i;
        const int idy = _y_addr[i];
        const int idz = _z_addr[i];

        outxx[j*nfi +          idx] += fac * xpow1 * xpow2 * buf[j*nfi0 + i];
        outxy[j*nfi + _y_addr[idx]] += fac * xpow2 * ypow2 * buf[j*nfi0 + i];
        outxz[j*nfi + _z_addr[idx]] += fac * xpow2 * zpow2 * buf[j*nfi0 + i];

        outyx[j*nfi +          idy] += fac * xpow2 * ypow2 * buf[j*nfi0 + i];
        outyy[j*nfi + _y_addr[idy]] += fac * ypow1 * ypow2 * buf[j*nfi0 + i];
        outyz[j*nfi + _z_addr[idy]] += fac * ypow2 * zpow2 * buf[j*nfi0 + i];

        outzx[j*nfi +          idz] += fac * xpow2 * zpow2 * buf[j*nfi0 + i];
        outzy[j*nfi + _y_addr[idz]] += fac * ypow2 * zpow2 * buf[j*nfi0 + i];
        outzz[j*nfi + _z_addr[idz]] += fac * zpow1 * zpow2 * buf[j*nfi0 + i];
    }
    __syncthreads();
}

__device__
void _li_up_down(double *out, double *buf, const int li, const int lj){
    const int nfi = (li+1) * (li+2) / 2;
    const int nfj = (lj+1) * (lj+2) / 2;
    double *outxx = out;
    double *outxy = out + nfi*nfj;
    double *outxz = out + 2*nfi*nfj;
    double *outyx = out + 3*nfi*nfj;
    double *outyy = out + 4*nfi*nfj;
    double *outyz = out + 5*nfi*nfj;
    double *outzx = out + 6*nfi*nfj;
    double *outzy = out + 7*nfi*nfj;
    double *outzz = out + 8*nfi*nfj;

    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij % nfi;
        const int j = ij / nfi;
        const double yfac = _cart_pow_y[i];
        const double zfac = _cart_pow_z[i];
        const double xfac = li-1 - _cart_pow_y[i] - _cart_pow_z[i];

        outxx[j*nfi +          i] += xfac * buf[j*nfi + i];
        outxy[j*nfi + _y_addr[i]] += yfac * buf[j*nfi + i];
        outxz[j*nfi + _z_addr[i]] += zfac * buf[j*nfi + i];

        outyx[j*nfi +          i] += xfac * buf[j*nfi + _y_addr[i]];
        outyy[j*nfi + _y_addr[i]] += yfac * buf[j*nfi + _y_addr[i]];
        outyz[j*nfi + _z_addr[i]] += zfac * buf[j*nfi + _y_addr[i]];

        outzx[j*nfi +          i] += xfac * buf[j*nfi + _z_addr[i]];
        outzy[j*nfi + _y_addr[i]] += yfac * buf[j*nfi + _z_addr[i]];
        outzz[j*nfi + _z_addr[i]] += zfac * buf[j*nfi + _z_addr[i]];
    }
    __syncthreads();
}

__device__
void _li_down_down(double *out, double *buf, const int li, const int lj){
    const int nfi = (li+1) * (li+2) / 2;
    const int nfj = (lj+1) * (lj+2) / 2;
    const int nfi2 = (li+3) * (li+4) / 2;
    double *outxx = out;
    double *outxy = out + nfi*nfj;
    double *outxz = out + 2*nfi*nfj;
    double *outyx = out + 3*nfi*nfj;
    double *outyy = out + 4*nfi*nfj;
    double *outyz = out + 5*nfi*nfj;
    double *outzx = out + 6*nfi*nfj;
    double *outzy = out + 7*nfi*nfj;
    double *outzz = out + 8*nfi*nfj;
    const double fac = _ecp_fac[li] / _ecp_fac[li+2];
    for (int ij = threadIdx.x; ij < nfi*nfj; ij+=blockDim.x){
        const int i = ij % nfi;
        const int j = ij / nfi;

        const int idx = i;
        const int idy = _y_addr[i];
        const int idz = _z_addr[i];

        outxx[j*nfi + i] += fac * buf[j*nfi2 + idx];
        outxy[j*nfi + i] += fac * buf[j*nfi2 + idy];
        outxz[j*nfi + i] += fac * buf[j*nfi2 + idz];

        outyx[j*nfi + i] += fac * buf[j*nfi2 + _y_addr[idx]];
        outyy[j*nfi + i] += fac * buf[j*nfi2 + _y_addr[idy]];
        outyz[j*nfi + i] += fac * buf[j*nfi2 + _y_addr[idz]];

        outzx[j*nfi + i] += fac * buf[j*nfi2 + _z_addr[idx]];
        outzy[j*nfi + i] += fac * buf[j*nfi2 + _z_addr[idy]];
        outzz[j*nfi + i] += fac * buf[j*nfi2 + _z_addr[idz]];
    }
    __syncthreads();
}
*/