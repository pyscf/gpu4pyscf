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

 template <int l> __device__
static void ang_nuc_l(double *omega, double rx, double ry, double rz){
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
    sph2cart(omega, l, c);
}

__device__
static void ang_nuc_part(double *omega_cum, int l, double rx, double ry, double rz){
    /*
    Accumulated angular ECP part in Cartesian
    Angular momentum L = 0, 1, ... LMAX
    Cartesian xyz --> Spherical --> Cartesian

    The code is generated with generate_ang_nuc.py
    */
    double *omega = omega_cum;
    if (l >= 0){
        omega[0] = 0.07957747154594767;
        omega += 1;
    }

    if (l >= 1){
        omega[0] = 0.2387324146378430 * rx;
        omega[1] = 0.2387324146378430 * ry;
        omega[2] = 0.2387324146378430 * rz;
        omega += 3;
    }

    if (l >= 2){
        double g[6];
        g[0] = rx * rx;
        g[1] = rx * ry;
        g[2] = rx * rz;
        g[3] = ry * ry;
        g[4] = ry * rz;
        g[5] = rz * rz;
        double c[5];
        cart2sph(c, 2, g);
        sph2cart(omega, 2, c);
        omega += 6;
    }

    if (l >= 3){
        ang_nuc_l<3>(omega, rx, ry, rz);
        omega += 10;
    }

    if (l >= 4){
        ang_nuc_l<4>(omega, rx, ry, rz);
        omega += 15;
    }

    if (l >= 5){
        ang_nuc_l<5>(omega, rx, ry, rz);
        omega += 21;
    }

    if (l >= 6) {
        ang_nuc_l<6>(omega, rx, ry, rz);
        omega += 28;
    }

    if (l >= 7) {
        ang_nuc_l<7>(omega, rx, ry, rz);
        omega += 36;
    }

    if (l >= 8) {
        ang_nuc_l<8>(omega, rx, ry, rz);
        omega += 45;
    }
}

__device__
double rad_part(int ish, const int *ecpbas, const double *env){
    const int npk = ecpbas[ish*BAS_SLOTS+NPRIM_OF];
    const int r_order = ecpbas[ish*BAS_SLOTS+RADI_POWER];
    const int exp_ptr = ecpbas[ish*BAS_SLOTS+PTR_EXP];
    const int coeff_ptr = ecpbas[ish*BAS_SLOTS+PTR_COEFF];

    double u1 = 0.0;
    const double r = r128[threadIdx.x];
    for (int kp = 0; kp < npk; kp++){
        const double ak = env[exp_ptr+kp];
        const double ck = env[coeff_ptr+kp];
        u1 += ck * exp(-ak * r * r);
    }
    return u1 * pow(r, r_order) * w128[threadIdx.x];
}

__device__
void cache_fac(double *fx, int LI, double *ri){
    const int LI1 = LI + 1;
    double xx[AO_LMAX+1], yy[AO_LMAX+1], zz[AO_LMAX+1];
    xx[0] = 1; yy[0] = 1; zz[0] = 1;
    for (int i = 1; i <= LI; i++){
        xx[i] = xx[i-1] * ri[0];
        yy[i] = yy[i-1] * ri[1];
        zz[i] = zz[i-1] * ri[2];
    }

    const int nfi = (LI1+1)*LI1/2;
    double *fy = fx + nfi;
    double *fz = fy + nfi;
    for (int i = 0; i <= LI; i++){
        int ioffset = i*(i+1)/2;
        for (int j = 0; j <= i; j++){
            const double bfac = _binom[ioffset+j]; // binom(i,j)
            fx[ioffset+j] = bfac * xx[i-j];
            fy[ioffset+j] = bfac * yy[i-j];
            fz[ioffset+j] = bfac * zz[i-j];
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
        d_out[0] = sdata[0];
    }
    __syncthreads();
}


__device__
void _l_up(double *out, double *buf, double fac, int li, int lj){
    const int nfi = (li+1) * (li+2) / 2;
    const int nfi0 = li * (lj+1) / 2;
    double *outx = out;
    double *outy = out + nfi*nfj;
    double *outz = outy + nfi*nfj;
    for (int ij = threadIdx.x; ij < nfi*nfi0; ij+=blockDim.x){
        const int i = ij % nfi;
        const int j = ij / nfi;
        double yfac = fac * (_cart_pow_y[i] + 1);
        double zfac = fac * (_cart_pow_z[i] + 1);
        double xfac = fac * (li-1 - _cart_pow_y[i] - _cart_pow_z[i]);

        outx[i*nfj + j] += yfac * buf[i*nfj + j];
        outy[i*nfj + j] += zfac * buf[i*nfj + j];
        outz[i*nfj + j] += xfac * buf[i*nfj + j];
    }
    __syncthreads();
}

__device__
void _l_down(double *gctr, double *buf, double fac, int li, int lj){
    const int nfi = (li+1) * (li+2) / 2;
    const int nfj = (lj+1) * (lj+2) / 2;
    
}