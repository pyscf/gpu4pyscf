#include "rys_roots.cuh"

#define SQRTPIE4        .8862269254527580136
#define PIE4            .7853981633974483096

__device__
static void rys_roots(int nroots, double x, double *rw)
{
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    int block_size = blockDim.x * blockDim.y;

    if (x < 3.e-7){
        int off = nroots * (nroots - 1) / 2;
        for (int i = 0; i < nroots; ++i)  {
            rw[t_id+(i*2  )*block_size] = ROOT_SMALLX_R0[off+i] + ROOT_SMALLX_R1[off+i] * x;
            rw[t_id+(i*2+1)*block_size] = ROOT_SMALLX_W0[off+i] + ROOT_SMALLX_W1[off+i] * x;
        }
        return;
    }

    if (x > 35+nroots*5) {
        int off = nroots * (nroots - 1) / 2;
        double t = sqrt(PIE4/x);
        for (int i = 0; i < nroots; ++i)  {
            rw[t_id+(i*2  )*block_size] = ROOT_LARGEX_R_DATA[off+i] / x;
            rw[t_id+(i*2+1)*block_size] = ROOT_LARGEX_W_DATA[off+i] * t;
        }
        return;
    }

    if (nroots == 1) {
        double tt = sqrt(x);
        double fmt0 = SQRTPIE4 / tt * erf(tt);
        rw[t_id+block_size] = fmt0;
        double e = exp(-x);
        double b = .5 / x;
        double fmt1 = b * (fmt0 - e);
        rw[t_id] = fmt1 / fmt0;
        return;
    }

    double *datax = ROOT_RW_DATA + DEGREE1*INTERVALS * nroots*(nroots-1);
    int it = (int)(x * .4);
    double u = (x - it * 2.5) * 0.8 - 1.;
    double u2 = u * 2.;
    for (int rt_id = 0; rt_id < nroots*2; ++rt_id) {
        double *c = datax + rt_id * DEGREE1 * INTERVALS;
        //for i in range(2, degree + 1):
        //    c0, c1 = c[degree-i] - c1, c0 + c1*u2
        double c0 = c[it + DEGREE   *INTERVALS];
        double c1 = c[it +(DEGREE-1)*INTERVALS];
        double c2, c3;
#pragma unroll
        for (int n = DEGREE-2; n > 0; n-=2) {
            c2 = c[it + n   *INTERVALS] - c1;
            c3 = c0 + c1*u2;
            c1 = c2 + c3*u2;
            c0 = c[it +(n-1)*INTERVALS] - c3;
        }
        if (DEGREE % 2 == 0) {
            c2 = c[it] - c1;
            c3 = c0 + c1*u2;
            rw[t_id + rt_id*block_size] = c2 + c3*u;
        } else {
            rw[t_id + rt_id*block_size] = c0 + c1*u;
        }
    }
}
