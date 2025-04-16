#include <float.h>
#define LSUM_MAX        (LMAX*4)
#define EPS_FLOAT64     DBL_EPSILON
#define SQRTPIE4        .886226925452758013

__device__
static void eval_gamma_inc_fn(double *f, double t, int m)
{
    int t_id = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    if (t < EPS_FLOAT64) {
        f[t_id] = 1.;
        for (int i = 1; i <= m; i++) {
            f[t_id + i*block_size] = 1./(2*i+1);
        }
    } else if (m > 0 && t < m*.5+.5) {
        double bi = m + .5;
        double e = .5 * exp(-t);
        double x = e;
        double s = e;
        double tol = EPS_FLOAT64 * e;
        while (x > tol) {
            bi += 1.;
            x *= t / bi;
            s += x;
        }
        double b = m + 0.5;
        double fval = s / b;
        f[t_id + m*block_size] = fval;
        for (int i = m-1; i >= 0; i--) {
            b -= 1.;
            fval = (e + t * fval) / b;
            f[t_id + i*block_size] = fval;
        }
    } else {
        double tt = sqrt(t);
        double fval = SQRTPIE4 / tt * erf(tt);
        f[t_id] = fval;
        if (m > 0) {
            double e = .5 * exp(-t);
            double b = 1. / t;
            double b1 = .5;
            for (int i = 1; i <= m; i++) {
                fval = b * (b1 * fval - e);
                f[t_id + i*block_size] = fval;
                b1 += 1.;
            }
        }
    }
}
