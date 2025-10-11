#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "vhf.cuh"

// sqrt(-log(1e-9))
#define R_GUESS_FAC     4.5f

void sr_eri_s_estimator(float *s_estimator, float omega,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        float *exps = malloc(sizeof(float) * nbas * 5);
        float *cs = exps + nbas;
        float *rx = cs + nbas;
        float *ry = rx + nbas;
        float *rz = ry + nbas;

        for (int n = 0; n < nbas; n++) {
                int ia = bas[ATOM_OF+n*BAS_SLOTS];
                int nprim = bas[NPRIM_OF+n*BAS_SLOTS];
                int nctr = bas[NCTR_OF+n*BAS_SLOTS];
                int ptr_coord = atm[PTR_COORD+ia*ATM_SLOTS];
                int ptr_coeff = bas[PTR_COEFF+n*BAS_SLOTS];
                exps[n] = env[bas[PTR_EXP+n*BAS_SLOTS] + nprim-1];
                rx[n] = env[ptr_coord+0];
                ry[n] = env[ptr_coord+1];
                rz[n] = env[ptr_coord+2];

                float c_max = fabs(env[ptr_coeff + nprim-1]);
                for (int m = 1; m < nctr; m++) {
                        float c1 = fabs(env[ptr_coeff + (m+1)*nprim-1]);
                        c_max = MAX(c_max, c1);
                }
                cs[n] = c_max;
        }
        float omega2 = omega * omega;

#pragma omp parallel
{
        float fac_guess = .5f - logf(omega2)/4;
        int ish, jsh, li, lj;
        float ai, aj, aij, ai_aij, a1, ci, cj;
        float xi, yi, zi, xj, yj, zj;
        float dx, dy, dz, r2, v, log_fac;
#pragma omp for schedule(dynamic, 1)
        for (ish = 0; ish < nbas; ish++) {
                li = bas[ANG_OF+ish*BAS_SLOTS];
                ai = exps[ish];
                ci = cs[ish];
                xi = rx[ish];
                yi = ry[ish];
                zi = rz[ish];
#pragma GCC ivdep
                for (jsh = 0; jsh <= ish; jsh++) {
                        lj = bas[ANG_OF+jsh*BAS_SLOTS];
                        aj = exps[jsh];
                        cj = cs[jsh];
                        xj = rx[jsh];
                        yj = ry[jsh];
                        zj = rz[jsh];
                        dx = xj - xi;
                        dy = yj - yi;
                        dz = zj - zi;
                        aij = ai + aj;
                        ai_aij = ai / aij;
                        a1 = ai_aij * aj;

                        float omega_aij = omega2/(omega2+aij);
                        float r_guess = R_GUESS_FAC / sqrtf(aij * omega_aij);
                        float rt_aij = omega_aij * r_guess;
                        // log(ci*cj * ((2*li+1)*(2*lj+1))**.5/(4*pi) * (pi/aij)**1.5)
                        log_fac = logf(ci*cj * sqrtf((2*li+1.f)*(2*lj+1.f))/(4*M_PI))
                                + 1.5f*logf(M_PI/aij) + fac_guess;
                        r2 = dx * dx + dy * dy + dz * dz;
                        v = (li+lj)*logf(MAX(rt_aij, 1.f)) - a1*r2 + log_fac;
                        s_estimator[ish*nbas+jsh] = v;
                        s_estimator[jsh*nbas+ish] = v;
                }
        }
}
        free(exps);
}

void sr_eri_s_estimator_v2(float *s_estimator, float omega,
                           float *diffuse_exps, float *diffuse_ctr_coef,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        float *rx = malloc(sizeof(float) * nbas * 3);
        float *ry = rx + nbas;
        float *rz = ry + nbas;

        for (int n = 0; n < nbas; n++) {
                int ia = bas[ATOM_OF+n*BAS_SLOTS];
                int ptr_coord = atm[PTR_COORD+ia*ATM_SLOTS];
                rx[n] = env[ptr_coord+0];
                ry[n] = env[ptr_coord+1];
                rz[n] = env[ptr_coord+2];
        }
        float omega2 = omega * omega;

#pragma omp parallel
{
        float fac_guess = .5f - logf(omega2)/4;
        int ish, jsh, li, lj;
        float ai, aj, aij, ci, cj;
        float xi, yi, zi, xj, yj, zj;
        float dx, dy, dz, r2, log_fac;
#pragma omp for schedule(dynamic, 1)
        for (ish = 0; ish < nbas; ish++) {
                li = bas[ANG_OF+ish*BAS_SLOTS];
                ai = diffuse_exps[ish];
                ci = diffuse_ctr_coef[ish];
                xi = rx[ish];
                yi = ry[ish];
                zi = rz[ish];
#pragma GCC ivdep
                for (jsh = 0; jsh <= ish; jsh++) {
                        lj = bas[ANG_OF+jsh*BAS_SLOTS];
                        aj = diffuse_exps[jsh];
                        cj = diffuse_ctr_coef[jsh];
                        xj = rx[jsh];
                        yj = ry[jsh];
                        zj = rz[jsh];
                        dx = xj - xi;
                        dy = yj - yi;
                        dz = zj - zi;
                        aij = ai + aj;
                        float ai_aij = ai / aij;
                        float aj_aij = aj / aij;
                        float theta_ij = ai_aij * aj;

                        float omega_aij = omega2/(omega2+aij);
                        float r_guess = R_GUESS_FAC / sqrtf(aij * omega_aij);
                        // log(ci*cj * ((2*li+1)*(2*lj+1))**.5/(4*pi) * (pi/aij)**1.5)
                        log_fac = logf(ci*cj * sqrtf((2*li+1.f)*(2*lj+1.f))/(4*M_PI))
                                + 1.7171f - 1.5f*logf(aij) + fac_guess;
                        r2 = dx * dx + dy * dy + dz * dz;
                        float dri = aj_aij * r_guess;
                        float drj = ai_aij * r_guess;
                        float dri_fac = .5f*li * logf(.5f*li/aij + dri*dri + 1e-9f);
                        float drj_fac = .5f*lj * logf(.5f*lj/aij + drj*drj + 1e-9f);
                        float estimator = dri_fac + drj_fac - theta_ij*r2 + log_fac;
                        s_estimator[ish*nbas+jsh] = estimator;
                        s_estimator[jsh*nbas+ish] = estimator;
                }
        }
}
        free(rx);
}
