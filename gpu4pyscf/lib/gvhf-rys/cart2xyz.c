// Edit based on pyscf/lib/dft/grid_common.c

#include <stdint.h>
#include <omp.h>
#include "vhf.cuh"

// up to l=7
#define L_SLOTS 8

static int _LEN_CART0[] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136
};

static int _BINOMIAL_COEF[] = {
    1,
    1,   1,
    1,   2,   1,
    1,   3,   3,   1,
    1,   4,   6,   4,   1,
    1,   5,  10,  10,   5,   1,
    1,   6,  15,  20,  15,   6,   1,
    1,   7,  21,  35,  35,  21,   7,   1,
    1,   8,  28,  56,  70,  56,  28,   8,   1,
    1,   9,  36,  84, 126, 126,  84,  36,   9,   1,
    1,  10,  45, 120, 210, 252, 210, 120,  45,  10,   1,
    1,  11,  55, 165, 330, 462, 462, 330, 165,  55,  11,   1,
    1,  12,  66, 220, 495, 792, 924, 792, 495, 220,  66,  12,   1,
    1,  13,  78, 286, 715,1287,1716,1716,1287, 715, 286,  78,  13,   1,
    1,  14,  91, 364,1001,2002,3003,3432,3003,2002,1001, 364,  91,  14,   1,
    1,  15, 105, 455,1365,3003,5005,6435,6435,5005,3003,1365, 455, 105,  15,   1,
};

#define BINOMIAL(n, i)  (_BINOMIAL_COEF[_LEN_CART0[n]+i])

static void _get_dm_to_dm_xyz_coeff(double* pcx, double* rij, int lmax)
{
    int lmax1 = lmax + 1;
    int l, lx;
    double rx_pow[L_SLOTS];
    double ry_pow[L_SLOTS];
    double rz_pow[L_SLOTS];

    rx_pow[0] = 1.0;
    ry_pow[0] = 1.0;
    rz_pow[0] = 1.0;
    for (lx = 1; lx <= lmax; lx++) {
        rx_pow[lx] = rx_pow[lx-1] * rij[0];
        ry_pow[lx] = ry_pow[lx-1] * rij[1];
        rz_pow[lx] = rz_pow[lx-1] * rij[2];
    }

    double *pcy = pcx + lmax1 * lmax1;
    double *pcz = pcy + lmax1 * lmax1;
    for (l = 0; l <= lmax; l++){
        for (lx = 0; lx <= l; lx++) {
            pcx[l*lmax1+lx] = BINOMIAL(l, lx) * rx_pow[l-lx];
            pcy[l*lmax1+lx] = BINOMIAL(l, lx) * ry_pow[l-lx];
            pcz[l*lmax1+lx] = BINOMIAL(l, lx) * rz_pow[l-lx];
        }
    }
}

static void _dm_to_dm_xyz(double* dm_xyz, double* dm, int nao, int li, int lj, double* ri, double* rj)
{
    double rij[3];
    rij[0] = ri[0] - rj[0];
    rij[1] = ri[1] - rj[1];
    rij[2] = ri[2] - rj[2];

    int lj1 = lj + 1;
    int lij = li + lj;
    int l1 = lij + 1;
    int l1l1 = l1 * l1;
    double pcx[L_SLOTS*L_SLOTS*3];
    double *pcy = pcx + lj1 * lj1;
    double *pcz = pcy + lj1 * lj1;
    _get_dm_to_dm_xyz_coeff(pcx, rij, lj);

    for (int lx = 0; lx <= lij; lx++) {
        for (int ly = 0; ly <= lij-lx; ly++) {
            for (int lz = 0; lz <= lij-lx-ly; lz++) {
                dm_xyz[lx*l1l1+ly*l1+lz] = 0.;
            }
        }
    }

    for (int i = 0, lx_i = li; lx_i >= 0; lx_i--) {
        for (int ly_i = li-lx_i; ly_i >= 0; ly_i--, i++) {
            int lz_i = li - lx_i - ly_i;
            for (int j = 0, lx_j = lj; lx_j >= 0; lx_j--) {
                for (int ly_j = lj-lx_j; ly_j >= 0; ly_j--, j++) {
                    int lz_j = lj - lx_j - ly_j;
                    double dm_ij = dm[i*nao+j];
                    for (int jx = 0; jx <= lx_j; jx++) {
                        double cx = pcx[jx+lx_j*lj1];
                        int lx = lx_i + jx;
                        for (int jy = 0; jy <= ly_j; jy++) {
                            double cxy = cx * pcy[jy+ly_j*lj1];
                            int ly = ly_i + jy;
                            for (int jz = 0; jz <= lz_j; jz++) {
                                double cxyz = cxy * pcz[jz+lz_j*lj1];
                                int lz = lz_i + jz;
                                dm_xyz[lx*l1l1+ly*l1+lz] += cxyz * dm_ij;
                            }
                        }
                    }
                }
            }
        }
    }
}

static void _dm_xyz_to_dm(double* dm_xyz, double* dm, int nao, int li, int lj, double* ri, double* rj)
{
    double rij[3];
    rij[0] = ri[0] - rj[0];
    rij[1] = ri[1] - rj[1];
    rij[2] = ri[2] - rj[2];

    int lj1 = lj + 1;
    int l1 = li + lj + 1;
    int l1l1 = l1 * l1;
    double pcx[L_SLOTS*L_SLOTS*3];
    double *pcy = pcx + lj1 * lj1;
    double *pcz = pcy + lj1 * lj1;
    _get_dm_to_dm_xyz_coeff(pcx, rij, lj);

    for (int i = 0, lx_i = li; lx_i >= 0; lx_i--) {
        for (int ly_i = li-lx_i; ly_i >= 0; ly_i--, i++) {
            int lz_i = li - lx_i - ly_i;
            for (int j = 0, lx_j = lj; lx_j >= 0; lx_j--) {
                for (int ly_j = lj-lx_j; ly_j >= 0; ly_j--, j++) {
                    int lz_j = lj - lx_j - ly_j;
                    double dm_ij = 0;
                    for (int jx = 0; jx <= lx_j; jx++) {
                        double cx = pcx[jx+lx_j*lj1];
                        int lx = lx_i + jx;
                        for (int jy = 0; jy <= ly_j; jy++) {
                            double cy = pcy[jy+ly_j*lj1];
                            int ly = ly_i + jy;
                            for (int jz = 0; jz <= lz_j; jz++) {
                                double cz = pcz[jz+lz_j*lj1];
                                int lz = lz_i + jz;
                                dm_ij += cx*cy*cz * dm_xyz[lx*l1l1+ly*l1+lz];
                            }
                        }
                    }
                    dm[i*nao+j] = dm_ij;
                }
            }
        }
    }
}

void transform_cart_to_xyz(double *dm_xyz, double *dm, int *ao_loc, int *pair_loc,
                           int *bas, int nbas, double *env)
{
#pragma omp parallel
{
    int nao = ao_loc[nbas];
    double cache[L_SLOTS*L_SLOTS*L_SLOTS*8];
#pragma omp for schedule(dynamic, 4)
    for (int ijsh = 0; ijsh < nbas*nbas; ijsh++) {
        int ish = ijsh / nbas;
        int jsh = ijsh % nbas;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int li = bas[ish*BAS_SLOTS+ANG_OF];
        int lj = bas[jsh*BAS_SLOTS+ANG_OF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        _dm_to_dm_xyz(cache, dm+i0*nao+j0, nao, li, lj, ri, rj);

        int lij = li + lj;
        int l1 = lij + 1;
        double *pdm_xyz = dm_xyz + pair_loc[ish*nbas+jsh];
        for (int ix = 0, n = 0; ix <= lij; ix++) {
            for (int iy = 0; iy <= lij-ix; iy++) {
                for (int iz = 0; iz <= lij-ix-iy; iz++, n++) {
                    pdm_xyz[n] = cache[(ix*l1+iy)*l1+iz];
                }
            }
        }
    }
}
}


void transform_xyz_to_cart(double *vj, double *vj_xyz, int *ao_loc, int *pair_loc,
                           int *bas, int nbas, double *env)
{
#pragma omp parallel
{
    int nao = ao_loc[nbas];
    double cache[L_SLOTS*L_SLOTS*L_SLOTS*8];
#pragma omp for schedule(dynamic, 4)
    for (int ijsh = 0; ijsh < nbas*nbas; ijsh++) {
        int ish = ijsh / nbas;
        int jsh = ijsh % nbas;
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int li = bas[ish*BAS_SLOTS+ANG_OF];
        int lj = bas[jsh*BAS_SLOTS+ANG_OF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];

        int lij = li + lj;
        int l1 = lij + 1;
        double *pvj_xyz = vj_xyz + pair_loc[ish*nbas+jsh];
        for (int ix = 0, n = 0; ix <= lij; ix++) {
            for (int iy = 0; iy <= lij-ix; iy++) {
                for (int iz = 0; iz <= lij-ix-iy; iz++, n++) {
                    cache[(ix*l1+iy)*l1+iz] = pvj_xyz[n];
                }
            }
        }
        _dm_xyz_to_dm(cache, vj+i0*nao+j0, nao, li, lj, ri, rj);
    }
}
}
