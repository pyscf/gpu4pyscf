#include <stdlib.h>
#include <math.h>
#include "gvhf-rys/vhf.cuh"

#define Ex_at(i,j,t)    Ex[(i)*stride1+(j)*stride2+t]
#define Ey_at(i,j,t)    Ey[(i)*stride1+(j)*stride2+t]
#define Ez_at(i,j,t)    Ez[(i)*stride1+(j)*stride2+t]

void get_E_cart_components(double *Ecart, int li, int lj, double ai, double aj,
                           double *Ra, double *Rb)
{
        double aij = ai + aj;
        double xixj = Ra[0] - Rb[0];
        double yiyj = Ra[1] - Rb[1];
        double zizj = Ra[2] - Rb[2];
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xixj*xixj + yiyj*yiyj + zizj*zizj));
        double Xp = (ai * Ra[0] + aj * Rb[0]) / aij;
        double Yp = (ai * Ra[1] + aj * Rb[1]) / aij;
        double Zp = (ai * Ra[2] + aj * Rb[2]) / aij;
        double Xpa = Xp - Ra[0];
        double Ypa = Yp - Ra[1];
        double Zpa = Zp - Ra[2];
        double Xpb = Xp - Rb[0];
        double Ypb = Yp - Rb[1];
        double Zpb = Zp - Rb[2];
        int lij = li + lj;
        int stride2 = lij+1;
        int stride1 = (lj+1) * stride2;
        int Ex_size = (li+1) * stride1;
        double *Ex = Ecart;
        double *Ey = Ex + Ex_size;
        double *Ez = Ey + Ex_size;
        int i, j, t;
        double fac, fac1;

        Ex_at(0,0,0) = 1.;
        Ey_at(0,0,0) = 1.;
        Ez_at(0,0,0) = Kab;
        for (t = 1; t <= lij; t++) {
                Ex_at(0,0,t) = 0.;
                Ey_at(0,0,t) = 0.;
                Ez_at(0,0,t) = 0.;
        }

        for (j = 1; j <= lj; j++) {
                Ex_at(0,j,0) = Xpb * Ex_at(0,j-1,0) + Ex_at(0,j-1,1);
                Ey_at(0,j,0) = Ypb * Ey_at(0,j-1,0) + Ey_at(0,j-1,1);
                Ez_at(0,j,0) = Zpb * Ez_at(0,j-1,0) + Ez_at(0,j-1,1);
                for (t = 1; t <= lij; t++) {
                        fac = j/(2*aij*t);
                        Ex_at(0,j,t) = fac * Ex_at(0,j-1,t-1);
                        Ey_at(0,j,t) = fac * Ey_at(0,j-1,t-1);
                        Ez_at(0,j,t) = fac * Ez_at(0,j-1,t-1);
                }
        }

        for (i = 1; i <= li; i++) {
                Ex_at(i,0,0) = Xpa * Ex_at(i-1,0,0) + Ex_at(i-1,0,1);
                Ey_at(i,0,0) = Ypa * Ey_at(i-1,0,0) + Ey_at(i-1,0,1);
                Ez_at(i,0,0) = Zpa * Ez_at(i-1,0,0) + Ez_at(i-1,0,1);
                for (t = 1; t <= lij; t++) {
                        fac = i/(2*aij*t);
                        Ex_at(i,0,t) = fac * Ex_at(i-1,0,t-1);
                        Ey_at(i,0,t) = fac * Ey_at(i-1,0,t-1);
                        Ez_at(i,0,t) = fac * Ez_at(i-1,0,t-1);
                }
        }

        for (i = 1; i <= li; i++) {
                for (j = 1; j <= lj; j++) {
                        Ex_at(i,j,0) = Xpb * Ex_at(i,j-1,0) + Ex_at(i,j-1,1);
                        Ey_at(i,j,0) = Ypb * Ey_at(i,j-1,0) + Ey_at(i,j-1,1);
                        Ez_at(i,j,0) = Zpb * Ez_at(i,j-1,0) + Ez_at(i,j-1,1);
                        for (t = 1; t <= lij; t++) {
                                fac = i/(2*aij*t);
                                fac1 = j/(2*aij*t);
                                Ex_at(i,j,t) = fac*Ex_at(i-1,j,t-1) + fac1*Ex_at(i,j-1,t-1);
                                Ey_at(i,j,t) = fac*Ey_at(i-1,j,t-1) + fac1*Ey_at(i,j-1,t-1);
                                Ez_at(i,j,t) = fac*Ez_at(i-1,j,t-1) + fac1*Ez_at(i,j-1,t-1);
                        }
                }
        }
}

// Shape of E tensor is [:li+lj,:li,:lj]
void get_E_tensor(double *Et, int li, int lj, double ai, double aj,
                  double *Ra, double *Rb, double *buf)
{
        get_E_cart_components(buf, li, lj, ai, aj, Ra, Rb);
        int lij = li + lj;
        int stride2 = lij+1;
        int stride1 = (lj+1) * stride2;
        int Ex_size = (li+1) * stride1;
        double *Ex = buf;
        double *Ey = Ex + Ex_size;
        double *Ez = Ey + Ex_size;
        int t, u, v, n;
        int ix, iy, iz;
        int jx, jy, jz;

        n = 0;
        // products subject to t+u+v <= li+lj
        for (t = 0; t <= lij; t++) {
        for (u = 0; u <= lij-t; u++) {
        for (v = 0; v <= lij-t-u; v++) {
                for (ix = li; ix >= 0; ix--) {
                for (iy = li-ix; iy >= 0; iy--) {
                        iz = li - ix - iy;
                        for (jx = lj; jx >= 0; jx--) {
                        for (jy = lj-jx; jy >= 0; jy--) {
                                jz = lj - jx - jy;
                                Et[n] = Ex_at(ix,jx,t) * Ey_at(iy,jy,u) * Ez_at(iz,jz,v);
                                n++;
                        } }
                } }
        } } }
}

void Et_dot_dm(double *Et_dm, double *dm, int *ao_loc, int *pair_loc,
               int *pair_lst, int npairs, int *p2c_mapping,
               int p_nbas, int c_nbas, int *bas, double *env)
{
        int l2 = 2*LMAX;
        int Et_size = (l2+1)*(l2+2)*(l2+3)/6*NCART_MAX*NCART_MAX;
        int Ex_size = (2*LMAX+1)*(LMAX+1)*(LMAX+1);
        double *Et = malloc(sizeof(double) * (Et_size+3*Ex_size));
        double *buf = Et + Et_size;

        size_t nao = ao_loc[c_nbas];
        for (int task_ij = 0; task_ij < npairs; task_ij++) {
                int pair_ij = pair_lst[task_ij];
                int ish = pair_ij / p_nbas;
                int jsh = pair_ij % p_nbas;
                int ctr_ish = p2c_mapping[ish];
                int ctr_jsh = p2c_mapping[jsh];
                int li = bas[ish*BAS_SLOTS+ANG_OF];
                int lj = bas[jsh*BAS_SLOTS+ANG_OF];
                double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
                double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
                double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
                double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
                double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
                double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
                double *rho = Et_dm + pair_loc[task_ij];
                int lij = li + lj;
                int nfi = (li + 1) * (li + 2) / 2;
                int nfj = (lj + 1) * (lj + 2) / 2;
                int Et_len = (lij + 1) * (lij + 2) * (lij + 3) / 6;
                get_E_tensor(Et, li, lj, ai, aj, ri, rj, buf);
                double cc = ci * cj;
                double *pdm = dm + ao_loc[ctr_ish] * nao + ao_loc[ctr_jsh];
                for (int n = 0, t = 0; t < Et_len; t++) {
                        double rho_t = 0.;
                        for (int i = 0; i < nfi; i++) {
                        for (int j = 0; j < nfj; j++, n++) {
                                rho_t += Et[n] * cc * pdm[i*nao+j];
                        } }
                        rho[t] = rho_t;
                }
        }
        free(Et);
}

void jengine_dot_Et(double *vj, double *jvec, int *ao_loc, int *pair_loc,
                    int *pair_lst, int npairs, int *p2c_mapping,
                    int p_nbas, int c_nbas, int *bas, double *env)
{
        int l2 = 2*LMAX;
        int Et_size = (l2+1)*(l2+2)*(l2+3)/6*NCART_MAX*NCART_MAX;
        int Ex_size = (2*LMAX+1)*(LMAX+1)*(LMAX+1);
        double *Et = malloc(sizeof(double) * (Et_size+3*Ex_size));
        double *buf = Et + Et_size;

        size_t nao = ao_loc[c_nbas];
        for (int task_ij = 0; task_ij < npairs; task_ij++) {
                int pair_ij = pair_lst[task_ij];
                int ish = pair_ij / p_nbas;
                int jsh = pair_ij % p_nbas;
                int ctr_ish = p2c_mapping[ish];
                int ctr_jsh = p2c_mapping[jsh];
                int li = bas[ish*BAS_SLOTS+ANG_OF];
                int lj = bas[jsh*BAS_SLOTS+ANG_OF];
                double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];
                double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];
                double ci = env[bas[ish*BAS_SLOTS+PTR_COEFF]];
                double cj = env[bas[jsh*BAS_SLOTS+PTR_COEFF]];
                double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
                double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
                double *jvec_ij = jvec + pair_loc[task_ij];
                int lij = li + lj;
                int nfi = (li + 1) * (li + 2) / 2;
                int nfj = (lj + 1) * (lj + 2) / 2;
                int Et_len = (lij + 1) * (lij + 2) * (lij + 3) / 6;
                get_E_tensor(Et, li, lj, ai, aj, ri, rj, buf);
                double cc = ci * cj;
                double *pj = vj + ao_loc[ctr_ish] * nao + ao_loc[ctr_jsh];
                for (int n = 0, t = 0; t < Et_len; t++) {
                        double fac = cc * jvec_ij[t];
                        for (int i = 0; i < nfi; i++) {
                        for (int j = 0; j < nfj; j++, n++) {
                                pj[i*nao+j] += Et[n] * fac;
                        } }
                }
        }
        free(Et);
}
