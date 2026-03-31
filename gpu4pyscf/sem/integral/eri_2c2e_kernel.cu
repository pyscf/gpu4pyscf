/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_70 eri_2c2e_kernel.cu -o liberi_2c2e_kernel.so

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define INV_SQRT(x) (1.0 / sqrt(x))
#define SQR(x) ((x) * (x))

__device__ double charg_kernel_device(
    double r, 
    int l1, int l2, int m, 
    double da, double db, 
    double add
) {
    double rsq = r * r;

    // Q-Q
    if (l1 == 0 && l2 == 0) {
        return INV_SQRT(rsq + add);
    }

    // Z-Q
    if (l1 == 1 && l2 == 0) {
        double t1 = INV_SQRT((r + da)*(r + da) + add);
        double t2 = INV_SQRT((r - da)*(r - da) + add);
        return 0.5 * (t2 - t1);
    }
    // Q-Z
    if (l1 == 0 && l2 == 1) {
        double t1 = INV_SQRT((r + db)*(r + db) + add);
        double t2 = INV_SQRT((r - db)*(r - db) + add);
        return 0.5 * (t1 - t2);
    }
    if (l1 == 1 && l2 == 1) {
        // Z-Z (Sigma) m=0
        if (m == 0) {
            double t1 = INV_SQRT((r + da - db)*(r + da - db) + add);
            double t2 = INV_SQRT((r - da + db)*(r - da + db) + add);
            double t3 = INV_SQRT((r - da - db)*(r - da - db) + add);
            double t4 = INV_SQRT((r + da + db)*(r + da + db) + add);
            return 0.25 * (t1 + t2 - t3 - t4);
        }
        // X-X (Pi) m=1
        else if (m == 1) {
            double t1 = INV_SQRT(rsq + (da - db)*(da - db) + add);
            double t2 = INV_SQRT(rsq + (da + db)*(da + db) + add);
            return 0.25 * (2.0 * t1 - 2.0 * t2);
        }
    }

    // Q-ZZ
    if (l1 == 0 && l2 == 2) {
        double t1 = INV_SQRT((r - db)*(r - db) + add);
        double t2 = INV_SQRT(rsq + db*db + add);
        double t3 = INV_SQRT((r + db)*(r + db) + add);
        return 0.25 * (t1 - 2.0 * t2 + t3);
    }
    // ZZ-Q
    if (l1 == 2 && l2 == 0) {
        double t1 = INV_SQRT((r - da)*(r - da) + add);
        double t2 = INV_SQRT(rsq + da*da + add);
        double t3 = INV_SQRT((r + da)*(r + da) + add);
        return 0.25 * (t1 - 2.0 * t2 + t3);
    }

    // Z-ZZ (m=0)
    if (l1 == 1 && l2 == 2 && m == 0) {
        double t1 = INV_SQRT(SQR(r - da - db) + add);
        double t2 = INV_SQRT(SQR(r - da) + db*db + add);
        double t3 = INV_SQRT(SQR(r + db - da) + add);
        double t4 = INV_SQRT(SQR(r - db + da) + add);
        double t5 = INV_SQRT(SQR(r + da) + db*db + add);
        double t6 = INV_SQRT(SQR(r + da + db) + add);
        return 0.125 * (t1 - 2.0*t2 + t3 - t4 + 2.0*t5 - t6);
    }
    // ZZ-Z (m=0)
    if (l1 == 2 && l2 == 1 && m == 0) {
        double t1 = INV_SQRT(SQR(r - da - db) + add);
        double t2 = INV_SQRT(SQR(r - db) + da*da + add);
        double t3 = INV_SQRT(SQR(r + da - db) + add);
        double t4 = INV_SQRT(SQR(r - da + db) + add);
        double t5 = INV_SQRT(SQR(r + db) + da*da + add);
        double t6 = INV_SQRT(SQR(r + da + db) + add);
        return 0.125 * (-t1 + 2.0*t2 - t3 + t4 - 2.0*t5 + t6);
    }
    // X-ZX m=1
    if (l1 == 1 && l2 == 2 && m == 1) {
        double ab = db / 1.4142135623730951;
        double t1 = INV_SQRT(SQR(r - ab) + SQR(da - ab) + add);
        double t2 = INV_SQRT(SQR(r + ab) + SQR(da - ab) + add);
        double t3 = INV_SQRT(SQR(r - ab) + SQR(da + ab) + add);
        double t4 = INV_SQRT(SQR(r + ab) + SQR(da + ab) + add);
        return 0.125 * (-2.0*t1 + 2.0*t2 + 2.0*t3 - 2.0*t4);
    }
    // ZX-X m=1
    if (l1 == 2 && l2 == 1 && m == 1) {
        double aa = da / 1.4142135623730951;
        double t1 = INV_SQRT(SQR(r + aa) + SQR(aa - db) + add);
        double t2 = INV_SQRT(SQR(r - aa) + SQR(aa - db) + add);
        double t3 = INV_SQRT(SQR(r + aa) + SQR(aa + db) + add);
        double t4 = INV_SQRT(SQR(r - aa) + SQR(aa + db) + add);
        return 0.125 * (-2.0*t1 + 2.0*t2 + 2.0*t3 - 2.0*t4);
    }

    if (l1 == 2 && l2 == 2) {
        // ZZ-ZZ m=0
        if (m == 0) {
            double t1 = INV_SQRT(SQR(r - da - db) + add);
            double t2 = INV_SQRT(SQR(r + da + db) + add);
            double t3 = INV_SQRT(SQR(r - da + db) + add);
            double t4 = INV_SQRT(SQR(r + da - db) + add);
            
            double t5 = INV_SQRT(SQR(r - da) + db*db + add);
            double t6 = INV_SQRT(SQR(r - db) + da*da + add);
            double t7 = INV_SQRT(SQR(r + da) + db*db + add);
            double t8 = INV_SQRT(SQR(r + db) + da*da + add);
            
            double t9  = INV_SQRT(rsq + SQR(da - db) + add);
            double t10 = INV_SQRT(rsq + SQR(da + db) + add);
            
            double zzzz = t1 + t2 + t3 + t4 - 2.0*(t5 + t6 + t7 + t8) + 2.0*(t9 + t10);
            
            double t11 = INV_SQRT(rsq + da*da + db*db + add);
            double xyxy = 4.0*t9 + 4.0*t10 - 8.0*t11;
            
            return (zzzz / 16.0) - (xyxy / 64.0);
        }
        // ZX-ZX m=1
        if (m == 1) {
            double aa = da / 1.4142135623730951;
            double ab = db / 1.4142135623730951;
            
            double v1 = INV_SQRT(SQR(r + aa - ab) + SQR(aa - ab) + add);
            double v2 = INV_SQRT(SQR(r + aa + ab) + SQR(aa - ab) + add);
            double v3 = INV_SQRT(SQR(r - aa - ab) + SQR(aa - ab) + add);
            double v4 = INV_SQRT(SQR(r - aa + ab) + SQR(aa - ab) + add);
            
            double v5 = INV_SQRT(SQR(r + aa - ab) + SQR(aa + ab) + add);
            double v6 = INV_SQRT(SQR(r + aa + ab) + SQR(aa + ab) + add);
            double v7 = INV_SQRT(SQR(r - aa - ab) + SQR(aa + ab) + add);
            double v8 = INV_SQRT(SQR(r - aa + ab) + SQR(aa + ab) + add);
            
            return 0.0625 * (2.0*v1 - 2.0*v2 - 2.0*v3 + 2.0*v4 - 2.0*v5 + 2.0*v6 + 2.0*v7 - 2.0*v8);
        }
        if (m == 2) {
            double t1 = INV_SQRT(rsq + SQR(da - db) + add);
            double t2 = INV_SQRT(rsq + SQR(da + db) + add);
            double t3 = INV_SQRT(rsq + da*da + db*db + add);
            return 0.0625 * (4.0*t1 + 4.0*t2 - 8.0*t3);
        }
    }

    return 0.0;
}

__device__ double rijkl_device(
    int ni, int nj,                         // 0-based atom indices
    int ij, int kl,                         // Orbital pair combinations (0..44)
    int li, int lj,                         // Angular momentum for left pair (0=s, 1=p, 2=d)
    int lk, int ll,                         // Angular momentum for right pair
    int ic,                                 // Core flag: 1=left is core, 2=right is core, 0=normal
    double r,                               // Interatomic distance in Bohr
    int n_atom,                             // Total number of atoms (for tensor striding)
    const double* __restrict__ po_tensor,   // Shape: (3, 3, 3, n_atom)
    const double* __restrict__ ddp_tensor,  // Shape: (3, 3, n_atom)
    const double* __restrict__ core_rho,    // Shape: (n_atom,)
    const double* __restrict__ ch           // Shape: (45, 3, 5)
) {

    int l1min = abs(li - lj); 
    if (l1min > 2) l1min = 2;
    int l1max = li + lj;      
    if (l1max > 2) l1max = 2;
    int l2min = abs(lk - ll); 
    if (l2min > 2) l2min = 2;
    int l2max = lk + ll;      
    if (l2max > 2) l2max = 2;

    double total = 0.0;

    for (int l1 = l1min; l1 <= l1max; ++l1) {
        double pij = 0.0;
        double dij = 0.0;
        
        if (l1 == 0) {
            // Special case: electron-core interaction for left atom
            if (ic == 1 && li == 0 && lj == 0) {
                pij = core_rho[ni];
            } else {
                pij = po_tensor[li * (9 * n_atom) + lj * (3 * n_atom) + 0 * n_atom + ni];
            }
            dij = 0.0;
        } else {
            pij = po_tensor[li * (9 * n_atom) + lj * (3 * n_atom) + l1 * n_atom + ni];
            dij = ddp_tensor[li * (3 * n_atom) + lj * n_atom + ni];
        }

        for (int l2 = l2min; l2 <= l2max; ++l2) {
            double pkl = 0.0;
            double dkl = 0.0;
            
            if (l2 == 0) {
                // Special case: electron-core interaction for right atom
                if (ic == 2 && lk == 0 && ll == 0) {
                    pkl = core_rho[nj];
                } else {
                    pkl = po_tensor[lk * (9 * n_atom) + ll * (3 * n_atom) + 0 * n_atom + nj];
                }
                dkl = 0.0;
            } else {
                pkl = po_tensor[lk * (9 * n_atom) + ll * (3 * n_atom) + l2 * n_atom + nj];
                dkl = ddp_tensor[lk * (3 * n_atom) + ll * n_atom + nj];
            }

            double add = (pij + pkl) * (pij + pkl);
            int lmin_m = l1 < l2 ? l1 : l2;

            double s1 = 0.0;
            for (int m = -lmin_m; m <= lmin_m; ++m) {
                double c1 = ch[ij * 15 + l1 * 5 + (m + 2)];
                double c2 = ch[kl * 15 + l2 * 5 + (m + 2)];
                double ccc = c1 * c2;

                if (ccc == 0.0) continue;

                int mm = abs(m);
                s1 += charg_kernel_device(r, l1, l2, mm, dij, dkl, add) * ccc;
            }
            total += s1;
        }
    }
    return total;
}


// Computes the first 22 main-group terms and core-core monopole G_AB
// TODO: This function is hardcoded for sp orbitals, this will be removed in the future
// TODO: All the input parameters are the same, or containing the same info as po_tensor and ddp_tensor
__device__ void reppd_device(
    int ni, int nj, double r, 
    const double* __restrict__ am,
    const double* __restrict__ ad, 
    const double* __restrict__ aq,
    const double* __restrict__ dd, 
    const double* __restrict__ qq,
    const double* __restrict__ core_rho, 
    int natorb_i, 
    int natorb_j, 
    const double HATREE2EV,
    double* __restrict__ ri,    // Output 2c2e for sp orbitals
    double& gab                 // Output
) {
    double td = 2.0;
    double half = 0.5;
    double rsq = r * r;

    // G_AB (Core-Core Monopole)
    double aee_cc = core_rho[ni] + core_rho[nj];
    aee_cc = aee_cc * aee_cc;
    gab = HATREE2EV / sqrt(rsq + aee_cc);

    double aee_te = SQR(half / am[ni] + half / am[nj]);

    // Matching Python: si = (env.natorb[ni] >= 3)
    bool si = (natorb_i >= 3);
    bool sj = (natorb_j >= 3);

    for (int i = 0; i < 22; ++i) ri[i] = 0.0;

    if (!si && !sj) {
        // H - H  (SS/SS)
        ri[0] = HATREE2EV / sqrt(rsq + aee_te);
    }
    else if (si && !sj) {
        // Heavy - H
        double da = dd[ni];
        double qa = qq[ni] * td;
        double ade = SQR(half / ad[ni] + half / am[nj]);
        double aqe = SQR(half / aq[ni] + half / am[nj]);

        double arg1 = SQR(r + da) + ade;
        double arg2 = SQR(r - da) + ade;
        double arg3 = SQR(r + qa) + aqe;
        double arg4 = SQR(r - qa) + aqe;
        double arg5 = rsq + aqe;
        double arg6 = arg5 + qa * qa;

        double ev1 = HATREE2EV / 2.0;
        double ev2 = HATREE2EV / 4.0;
        double ee = HATREE2EV / sqrt(rsq + aee_te);

        ri[0] = ee;
        ri[1] = -(ev1 / sqrt(arg1) - ev1 / sqrt(arg2));                 
        ri[2] = ee + ev2 / sqrt(arg3) + ev2 / sqrt(arg4) - ev1 / sqrt(arg5); 
        ri[3] = ee + ev1 / sqrt(arg6) - ev1 / sqrt(arg5);             
    }
    else if (!si && sj) {
        // ------ H - Heavy ------
        double db = dd[nj];
        double qb = qq[nj] * td;
        double aed = SQR(half / am[ni] + half / ad[nj]);
        double aeq = SQR(half / am[ni] + half / aq[nj]);

        double arg1 = SQR(r - db) + aed;
        double arg2 = SQR(r + db) + aed;
        double arg3 = SQR(r - qb) + aeq;
        double arg4 = SQR(r + qb) + aeq;
        double arg5 = rsq + aeq;
        double arg6 = arg5 + qb * qb;

        double ev1 = HATREE2EV / 2.0;
        double ev2 = HATREE2EV / 4.0;
        double ee = HATREE2EV / sqrt(rsq + aee_te);

        ri[0]  = ee;
        ri[4]  = -(ev1 / sqrt(arg1) - ev1 / sqrt(arg2));                 
        ri[10] = ee + ev2 / sqrt(arg3) + ev2 / sqrt(arg4) - ev1 / sqrt(arg5); 
        ri[11] = ee + ev1 / sqrt(arg6) - ev1 / sqrt(arg5);             
    }
    else {
        // ------ Heavy - Heavy ------
        double da = dd[ni], db = dd[nj];
        double qa = qq[ni] * td, qb = qq[nj] * td;

        double ade = SQR(half / ad[ni] + half / am[nj]);
        double aqe = SQR(half / aq[ni] + half / am[nj]);
        double aed = SQR(half / am[ni] + half / ad[nj]);
        double aeq = SQR(half / am[ni] + half / aq[nj]);
        double axx = SQR(half / ad[ni] + half / ad[nj]);
        double adq = SQR(half / ad[ni] + half / aq[nj]);
        double aqd = SQR(half / aq[ni] + half / ad[nj]);
        double aqq = SQR(half / aq[ni] + half / aq[nj]);

        double arg1  = SQR(r + da) + ade;
        double arg2  = SQR(r - da) + ade;
        double arg3  = SQR(r - qa) + aqe;
        double arg4  = SQR(r + qa) + aqe;
        double arg5  = rsq + aqe;
        double arg6  = arg5 + qa * qa;

        double arg7  = SQR(r - db) + aed;
        double arg8  = SQR(r + db) + aed;
        double arg9  = SQR(r - qb) + aeq;
        double arg10 = SQR(r + qb) + aeq;
        double arg11 = rsq + aeq;
        double arg12 = arg11 + qb * qb;

        double arg13 = rsq + axx + SQR(da - db);
        double arg14 = rsq + axx + SQR(da + db);
        double arg15 = SQR(r + da - db) + axx;
        double arg16 = SQR(r - da + db) + axx;
        double arg17 = SQR(r - da - db) + axx;
        double arg18 = SQR(r + da + db) + axx;

        double arg19 = SQR(r + da) + adq;
        double arg20 = arg19 + qb * qb;
        double arg21 = SQR(r - da) + adq;
        double arg22 = arg21 + qb * qb;

        double arg23 = SQR(r - db) + aqd;
        double arg24 = arg23 + qa * qa;
        double arg25 = SQR(r + db) + aqd;
        double arg26 = arg25 + qa * qa;

        double arg27 = SQR(r + da - qb) + adq;
        double arg28 = SQR(r - da - qb) + adq;
        double arg29 = SQR(r + da + qb) + adq;
        double arg30 = SQR(r - da + qb) + adq;

        double arg31 = SQR(r + qa - db) + aqd;
        double arg32 = SQR(r + qa + db) + aqd;
        double arg33 = SQR(r - qa - db) + aqd;
        double arg34 = SQR(r - qa + db) + aqd;

        double arg35 = rsq + aqq;
        double arg36 = arg35 + SQR(qa - qb);
        double arg37 = arg35 + SQR(qa + qb);
        double arg38 = arg35 + qa * qa;
        double arg39 = arg35 + qb * qb;
        double arg40 = arg38 + qb * qb;

        double arg41 = SQR(r - qb) + aqq;
        double arg42 = arg41 + qa * qa;
        double arg43 = SQR(r + qb) + aqq;
        double arg44 = arg43 + qa * qa;
        double arg45 = SQR(r + qa) + aqq;
        double arg46 = arg45 + qb * qb;
        double arg47 = SQR(r - qa) + aqq;
        double arg48 = arg47 + qb * qb;

        double arg49 = SQR(r + qa - qb) + aqq;
        double arg50 = SQR(r + qa + qb) + aqq;
        double arg51 = SQR(r - qa - qb) + aqq;
        double arg52 = SQR(r - qa + qb) + aqq;

        double qa0 = qq[ni];
        double qb0 = qq[nj];
        double arg53 = SQR(da - qb0) + SQR(r - qb0) + adq;
        double arg54 = SQR(da - qb0) + SQR(r + qb0) + adq;
        double arg55 = SQR(da + qb0) + SQR(r - qb0) + adq;
        double arg56 = SQR(da + qb0) + SQR(r + qb0) + adq;

        double arg57 = SQR(r + qa0) + SQR(qa0 - db) + aqd;
        double arg58 = SQR(r - qa0) + SQR(qa0 - db) + aqd;
        double arg59 = SQR(r + qa0) + SQR(qa0 + db) + aqd;
        double arg60 = SQR(r - qa0) + SQR(qa0 + db) + aqd;

        double arg64 = SQR(r + qa0 - qb0) + SQR(qa0 - qb0) + aqq;
        double arg65 = SQR(r + qa0 - qb0) + SQR(qa0 + qb0) + aqq;
        double arg66 = SQR(r + qa0 + qb0) + SQR(qa0 - qb0) + aqq;
        double arg67 = SQR(r + qa0 + qb0) + SQR(qa0 + qb0) + aqq;
        double arg68 = SQR(r - qa0 - qb0) + SQR(qa0 - qb0) + aqq;
        double arg69 = SQR(r - qa0 - qb0) + SQR(qa0 + qb0) + aqq;
        double arg70 = SQR(r - qa0 + qb0) + SQR(qa0 - qb0) + aqq;
        double arg71 = SQR(r - qa0 + qb0) + SQR(qa0 + qb0) + aqq;

        double ev1 = HATREE2EV / 2.0;
        double ev2 = HATREE2EV / 4.0;
        double ev3 = HATREE2EV / 8.0;
        double ev4 = HATREE2EV / 16.0;

        double ee     = HATREE2EV / sqrt(rsq + aee_te);
        double dze    = (-ev1 / sqrt(arg1))  + ev1 / sqrt(arg2);
        double qzze   =   ev2 / sqrt(arg3)   + ev2 / sqrt(arg4)   - ev1 / sqrt(arg5);
        double qxxe   =   ev1 / sqrt(arg6)   - ev1 / sqrt(arg5);
        double edz    = (-ev1 / sqrt(arg7))  + ev1 / sqrt(arg8);
        double eqzz   =   ev2 / sqrt(arg9)   + ev2 / sqrt(arg10)  - ev1 / sqrt(arg11);
        double eqxx   =   ev1 / sqrt(arg12)  - ev1 / sqrt(arg11);
        double dxdx   =   ev1 / sqrt(arg13)  - ev1 / sqrt(arg14);
        double dzdz   =   ev2 / sqrt(arg15)  + ev2 / sqrt(arg16)  - ev2 / sqrt(arg17) - ev2 / sqrt(arg18);
        double dzqxx  =   ev2 / sqrt(arg19)  - ev2 / sqrt(arg20)  - ev2 / sqrt(arg21) + ev2 / sqrt(arg22);
        double qxxdz  =   ev2 / sqrt(arg23)  - ev2 / sqrt(arg24)  - ev2 / sqrt(arg25) + ev2 / sqrt(arg26);
        double dzqzz  = (-ev3 / sqrt(arg27)) + ev3 / sqrt(arg28)  - ev3 / sqrt(arg29) + ev3 / sqrt(arg30)
                        - ev2 / sqrt(arg21)  + ev2 / sqrt(arg19);
        double qzzdz  = (-ev3 / sqrt(arg31)) + ev3 / sqrt(arg32)  - ev3 / sqrt(arg33) + ev3 / sqrt(arg34)
                        + ev2 / sqrt(arg23)  - ev2 / sqrt(arg25);
        double qxxqxx =   ev3 / sqrt(arg36)  + ev3 / sqrt(arg37)  - ev2 / sqrt(arg38) - ev2 / sqrt(arg39) + ev2 / sqrt(arg35);
        double qxxqyy =   ev2 / sqrt(arg40)  - ev2 / sqrt(arg38)  - ev2 / sqrt(arg39) + ev2 / sqrt(arg35);
        double qxxqzz =   ev3 / sqrt(arg42)  + ev3 / sqrt(arg44)  - ev3 / sqrt(arg41) - ev3 / sqrt(arg43) - ev2 / sqrt(arg38) + ev2 / sqrt(arg35);
        double qzzqxx =   ev3 / sqrt(arg46)  + ev3 / sqrt(arg48)  - ev3 / sqrt(arg45) - ev3 / sqrt(arg47) - ev2 / sqrt(arg39) + ev2 / sqrt(arg35);
        double qzzqzz =   ev4 / sqrt(arg49)  + ev4 / sqrt(arg50)  + ev4 / sqrt(arg51) + ev4 / sqrt(arg52)
                        - ev3 / sqrt(arg47)  - ev3 / sqrt(arg45)  - ev3 / sqrt(arg41)  - ev3 / sqrt(arg43) + ev2 / sqrt(arg35);
        double dxqxz  = (-ev2 / sqrt(arg53)) + ev2 / sqrt(arg54)  + ev2 / sqrt(arg55) - ev2 / sqrt(arg56);
        double qxzdx  = (-ev2 / sqrt(arg57)) + ev2 / sqrt(arg58)  + ev2 / sqrt(arg59) - ev2 / sqrt(arg60);
        double qxzqxz =   ev3 / sqrt(arg64)  - ev3 / sqrt(arg66)  - ev3 / sqrt(arg68) + ev3 / sqrt(arg70)
                        - ev3 / sqrt(arg65)  + ev3 / sqrt(arg67)  + ev3 / sqrt(arg69) - ev3 / sqrt(arg71);

        ri[0]  = ee;      ri[1]  = dze;             ri[2]  = ee + qzze; ri[3]  = ee + qxxe;
        ri[4]  = edz;     ri[5]  = dzdz;            ri[6]  = dxdx;      ri[7]  = edz + qzzdz;
        ri[8]  = edz + qxxdz; ri[9]  = qxzdx;       ri[10] = ee + eqzz; ri[11] = ee + eqxx;
        ri[12] = dze + dzqzz; ri[13] = dze + dzqxx; ri[14] = dxqxz;
        ri[15] = ee + eqzz + qzze + qzzqzz; ri[16] = ee + eqzz + qxxe + qxxqzz;
        ri[17] = ee + eqxx + qzze + qzzqxx; ri[18] = ee + eqxx + qxxe + qxxqxx;
        ri[19] = qxzqxz; ri[20] = ee + eqxx + qxxe + qxxqyy; ri[21] = half * (qxxqxx - qxxqyy);
    }
}


// (Computes the first 4 terms of the 10x2 electron-core repulsion matrix)
// TODO: this should be splitted into hcore part.
__device__ void spcore_device(
    int ni, int nj, double r, 
    const double* __restrict__ po_tensor, 
    const double* __restrict__ ddp_tensor, 
    const double* __restrict__ core_rho, 
    const double* __restrict__ tore, 
    int ele_i, int ele_j,                   // Global element indices to check heavy_atom
    int n_atom,                             // Number of atoms in molecule
    const double HATREE2EV,
    double* __restrict__ core
) {
    double r2 = r * r;

    double aci = core_rho[ni];
    double acj = core_rho[nj];

    // Extracting am[ni] equivalent (po_ss)
    double po_ss_ni = po_tensor[0 * (9 * n_atom) + 0 * (3 * n_atom) + 0 * n_atom + ni];
    double po_ss_nj = po_tensor[0 * (9 * n_atom) + 0 * (3 * n_atom) + 0 * n_atom + nj];

    double ssi = SQR(aci + po_ss_nj);
    double ssj = SQR(acj + po_ss_ni);

    core[0 * 2 + 0] = -tore[nj] * HATREE2EV / sqrt(r2 + ssj);
    core[0 * 2 + 1] = -tore[ni] * HATREE2EV / sqrt(r2 + ssi);

    bool heavy_i = (ele_i >= 3);
    bool heavy_j = (ele_j >= 3);

    if (heavy_i) {
        double po6_pp0_ni = po_tensor[1 * (9 * n_atom) + 1 * (3 * n_atom) + 0 * n_atom + ni]; 
        double ppj = SQR(acj + po6_pp0_ni);
        
        double da = ddp_tensor[0 * (3 * n_atom) + 1 * n_atom + ni];
        
        double qa = ddp_tensor[1 * (3 * n_atom) + 1 * n_atom + ni] * 0.7071067811865475;
        double twoqa = 2.0 * qa;
        
        double po1_sp_ni = po_tensor[0 * (9 * n_atom) + 1 * (3 * n_atom) + 1 * n_atom + ni];
        double po2_pp2_ni = po_tensor[1 * (9 * n_atom) + 1 * (3 * n_atom) + 2 * n_atom + ni];

        double adj = SQR(po1_sp_ni + acj);
        double aqj = SQR(po2_pp2_ni + acj);

        double x0 =  1.0  / sqrt(r2 + ppj);
        double x1 = -0.5  / sqrt(r2 + aqj);
        double x2 = -0.5  / sqrt(SQR(r + da) + adj);
        double x3 =  0.5  / sqrt(SQR(r - da) + adj);
        double x4 =  0.25 / sqrt(SQR(r - twoqa) + aqj);
        double x5 =  0.25 / sqrt(SQR(r + twoqa) + aqj);
        double x6 =  0.5  / sqrt(r2 + twoqa * twoqa + aqj);

        core[1 * 2 + 0] = -tore[nj] * (x2 + x3) * HATREE2EV;
        core[2 * 2 + 0] = -tore[nj] * (x0 + x1 + x4 + x5) * HATREE2EV;
        core[3 * 2 + 0] = -tore[nj] * (x0 + x1 + x6) * HATREE2EV;
    }

    if (heavy_j) {
        double po_pp0_nj = po_tensor[1 * (9 * n_atom) + 1 * (3 * n_atom) + 0 * n_atom + nj];
        double ppi = SQR(aci + po_pp0_nj);
        
        double db = ddp_tensor[0 * (3 * n_atom) + 1 * n_atom + nj];
        double qb = ddp_tensor[1 * (3 * n_atom) + 1 * n_atom + nj] * 0.7071067811865475;
        double twoqb = 2.0 * qb;
        
        double po_sp_nj = po_tensor[0 * (9 * n_atom) + 1 * (3 * n_atom) + 1 * n_atom + nj];
        double po_pp2_nj = po_tensor[1 * (9 * n_atom) + 1 * (3 * n_atom) + 2 * n_atom + nj];

        double adi = SQR(po_sp_nj + aci);
        double aqi = SQR(po_pp2_nj + aci);

        double xi0 =  1.0  / sqrt(r2 + ppi);
        double xi1 = -0.5  / sqrt(r2 + aqi);
        double xi2 = -0.5  / sqrt(SQR(r + db) + adi);
        double xi3 =  0.5  / sqrt(SQR(r - db) + adi);
        double xi4 =  0.25 / sqrt(SQR(r - twoqb) + aqi);
        double xi5 =  0.25 / sqrt(SQR(r + twoqb) + aqi);
        double xi6 =  0.5  / sqrt(r2 + twoqb * twoqb + aqi);

        // Core attraction on atom nj from atom ni's core
        core[1 * 2 + 1] = tore[ni] * (xi2 + xi3) * HATREE2EV;
        core[2 * 2 + 1] = -tore[ni] * (xi0 + xi1 + xi4 + xi5) * HATREE2EV;
        core[3 * 2 + 1] = -tore[ni] * (xi0 + xi1 + xi6) * HATREE2EV;
    }
}


// GLOBAL TRANSFORMATION KERNELS
// Replaces MOPAC's rotmat, tx, w2mat, elenuc, and ccrep with a unified 
// tensor contraction approach on GPU.
__device__ void compute_rotmat(
    double xi, double yi, double zi,
    double xj, double yj, double zj,
    double p[3][3], double d[5][5]
) {
    double dx = xj - xi;
    double dy = yj - yi;
    double dz = zj - zi;
    double b = dx * dx + dy * dy;
    double r = sqrt(b + dz * dz);
    double sqb = (b > 0.0) ? sqrt(b) : 0.0;
    double sb = (r > 0.0) ? sqb / r : 0.0;
    
    double ca = 0.0, sa = 0.0, cb = 0.0;
    if (sb > 1.0e-7) {
        ca = dx / sqb; 
        sa = dy / sqb; 
        cb = dz / r;
    } else {
        if (dz < 0.0) { ca = -1.0; cb = -1.0; }
        else if (dz > 0.0) { ca = 1.0; cb = 1.0; }
    }
    
    // P-orbital rotation matrix (3x3)
    p[0][0] = ca * sb; p[0][1] = sa * sb; p[0][2] = cb;
    p[1][0] = ca * cb; p[1][1] = sa * cb; p[1][2] = -sb;
    p[2][0] = -sa;     p[2][1] = ca;      p[2][2] = 0.0;
    
    // D-orbital rotation matrix (5x5)
    double pt5sq3 = 0.8660254037841; // sqrt(3)/2
    double c2a = 2.0 * ca * ca - 1.0;
    double c2b = 2.0 * cb * cb - 1.0;
    double s2a = 2.0 * sa * ca;
    double s2b = 2.0 * sb * cb;
    
    d[0][0] = pt5sq3 * c2a * sb * sb; d[1][0] = 0.5 * c2a * s2b;       d[2][0] = -s2a * sb;     d[3][0] = c2a * (cb * cb + 0.5 * sb * sb); d[4][0] = -s2a * cb;
    d[0][1] = pt5sq3 * ca * s2b;      d[1][1] = ca * c2b;              d[2][1] = -sa * cb;      d[3][1] = -0.5 * ca * s2b;                 d[4][1] = sa * sb;
    d[0][2] = cb * cb - 0.5 * sb * sb;d[1][2] = -pt5sq3 * s2b;         d[2][2] = 0.0;           d[3][2] = pt5sq3 * sb * sb;                d[4][2] = 0.0;
    d[0][3] = pt5sq3 * sa * s2b;      d[1][3] = sa * c2b;              d[2][3] = ca * cb;       d[3][3] = -0.5 * sa * s2b;                 d[4][3] = -ca * sb;
    d[0][4] = pt5sq3 * s2a * sb * sb; d[1][4] = 0.5 * s2a * s2b;       d[2][4] = c2a * sb;      d[3][4] = s2a * (cb * cb + 0.5 * sb * sb); d[4][4] = c2a * cb;
}


// Maps 0-based orbital pair (i, j) to 0-based flattened index (0..44)
__device__ inline int get_indexd(int i, int j) {
    return (i * (i + 1)) / 2 + j;
}

__device__ void build_pair_rotation_matrix(double R[45][45], const double p[3][3], const double d[5][5]) {
    // Initialize and clear the matrix
    for(int i=0; i<45; ++i) {
        for(int j=0; j<45; ++j) {
            R[i][j] = 0.0;
        }
    }
    
    // SS-block (mm=1)
    R[get_indexd(0, 0)][get_indexd(0, 0)] = 1.0;
    
    // SP-block (mm=2)
    // In mopac: sp[k, I] = p[k, I] -> p[Local, Global]
    for(int k=0; k<3; ++k) {
        for(int I=0; I<3; ++I) {
            R[get_indexd(I+1, 0)][get_indexd(k+1, 0)] = p[k][I];
        }
    }
    
    // PP-block (mm=3)
    for(int k=0; k<3; ++k) {
        for(int l=0; l<=k; ++l) {
            int kl = get_indexd(k+1, l+1); // Local pair
            if (k == l) {
                // p[Local][Global] rules: 0=X, 1=Y, 2=Z
                R[get_indexd(1,1)][kl] = p[k][0] * p[k][0];
                R[get_indexd(2,2)][kl] = p[k][1] * p[k][1];
                R[get_indexd(3,3)][kl] = p[k][2] * p[k][2];
                R[get_indexd(2,1)][kl] = p[k][0] * p[k][1];
                R[get_indexd(3,1)][kl] = p[k][0] * p[k][2];
                R[get_indexd(3,2)][kl] = p[k][1] * p[k][2];
            } else {
                R[get_indexd(1,1)][kl] = 2.0 * p[k][0] * p[l][0];
                R[get_indexd(2,2)][kl] = 2.0 * p[k][1] * p[l][1];
                R[get_indexd(3,3)][kl] = 2.0 * p[k][2] * p[l][2];
                R[get_indexd(2,1)][kl] = p[k][0] * p[l][1] + p[k][1] * p[l][0];
                R[get_indexd(3,1)][kl] = p[k][0] * p[l][2] + p[k][2] * p[l][0];
                R[get_indexd(3,2)][kl] = p[k][1] * p[l][2] + p[k][2] * p[l][1];
            }
        }
    }
    
    // SD-block (mm=4)
    for(int k=0; k<5; ++k) {
        for(int I=0; I<5; ++I) {
            R[get_indexd(I+4, 0)][get_indexd(k+4, 0)] = d[k][I];
        }
    }
    
    // DP-block (mm=5)
    for(int k=0; k<5; ++k) {
        for(int l=0; l<3; ++l) {
            int kl = get_indexd(k+4, l+1);
            for(int I=0; I<5; ++I) {
                for(int J=0; J<3; ++J) {
                    R[get_indexd(I+4, J+1)][kl] = d[k][I] * p[l][J];
                }
            }
        }
    }
    
    // DD-block (mm=6)
    for(int k=0; k<5; ++k) {
        for(int l=0; l<=k; ++l) {
            int kl = get_indexd(k+4, l+4);
            if (k == l) {
                for(int I=0; I<5; ++I) {
                    for(int J=0; J<=I; ++J) {
                        R[get_indexd(I+4, J+4)][kl] = d[k][I] * d[k][J];
                    }
                }
            } else {
                for(int I=0; I<5; ++I) {
                    for(int J=0; J<=I; ++J) {
                        R[get_indexd(I+4, J+4)][kl] = d[k][I] * d[l][J] + d[k][J] * d[l][I];
                    }
                }
            }
        }
    }
}


// PM6 Core-Core Repulsion (ccrep_pm6)
// TODO: this should be moved to a seperate cu file
__device__ double ccrep_pm6_device(
    int ele_i, int ele_j,  // Element indices (1-based)
    int ni, int nj, 
    int p_idx, 
    double r_angstrom, 
    double gab,
    const double* tore,     // (natom)
    const double* xfac,     // (npair)
    const double* alpb,     // (npair)
    const double* guess1,   // (natom, 4)
    const double* guess2,   // (natom, 4)
    const double* guess3,   // (natom, 4)
    // TODO: v_par6 only the first 4 terms are used, this can be simplifiled
    const double* v_par6    // (60)
) {
    double enuc = tore[ni] * tore[nj] * gab;
    
    double fff = xfac[p_idx];
    bool has_bond = fabs(fff) > 1e-5;
    double enuclr = 0.0;
    double abond = alpb[p_idx];
    
    if (has_bond) {
        if (abond < 1e-6) abond = 1.2;
        
        double r2 = r_angstrom * r_angstrom;
        double r6 = r2 * r2 * r2;
        double scale = 1.0 + 2.0 * fff * exp(-abond * (r_angstrom + 0.0003 * r6));
        
        int i_big = max(ele_i, ele_j);
        int j_small = min(ele_i, ele_j);
        
        if (j_small == 1) { // H-X bonds
            if (i_big == 6 || i_big == 7) { // C or N
                scale = 1.0 + 2.0 * fff * exp(-abond * r2);
            } else if (i_big == 8) { // O
                scale = 1.0 + 2.0 * fff * exp(-abond * r2) - v_par6[2] * exp(-2.0 * v_par6[3] * r_angstrom);
            }
        }
        if (j_small == 6 && i_big == 6) scale += v_par6[0] * exp(-v_par6[1] * r_angstrom); // C-C
        if (j_small == 8 && i_big == 14) scale -= 0.7e-3 * exp(-SQR(r_angstrom - 2.9)); // O-Si
        
        enuclr = enuc * scale;
    } else {
        bool in_f_block = (ele_i >= 57 && ele_i <= 71) || (ele_j >= 57 && ele_j <= 71);
        double k = in_f_block ? 3.0 : 2.18;
        double scale = 10.0 * exp(-k * r_angstrom);
        enuclr = fabs(scale * enuc) + enuc;
    }
    
    // VdW using Gaussian correction
    double scale_vdw = 0.0;
    double invr = 1.0 / fmax(r_angstrom, 1e-12);
    
    double ax_i = guess2[ni * 4 + 0] * SQR(r_angstrom - guess3[ni * 4 + 0]);
    if (ax_i < 25.0) scale_vdw += tore[ni] * tore[nj] * invr * guess1[ni * 4 + 0] * exp(-ax_i);
    
    double ax_j = guess2[nj * 4 + 0] * SQR(r_angstrom - guess3[nj * 4 + 0]);
    if (ax_j < 25.0) scale_vdw += tore[ni] * tore[nj] * invr * guess1[nj * 4 + 0] * exp(-ax_j);
    
    int i_max = (has_bond && abond > 1e-4) ? 0 : 4;
    for (int ig = 0; ig < i_max; ++ig) {
        double g1_i = guess1[ni * 4 + ig];
        if (g1_i != 0.0) {
            double ax = guess2[ni * 4 + ig] * SQR(r_angstrom - guess3[ni * 4 + ig]);
            if (ax <= 25.0) scale_vdw += tore[ni] * tore[nj] * invr * g1_i * exp(-ax);
        }
        double g1_j = guess1[nj * 4 + ig];
        if (g1_j != 0.0) {
            double ax = guess2[nj * 4 + ig] * SQR(r_angstrom - guess3[nj * 4 + ig]);
            if (ax <= 25.0) scale_vdw += tore[ni] * tore[nj] * invr * g1_j * exp(-ax);
        }
    }
    
    enuclr += scale_vdw;
    
    // Short distance repulsion
    double zi = pow(ele_i, 0.3333); //follow mopac the 1/3 is set to 0.3333
    double zj = pow(ele_j, 0.3333); //follow mopac the 1/3 is set to 0.3333
    // double zi = cbrt((double)ele_i); 
    // double zj = cbrt((double)ele_j); 
    double ax = r_angstrom / (zi + zj);

    if (ax < 3.0) {
        double ax3 = ax * ax * ax;
        double ax6 = ax3 * ax3;
        double ax12 = ax6 * ax6;
        double lj12 = 1.0e-8 / ax12;
        enuclr += fmin(lj12, 1.0e5);
    }
    return enuclr;
}


// this function only used for debug!
__global__ void multipole_eval_kernel(
    const int n_pairs,
    const double* __restrict__ r_vec,   // (n_pairs,)
    const int* __restrict__ l1_vec,     // (n_pairs,)
    const int* __restrict__ l2_vec,     // (n_pairs,)
    const int* __restrict__ m_vec,      // (n_pairs,)
    const double* __restrict__ da_vec,  // (n_pairs,)
    const double* __restrict__ db_vec,  // (n_pairs,)
    const double* __restrict__ add_vec, // (n_pairs,)
    double* __restrict__ out_vec        // (n_pairs,)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs) return;

    out_vec[idx] = charg_kernel_device(
        r_vec[idx],
        l1_vec[idx],
        l2_vec[idx],
        m_vec[idx],
        da_vec[idx],
        db_vec[idx],
        add_vec[idx]
    );
}

__global__ void solve_poij_kernel(
    const int n_atoms,
    const int* __restrict__ l_vec,     // (N,)
    const double* __restrict__ d_vec,  // (N,)
    const double* __restrict__ fg_vec, // (N,)
    double* __restrict__ rho_vec,      // (N,) Output
    const double hartree2ev            // Constant passed from Python
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_atoms) return;

    int l = l_vec[idx];
    double d = d_vec[idx];
    double fg = fg_vec[idx];

    const int niter = 100;
    const double epsil = 1e-8;
    const double g1 = 0.382;
    const double g2 = 0.618;
    
    const double ev4 = hartree2ev / 4.0;
    const double ev8 = hartree2ev / 8.0;

    if (l == 0) {
        if (fabs(fg) < 1e-12) {
            rho_vec[idx] = 1e6;
        } else {
            rho_vec[idx] = 0.5 * hartree2ev / fg;
        }
        return;
    }

    double dsq = d * d;
    double a1 = 0.1;
    double a2 = 5.0;
    double f1 = 0.0;
    double f2 = 0.0;
    
    for (int i = 0; i < niter; ++i) {
        double delta = a2 - a1;
        if (delta < epsil) {
            break;
        }

        double y1 = a1 + delta * g1;
        double y2 = a1 + delta * g2;

        if (l == 1) {
            double term1_y1 = ev4 * (1.0/y1 - INV_SQRT(y1*y1 + dsq));
            double term1_y2 = ev4 * (1.0/y2 - INV_SQRT(y2*y2 + dsq));
            f1 = (term1_y1 - fg) * (term1_y1 - fg);
            f2 = (term1_y2 - fg) * (term1_y2 - fg);
        } else {
            double t1_y1 = INV_SQRT(y1*y1 + dsq * 0.5);
            double t2_y1 = INV_SQRT(y1*y1 + dsq);
            double term_y1 = ev8 * (1.0/y1 - 2.0 * t1_y1 + t2_y1);

            double t1_y2 = INV_SQRT(y2*y2 + dsq * 0.5);
            double t2_y2 = INV_SQRT(y2*y2 + dsq);
            double term_y2 = ev8 * (1.0/y2 - 2.0 * t1_y2 + t2_y2);

            f1 = (term_y1 - fg) * (term_y1 - fg);
            f2 = (term_y2 - fg) * (term_y2 - fg);
        }

        if (f1 < f2) {
            a2 = y2;
        } else {
            a1 = y1;
        }
    }

    if (f1 >= f2) {
        rho_vec[idx] = a2;
    } else {
        rho_vec[idx] = a1;
    }
}


// this function only used for debug!
__global__ void test_rijkl_kernel(
    int n_tasks, 
    int n_atom,
    const int* __restrict__ ni_vec, const int* __restrict__ nj_vec,
    const int* __restrict__ ij_vec, const int* __restrict__ kl_vec,
    const int* __restrict__ li_vec, const int* __restrict__ lj_vec,
    const int* __restrict__ lk_vec, const int* __restrict__ ll_vec,
    const int* __restrict__ ic_vec, const double* __restrict__ r_vec,
    const double* __restrict__ po_tensor,
    const double* __restrict__ ddp_tensor,
    const double* __restrict__ core_rho,
    const double* __restrict__ ch,
    double* __restrict__ out_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tasks) return;

    out_val[idx] = rijkl_device(
        ni_vec[idx], nj_vec[idx], 
        ij_vec[idx], kl_vec[idx],
        li_vec[idx], lj_vec[idx], 
        lk_vec[idx], ll_vec[idx],
        ic_vec[idx], r_vec[idx], 
        n_atom,
        po_tensor, ddp_tensor, core_rho, ch
    );
}


// Ultimate scheduling kernel: 1 Block = 1 atom pair
// Processes the full 491 rep terms, core, and gab
// This functions is subtracted and rewritten from reppd2, reppd, fordd.
__global__ void calc_local_rep_core_kernel(
    int n_pairs,
    const int* __restrict__ pair_i_vec,
    const int* __restrict__ pair_j_vec,
    const int* __restrict__ ele_id,
    const double* __restrict__ r_vec,
    int n_atom,
    const double* __restrict__ am, 
    const double* __restrict__ ad, 
    const double* __restrict__ aq, 
    const double* __restrict__ dd, 
    const double* __restrict__ qq,
    const double* __restrict__ po_tensor,
    const double* __restrict__ ddp_tensor,
    const double* __restrict__ core_rho,
    const double* __restrict__ ch,
    const double* __restrict__ tore,
    const int* __restrict__ natorb,
    const bool* __restrict__ dorbs,
    // 8 1D arrays containing parallel instructions (length 491, from fordd)
    const int* __restrict__ task_action,
    const int* __restrict__ task_target,
    const int* __restrict__ task_ij,
    const int* __restrict__ task_kl,
    const int* __restrict__ task_li,
    const int* __restrict__ task_lj,
    const int* __restrict__ task_lk,
    const int* __restrict__ task_ll,
    const double HATREE2EV,
    // Output arrays
    double* __restrict__ rep_out,   // (n_pairs, 491)
    double* __restrict__ core_out,  // (n_pairs, 10, 2)
    double* __restrict__ gab_out    // (n_pairs)
) {
    int p_idx = blockIdx.x;
    if (p_idx >= n_pairs) return;

    int ni = pair_i_vec[p_idx]; // Atom index
    int nj = pair_j_vec[p_idx]; // Atom index
    double r = r_vec[p_idx];

    int e_i = ele_id[ni]; // Element index
    int e_j = ele_id[nj]; // Element index

    __shared__ double s_ri[22];     // sp parts
    __shared__ double s_rep[491];   // spd parts
    __shared__ double s_core[20];   // 10 rows, 2 cols
    __shared__ double s_gab;

    int tid = threadIdx.x;

    if (tid < 20) s_core[tid] = 0.0;
    for (int t = tid; t < 491; t += blockDim.x) s_rep[t] = 0.0;
    __syncthreads();

    // Thread 0 handles serial computation of prerequisite physical quantities
    if (tid == 0) {
        reppd_device(ni, nj, r, am, ad, aq, dd, qq, core_rho, natorb[ni], natorb[nj], HATREE2EV, s_ri, s_gab);
        spcore_device(ni, nj, r, po_tensor, ddp_tensor, core_rho, tore, e_i, e_j, n_atom, HATREE2EV, s_core);

        // Compute d-orbital electron-core repulsion (fills rows 4~9 of the core matrix)
        if (dorbs[nj]) {
            s_core[4 * 2 + 1] = -rijkl_device(ni, nj, 0,  4,  0,0, 2,0, 1, r, n_atom, po_tensor, ddp_tensor, core_rho, ch) * HATREE2EV * tore[ni]; // <S S | D S>
            s_core[5 * 2 + 1] = -rijkl_device(ni, nj, 0, 12,  0,0, 2,1, 1, r, n_atom, po_tensor, ddp_tensor, core_rho, ch) * HATREE2EV * tore[ni]; // <S S | D P>
            s_core[6 * 2 + 1] = -rijkl_device(ni, nj, 0, 30,  0,0, 2,2, 1, r, n_atom, po_tensor, ddp_tensor, core_rho, ch) * HATREE2EV * tore[ni]; // <S S | D D>
            s_core[7 * 2 + 1] = -rijkl_device(ni, nj, 0, 20,  0,0, 2,1, 1, r, n_atom, po_tensor, ddp_tensor, core_rho, ch) * HATREE2EV * tore[ni]; // <S S | D+ P+>
            s_core[8 * 2 + 1] = -rijkl_device(ni, nj, 0, 35,  0,0, 2,2, 1, r, n_atom, po_tensor, ddp_tensor, core_rho, ch) * HATREE2EV * tore[ni]; // <S S | D+ D+>
            s_core[9 * 2 + 1] = -rijkl_device(ni, nj, 0, 42,  0,0, 2,2, 1, r, n_atom, po_tensor, ddp_tensor, core_rho, ch) * HATREE2EV * tore[ni]; // <S S | D# D#>
        }
        if (dorbs[ni]) {
            s_core[4 * 2 + 0] = -rijkl_device(ni, nj,  4, 0,  2,0, 0,0, 2, r, n_atom, po_tensor, ddp_tensor, core_rho, ch) * HATREE2EV * tore[nj];
            s_core[5 * 2 + 0] = -rijkl_device(ni, nj, 12, 0,  2,1, 0,0, 2, r, n_atom, po_tensor, ddp_tensor, core_rho, ch) * HATREE2EV * tore[nj];
            s_core[6 * 2 + 0] = -rijkl_device(ni, nj, 30, 0,  2,2, 0,0, 2, r, n_atom, po_tensor, ddp_tensor, core_rho, ch) * HATREE2EV * tore[nj];
            s_core[7 * 2 + 0] = -rijkl_device(ni, nj, 20, 0,  2,1, 0,0, 2, r, n_atom, po_tensor, ddp_tensor, core_rho, ch) * HATREE2EV * tore[nj];
            s_core[8 * 2 + 0] = -rijkl_device(ni, nj, 35, 0,  2,2, 0,0, 2, r, n_atom, po_tensor, ddp_tensor, core_rho, ch) * HATREE2EV * tore[nj];
            s_core[9 * 2 + 0] = -rijkl_device(ni, nj, 42, 0,  2,2, 0,0, 2, r, n_atom, po_tensor, ddp_tensor, core_rho, ch) * HATREE2EV * tore[nj];
        }
    }
    __syncthreads();

    // parallel evaluation of 491 terms (Stage 1: Direct computation)
    for (int t = tid; t < 491; t += blockDim.x) {
        int action = task_action[t];
        
        bool valid_i = dorbs[ni] ? true : (task_li[t] == 0 ? true : (task_li[t] <= 1 && e_i >= 3));
        bool valid_j = dorbs[nj] ? true : (task_lk[t] == 0 ? true : (task_lk[t] <= 1 && e_j >= 3));
        
        if (action == 0) {
            // Directly fetch the first 22 main-group results
            s_rep[t] = s_ri[task_target[t]];
        } else if (action == 1 && valid_i && valid_j) {
            s_rep[t] = rijkl_device(
                ni, nj, task_ij[t], task_kl[t], 
                task_li[t], task_lj[t], task_lk[t], task_ll[t], 
                0, r, n_atom, po_tensor, ddp_tensor, core_rho, ch
            ) * HATREE2EV;
        }
    }
    __syncthreads();

    // parallel evaluation of 491 terms (Stage 2: Symmetry copying)
    for (int t = tid; t < 491; t += blockDim.x) {
        int action = task_action[t];
        if (action == 2) {
            s_rep[t] = s_rep[task_target[t]];
        } else if (action == 3) {
            s_rep[t] = -s_rep[task_target[t]];
        }
    }
    __syncthreads();

    // Flush the computed results into Global Memory at once
    for (int t = tid; t < 491; t += blockDim.x) {
        rep_out[p_idx * 491 + t] = s_rep[t];
    }
    if (tid < 20) {
        core_out[p_idx * 20 + tid] = s_core[tid];
    }
    if (tid == 0) {
        gab_out[p_idx] = s_gab;
    }
}

// HARDCODED MAPPINGS
// Dense 1D index (0..44) to 2D orbital index (i)
__device__ const int DENSE_TO_I[45] = {
    0,
    1, 1,
    2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8
};

// Dense 1D index (0..44) to 2D orbital index (j)
__device__ const int DENSE_TO_J[45] = {
    0,
    0, 1,
    0, 1, 2,
    0, 1, 2, 3,
    0, 1, 2, 3, 4,
    0, 1, 2, 3, 4, 5,
    0, 1, 2, 3, 4, 5, 6,
    0, 1, 2, 3, 4, 5, 6, 7,
    0, 1, 2, 3, 4, 5, 6, 7, 8
};

// It is indexd in the mopac
__device__ const int MOPAC_INDEXD[9][9] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8},
    { 1,  9, 10, 11, 12, 13, 14, 15, 16},
    { 2, 10, 17, 18, 19, 20, 21, 22, 23},
    { 3, 11, 18, 24, 25, 26, 27, 28, 29},
    { 4, 12, 19, 25, 30, 31, 32, 33, 34},
    { 5, 13, 20, 26, 31, 35, 36, 37, 38},
    { 6, 14, 21, 27, 32, 36, 39, 40, 41},
    { 7, 15, 22, 28, 33, 37, 40, 42, 43},
    { 8, 16, 23, 29, 34, 38, 41, 43, 44}
};


__global__ void global_transform_kernel(
    int n_pairs,
    const int* __restrict__ pair_i_vec, const int* __restrict__ pair_j_vec,
    const int* __restrict__ ele_id,
    const double* __restrict__ coords, 
    const double* __restrict__ rep_in, 
    const double* __restrict__ core_in,
    const double* __restrict__ gab_in, 
    const int* __restrict__ ind2_arr,  
    const int* __restrict__ natorb,
    const int* __restrict__ kr_offsets,
    const double* __restrict__ tore, const double* __restrict__ xfac, 
    const double* __restrict__ alpb, const double* __restrict__ guess1, 
    const double* __restrict__ guess2, const double* __restrict__ guess3, 
    const double* __restrict__ v_par6,
    const double BOHR,
    double* __restrict__ w_out,        
    double* __restrict__ e1b_out,      
    double* __restrict__ e2a_out,      
    double* __restrict__ enuc_out
) {
    int p_idx = blockIdx.x;
    if (p_idx >= n_pairs) return;

    int tid = threadIdx.x;
    int ni = pair_i_vec[p_idx];
    int nj = pair_j_vec[p_idx];
    int ele_i = ele_id[ni];
    int ele_j = ele_id[nj];

    int ii = natorb[ni];
    int kk = natorb[nj];
    int limij = ii * (ii + 1) / 2;
    int limkl = kk * (kk + 1) / 2;

    __shared__ double s_R[45][45];
    __shared__ double s_V[45][45];
    __shared__ double s_L_A[45];
    __shared__ double s_L_B[45];

    if (tid == 0) {
        double xi = coords[ni * 3 + 0], yi = coords[ni * 3 + 1], zi = coords[ni * 3 + 2];
        double xj = coords[nj * 3 + 0], yj = coords[nj * 3 + 1], zj = coords[nj * 3 + 2];
        
        double p[3][3] = {0}, d[5][5] = {0};
        compute_rotmat(xi, yi, zi, xj, yj, zj, p, d);
        build_pair_rotation_matrix(s_R, p, d);

        for(int i=0; i<45; i++) { s_L_A[i] = 0.0; s_L_B[i] = 0.0; }
        
        s_L_A[0] = core_in[p_idx * 20 + 0*2 + 0];  
        s_L_B[0] = core_in[p_idx * 20 + 0*2 + 1]; 
        
        if (ii >= 4) { 
            s_L_A[1] = core_in[p_idx * 20 + 1*2 + 0]; s_L_A[2] = core_in[p_idx * 20 + 2*2 + 0];
            s_L_A[5] = core_in[p_idx * 20 + 3*2 + 0]; s_L_A[9] = core_in[p_idx * 20 + 3*2 + 0];
        }
        if (kk >= 4) { 
            s_L_B[1] = core_in[p_idx * 20 + 1*2 + 1]; s_L_B[2] = core_in[p_idx * 20 + 2*2 + 1];
            s_L_B[5] = core_in[p_idx * 20 + 3*2 + 1]; s_L_B[9] = core_in[p_idx * 20 + 3*2 + 1];
        }
        if (ii >= 9) { 
            s_L_A[10] = core_in[p_idx * 20 + 4*2 + 0]; s_L_A[11] = core_in[p_idx * 20 + 5*2 + 0];
            s_L_A[17] = core_in[p_idx * 20 + 7*2 + 0]; s_L_A[24] = core_in[p_idx * 20 + 7*2 + 0];
            s_L_A[14] = core_in[p_idx * 20 + 6*2 + 0]; s_L_A[20] = core_in[p_idx * 20 + 8*2 + 0];
            s_L_A[27] = core_in[p_idx * 20 + 8*2 + 0]; s_L_A[35] = core_in[p_idx * 20 + 9*2 + 0];
            s_L_A[44] = core_in[p_idx * 20 + 9*2 + 0];
        }
        if (kk >= 9) { 
            s_L_B[10] = core_in[p_idx * 20 + 4*2 + 1]; s_L_B[11] = core_in[p_idx * 20 + 5*2 + 1];
            s_L_B[17] = core_in[p_idx * 20 + 7*2 + 1]; s_L_B[24] = core_in[p_idx * 20 + 7*2 + 1];
            s_L_B[14] = core_in[p_idx * 20 + 6*2 + 1]; s_L_B[20] = core_in[p_idx * 20 + 8*2 + 1];
            s_L_B[27] = core_in[p_idx * 20 + 8*2 + 1]; s_L_B[35] = core_in[p_idx * 20 + 9*2 + 1];
            s_L_B[44] = core_in[p_idx * 20 + 9*2 + 1];
        }
        
        double dx = xj - xi, dy = yj - yi, dz = zj - zi;
        double r_bohr = sqrt(dx*dx + dy*dy + dz*dz);
        double gab = gab_in[p_idx];
        double r_angstrom = r_bohr * BOHR;
        enuc_out[p_idx] = ccrep_pm6_device(ele_i, ele_j, ni, nj, p_idx, r_angstrom, 
            gab, tore, xfac, alpb, guess1, guess2, guess3, v_par6);
    }
    __syncthreads();

    // if (R_out != nullptr) {
    //     for (int idx = tid; idx < 45 * 45; idx += blockDim.x) {
    //         int row = idx / 45;
    //         int col = idx % 45;
    //         R_out[p_idx * 2025 + idx] = s_R[row][col];
    //     }
    // }
    // __syncthreads();

    // ---------------------------------------------------------
    // Tensor Contraction 1
    // ---------------------------------------------------------
    for (int idx = tid; idx < limij * limkl; idx += blockDim.x) {
        int ij = idx / limkl;
        int KL = idx % limkl;
        
        // Direct lookup for local coordinates (i1, j1)
        int i1 = DENSE_TO_I[ij];
        int j1 = DENSE_TO_J[ij];
        int ij_mopac = MOPAC_INDEXD[i1][j1];

        double v_val = 0.0;
        for (int kl = 0; kl < limkl; ++kl) {
            // Direct lookup for local coordinates (k1, l1)
            int k1 = DENSE_TO_I[kl];
            int l1 = DENSE_TO_J[kl];
            
            // Direct lookup in MOPAC's mapping matrix
            int kl_mopac = MOPAC_INDEXD[k1][l1];
            
            // Query the pure index without forced symmetry
            int rep_idx = ind2_arr[ij_mopac * 45 + kl_mopac];
            
            double wrepp = (rep_idx != -1) ? rep_in[p_idx * 491 + rep_idx] : 0.0;
            v_val += wrepp * s_R[KL][kl];
        }
        s_V[ij][KL] = v_val;
    }
    __syncthreads();

    // ---------------------------------------------------------
    // Tensor Contraction 2
    // ---------------------------------------------------------
    int kr = kr_offsets[p_idx];
    for (int idx = tid; idx < limij * limkl; idx += blockDim.x) {
        int IJ = idx / limkl;
        int KL = idx % limkl;
        double w_val = 0.0;
        
        for (int ij = 0; ij < limij; ++ij) {
            w_val += s_R[IJ][ij] * s_V[ij][KL];
        }
        w_out[kr + IJ * limkl + KL] = w_val;
    }
    
    // ---------------------------------------------------------
    // Transform Elenuc Integrals
    // ---------------------------------------------------------
    for (int IJ = tid; IJ < limij; IJ += blockDim.x) {
        double h_val = 0.0;
        for (int ij = 0; ij < limij; ++ij) h_val += s_R[IJ][ij] * s_L_A[ij];
        e1b_out[p_idx * 45 + IJ] = h_val;
    }
    
    for (int KL = tid; KL < limkl; KL += blockDim.x) {
        double h_val = 0.0;
        for (int kl = 0; kl < limkl; ++kl) h_val += s_R[KL][kl] * s_L_B[kl];
        e2a_out[p_idx * 45 + KL] = h_val;
    }
}


extern "C" {
// this function only used for debug!
void launch_multipole_eval_kernel_c(
    const int n_pairs,
    const double* r_vec,
    const int* l1_vec,
    const int* l2_vec,
    const int* m_vec,
    const double* da_vec,
    const double* db_vec,
    const double* add_vec,
    double* out_vec
) {
    int threads_per_block = 128;
    int blocks_per_grid = (n_pairs + threads_per_block - 1) / threads_per_block;
    multipole_eval_kernel<<<blocks_per_grid, threads_per_block>>>(
        n_pairs, r_vec, l1_vec, l2_vec, m_vec, da_vec, db_vec, add_vec, out_vec
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_solve_poij_kernel_c(
    const int n_atoms,
    const int* l_vec,
    const double* d_vec,
    const double* fg_vec,
    double* rho_vec,
    const double hartree2ev
) {
    int threads_per_block = 128;
    int blocks_per_grid = (n_atoms + threads_per_block - 1) / threads_per_block;

    solve_poij_kernel<<<blocks_per_grid, threads_per_block>>>(
        n_atoms, l_vec, d_vec, fg_vec, rho_vec, hartree2ev
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error (solve_poij): %s\n", cudaGetErrorString(err));
    }
}

void launch_test_rijkl_kernel_c(
    int n_tasks, int n_atom,
    const int* ni_vec, const int* nj_vec,
    const int* ij_vec, const int* kl_vec,
    const int* li_vec, const int* lj_vec,
    const int* lk_vec, const int* ll_vec,
    const int* ic_vec, const double* r_vec,
    const double* po_tensor, const double* ddp_tensor,
    const double* core_rho, const double* ch,
    double* out_val
) {
    int threads = 128;
    int blocks = (n_tasks + threads - 1) / threads;
    
    test_rijkl_kernel<<<blocks, threads>>>(
        n_tasks, n_atom,
        ni_vec, nj_vec, ij_vec, kl_vec,
        li_vec, lj_vec, lk_vec, ll_vec,
        ic_vec, r_vec,
        po_tensor, ddp_tensor, core_rho, ch,
        out_val
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error (test_rijkl_kernel): %s\n", cudaGetErrorString(err));
    }
}

void launch_calc_local_rep_core_kernel_c(
    int n_pairs,
    const int* pair_i_vec, const int* pair_j_vec, const int* ele_id, const double* r_vec,
    int n_atom,
    const double* am, const double* ad, const double* aq, 
    const double* dd, const double* qq,
    const double* po_tensor, const double* ddp_tensor, const double* core_rho, const double* ch,
    const double* tore, const int* natorb, const bool* dorbs,
    const int* task_action, const int* task_target,
    const int* task_ij, const int* task_kl,
    const int* task_li, const int* task_lj,
    const int* task_lk, const int* task_ll,
    const double HATREE2EV,
    double* rep_out, double* core_out, double* gab_out
) {
    int threads = 128;
    int blocks = n_pairs; 
    
    calc_local_rep_core_kernel<<<blocks, threads>>>(
        n_pairs, pair_i_vec, pair_j_vec, ele_id, r_vec, n_atom,
        am, ad, aq, dd, qq,
        po_tensor, ddp_tensor, core_rho, ch,
        tore, natorb, dorbs,
        task_action, task_target, task_ij, task_kl,
        task_li, task_lj, task_lk, task_ll,
        HATREE2EV,
        rep_out, core_out, gab_out
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}

void launch_global_transform_kernel_c(
    int n_pairs,
    const int* pair_i_vec, const int* pair_j_vec, const int* ele_id,
    const double* coords, const double* rep_in, const double* core_in, const double* gab_in,
    const int* ind2_arr, const int* natorb, const int* kr_offsets,
    const double* tore, const double* xfac, const double* alpb, 
    const double* guess1, const double* guess2, const double* guess3, 
    const double* v_par6, const double BOHR,
    double* w_out, double* e1b_out, double* e2a_out, double* enuc_out
) {
    int threads = 128;
    int blocks = n_pairs; 
    
    global_transform_kernel<<<blocks, threads>>>(
        n_pairs, pair_i_vec, pair_j_vec, ele_id, coords,
        rep_in, core_in, gab_in, ind2_arr, natorb, kr_offsets,
        tore, xfac, alpb, guess1, guess2, guess3, v_par6, BOHR,
        w_out, e1b_out, e2a_out, enuc_out
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Launch Error (Global Transform): %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"