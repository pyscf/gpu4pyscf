/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2023 Qiming Sun
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

__global__
static void GINTint2e_get_veff_ip1_kernel0010(GINTEnvVars envs,
                                              JKMatrix jk,
                                              BasisProdOffsets offsets)
{
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;
  double norm = envs.fac;
  double omega = envs.omega;
  int *bas_pair2bra = c_bpcache.bas_pair2bra;
  int *bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];
  int nprim_ij = envs.nprim_ij;
  int nprim_kl = envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

  double* __restrict__ a12 = c_bpcache.a12;
  double* __restrict__ e12 = c_bpcache.e12;
  double* __restrict__ x12 = c_bpcache.x12;
  double* __restrict__ y12 = c_bpcache.y12;
  double* __restrict__ z12 = c_bpcache.z12;
  double * __restrict__ i_exponent = c_bpcache.a1;
  double * __restrict__ j_exponent = c_bpcache.a2;

  int ij, kl;
  int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
  int nbas = c_bpcache.nbas;
  double* __restrict__ bas_x = c_bpcache.bas_coords;
  double* __restrict__ bas_y = bas_x + nbas;
  double* __restrict__ bas_z = bas_y + nbas;

  double gout0_ix = 0;
  double gout1_ix = 0;
  double gout2_ix = 0;
  double gout0_iy = 0;
  double gout1_iy = 0;
  double gout2_iy = 0;
  double gout0_iz = 0;
  double gout1_iz = 0;
  double gout2_iz = 0;
  double gout0_jx = 0;
  double gout1_jx = 0;
  double gout2_jx = 0;
  double gout0_jy = 0;
  double gout1_jy = 0;
  double gout2_jy = 0;
  double gout0_jz = 0;
  double gout1_jz = 0;
  double gout2_jz = 0;

  double xi = bas_x[ish];
  double yi = bas_y[ish];
  double zi = bas_z[ish];
  double ABx = xi - bas_x[jsh];
  double ABy = yi - bas_y[jsh];
  double ABz = zi - bas_z[jsh];
  double xk = bas_x[ksh];
  double yk = bas_y[ksh];
  double zk = bas_z[ksh];

  prim_ij0 = prim_ij;
  prim_ij1 = prim_ij + nprim_ij;
  prim_kl0 = prim_kl;
  prim_kl1 = prim_kl + nprim_kl;
  double rw[4];
  int irys;
  for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    double ai = i_exponent[ij] * 2.0;
    double aj = j_exponent[ij] * 2.0;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
      double akl = a12[kl];
      double ekl = e12[kl];
      double xkl = x12[kl];
      double ykl = y12[kl];
      double zkl = z12[kl];
      double xijxkl = xij - xkl;
      double yijykl = yij - ykl;
      double zijzkl = zij - zkl;
      double aijkl = aij + akl;
      double a1 = aij * akl;
      double a0 = a1 / aijkl;
      double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
      a0 *= theta;
      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      double fac = norm * eij * ekl * sqrt(a0 / (a1 * a1 * a1));
      //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

      GINTrys_root<2>(x, rw);
      GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
        double gz0 = rw[irys+2] * fac;
        double root0 = rw[irys];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double B00 = u2 * tmp4;
        double tmp1 = 2 * B00;

        double tmp2 = tmp1 * akl;
        double C00x = xij - xi - tmp2 * xijxkl;
        double C00y = yij - yi - tmp2 * yijykl;
        double C00z = zij - zi - tmp2 * zijzkl;

        double tmp3 = tmp1 * aij;
        double D00x = xkl - xk + tmp3 * xijxkl;
        double D00y = ykl - yk + tmp3 * yijykl;
        double D00z = zkl - zk + tmp3 * zijzkl;

        double gx0 = 1;
        double gy0 = 1;
        double gx1,gx3,gx4,gx2,gx5;
        gx1 = C00x*gx0;
        gx3 = D00x*gx0;
        gx4 = B00*gx0+D00x*gx1;
        gx2 = ABx*gx0+gx1;
        gx5 = ABx*gx3+gx4;
        double gy1,gy3,gy4,gy2,gy5;
        gy1 = C00y*gy0;
        gy3 = D00y*gy0;
        gy4 = B00*gy0+D00y*gy1;
        gy2 = ABy*gy0+gy1;
        gy5 = ABy*gy3+gy4;
        double gz1,gz3,gz4,gz2,gz5;
        gz1 = C00z*gz0;
        gz3 = D00z*gz0;
        gz4 = B00*gz0+D00z*gz1;
        gz2 = ABz*gz0+gz1;
        gz5 = ABz*gz3+gz4;

        gout0_ix += (ai*gx4)*gy0*gz0;
        gout1_ix += (ai*gx1)*gy3*gz0;
        gout2_ix += (ai*gx1)*gy0*gz3;
        gout0_iy += gx3*(ai*gy1)*gz0;
        gout1_iy += gx0*(ai*gy4)*gz0;
        gout2_iy += gx0*(ai*gy1)*gz3;
        gout0_iz += gx3*gy0*(ai*gz1);
        gout1_iz += gx0*gy3*(ai*gz1);
        gout2_iz += gx0*gy0*(ai*gz4);
        gout0_jx += (aj*gx5)*gy0*gz0;
        gout1_jx += (aj*gx2)*gy3*gz0;
        gout2_jx += (aj*gx2)*gy0*gz3;
        gout0_jy += gx3*(aj*gy2)*gz0;
        gout1_jy += gx0*(aj*gy5)*gz0;
        gout2_jy += gx0*(aj*gy2)*gz3;
        gout0_jz += gx3*gy0*(aj*gz2);
        gout1_jz += gx0*gy3*(aj*gz2);
        gout2_jz += gx0*gy0*(aj*gz5);
      }
    } }

  int *ao_loc = c_bpcache.ao_loc;
  int i0 = ao_loc[ish];
  int j0 = ao_loc[jsh];
  int k0 = ao_loc[ksh];
  int l0 = ao_loc[lsh];
  int n_dm = jk.n_dm;
  int nao = jk.nao;
  double* __restrict__ dm = jk.dm;
  double *vj = jk.vj;
  double *vk = jk.vk;
  double d, d0, d1, d2, d3;
  for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
    if(vj != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(j0+0)];
      d1 = dm[(k0+0)+nao*(l0+0)];
      d2 = dm[(k0+1)+nao*(l0+0)];
      d3 = dm[(k0+2)+nao*(l0+0)];

      d = d0*d1;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d0*d2;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d0*d3;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      atomicAdd(vj+ish*3  , shell_ix);
      atomicAdd(vj+ish*3+1, shell_iy);
      atomicAdd(vj+ish*3+2, shell_iz);
      atomicAdd(vj+jsh*3  , shell_jx);
      atomicAdd(vj+jsh*3+1, shell_jy);
      atomicAdd(vj+jsh*3+2, shell_jz);
    }
    if(vk != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(k0+0)];
      d1 = dm[(i0+0)+nao*(k0+1)];
      d2 = dm[(i0+0)+nao*(k0+2)];
      d3 = dm[(j0+0)+nao*(l0+0)];

      d = d0*d3;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d1*d3;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d2*d3;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d0 = dm[(i0+0)+nao*(l0+0)];
      d1 = dm[(j0+0)+nao*(k0+0)];
      d2 = dm[(j0+0)+nao*(k0+1)];
      d3 = dm[(j0+0)+nao*(k0+2)];

      d = d0*d1;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d0*d2;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d0*d3;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      atomicAdd(vk+ish*3  , shell_ix);
      atomicAdd(vk+ish*3+1, shell_iy);
      atomicAdd(vk+ish*3+2, shell_iz);
      atomicAdd(vk+jsh*3  , shell_jx);
      atomicAdd(vk+jsh*3+1, shell_jy);
      atomicAdd(vk+jsh*3+2, shell_jz);
    }
  }
}

__global__
static void GINTint2e_get_veff_ip1_kernel0011(GINTEnvVars envs,
                                              JKMatrix jk,
                                              BasisProdOffsets offsets)
{
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;
  double norm = envs.fac;
  double omega = envs.omega;
  int *bas_pair2bra = c_bpcache.bas_pair2bra;
  int *bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];
  int nprim_ij = envs.nprim_ij;
  int nprim_kl = envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

  double* __restrict__ a12 = c_bpcache.a12;
  double* __restrict__ e12 = c_bpcache.e12;
  double* __restrict__ x12 = c_bpcache.x12;
  double* __restrict__ y12 = c_bpcache.y12;
  double* __restrict__ z12 = c_bpcache.z12;
  double * __restrict__ i_exponent = c_bpcache.a1;
  double * __restrict__ j_exponent = c_bpcache.a2;

  int ij, kl;
  int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
  int nbas = c_bpcache.nbas;
  double* __restrict__ bas_x = c_bpcache.bas_coords;
  double* __restrict__ bas_y = bas_x + nbas;
  double* __restrict__ bas_z = bas_y + nbas;

  double gout0_ix = 0;
  double gout1_ix = 0;
  double gout2_ix = 0;
  double gout3_ix = 0;
  double gout4_ix = 0;
  double gout5_ix = 0;
  double gout6_ix = 0;
  double gout7_ix = 0;
  double gout8_ix = 0;
  double gout0_iy = 0;
  double gout1_iy = 0;
  double gout2_iy = 0;
  double gout3_iy = 0;
  double gout4_iy = 0;
  double gout5_iy = 0;
  double gout6_iy = 0;
  double gout7_iy = 0;
  double gout8_iy = 0;
  double gout0_iz = 0;
  double gout1_iz = 0;
  double gout2_iz = 0;
  double gout3_iz = 0;
  double gout4_iz = 0;
  double gout5_iz = 0;
  double gout6_iz = 0;
  double gout7_iz = 0;
  double gout8_iz = 0;
  double gout0_jx = 0;
  double gout1_jx = 0;
  double gout2_jx = 0;
  double gout3_jx = 0;
  double gout4_jx = 0;
  double gout5_jx = 0;
  double gout6_jx = 0;
  double gout7_jx = 0;
  double gout8_jx = 0;
  double gout0_jy = 0;
  double gout1_jy = 0;
  double gout2_jy = 0;
  double gout3_jy = 0;
  double gout4_jy = 0;
  double gout5_jy = 0;
  double gout6_jy = 0;
  double gout7_jy = 0;
  double gout8_jy = 0;
  double gout0_jz = 0;
  double gout1_jz = 0;
  double gout2_jz = 0;
  double gout3_jz = 0;
  double gout4_jz = 0;
  double gout5_jz = 0;
  double gout6_jz = 0;
  double gout7_jz = 0;
  double gout8_jz = 0;

  double xi = bas_x[ish];
  double yi = bas_y[ish];
  double zi = bas_z[ish];
  double ABx = xi - bas_x[jsh];
  double ABy = yi - bas_y[jsh];
  double ABz = zi - bas_z[jsh];
  double xk = bas_x[ksh];
  double yk = bas_y[ksh];
  double zk = bas_z[ksh];
  double CDx = xk - bas_x[lsh];
  double CDy = yk - bas_y[lsh];
  double CDz = zk - bas_z[lsh];
  prim_ij0 = prim_ij;
  prim_ij1 = prim_ij + nprim_ij;
  prim_kl0 = prim_kl;
  prim_kl1 = prim_kl + nprim_kl;
  double rw[4];
  int irys;
  for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    double ai = i_exponent[ij] * 2.0;
    double aj = j_exponent[ij] * 2.0;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
      double akl = a12[kl];
      double ekl = e12[kl];
      double xkl = x12[kl];
      double ykl = y12[kl];
      double zkl = z12[kl];
      double xijxkl = xij - xkl;
      double yijykl = yij - ykl;
      double zijzkl = zij - zkl;
      double aijkl = aij + akl;
      double a1 = aij * akl;
      double a0 = a1 / aijkl;
      double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
      a0 *= theta;
      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      double fac = norm * eij * ekl * sqrt(a0 / (a1 * a1 * a1));
      //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

      GINTrys_root<2>(x, rw);
      GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
        double gz0 = rw[irys+2] * fac;
        double root0 = rw[irys];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double B00 = u2 * tmp4;
        double tmp1 = 2 * B00;

        double tmp2 = tmp1 * akl;
        double C00x = xij - xi - tmp2 * xijxkl;
        double C00y = yij - yi - tmp2 * yijykl;
        double C00z = zij - zi - tmp2 * zijzkl;
        double B01 = B00 + tmp4 * aij;
        double tmp3 = tmp1 * aij;
        double D00x = xkl - xk + tmp3 * xijxkl;
        double D00y = ykl - yk + tmp3 * yijykl;
        double D00z = zkl - zk + tmp3 * zijzkl;

        double gx0 = 1;
        double gy0 = 1;
        double gx1,gx3,gx4,gx6,gx7,gx9,gx10,gx2,gx5,gx8,gx11;
        gx1 = C00x*gx0;
        gx3 = D00x*gx0;
        gx4 = B00*gx0+D00x*gx1;
        gx6 = B01*gx0+D00x*gx3;
        gx7 = B01*gx1+B00*gx3+D00x*gx4;
        gx2 = ABx*gx0+gx1;
        gx5 = ABx*gx3+gx4;
        gx8 = ABx*gx6+gx7;
        gx9 = CDx*gx3+gx6;
        gx10 = CDx*gx4+gx7;
        gx11 = CDx*gx5+gx8;
        gx6 = CDx*gx0+gx3;
        gx7 = CDx*gx1+gx4;
        gx8 = CDx*gx2+gx5;
        double gy1,gy3,gy4,gy6,gy7,gy9,gy10,gy2,gy5,gy8,gy11;
        gy1 = C00y*gy0;
        gy3 = D00y*gy0;
        gy4 = B00*gy0+D00y*gy1;
        gy6 = B01*gy0+D00y*gy3;
        gy7 = B01*gy1+B00*gy3+D00y*gy4;
        gy2 = ABy*gy0+gy1;
        gy5 = ABy*gy3+gy4;
        gy8 = ABy*gy6+gy7;
        gy9 = CDy*gy3+gy6;
        gy10 = CDy*gy4+gy7;
        gy11 = CDy*gy5+gy8;
        gy6 = CDy*gy0+gy3;
        gy7 = CDy*gy1+gy4;
        gy8 = CDy*gy2+gy5;
        double gz1,gz3,gz4,gz6,gz7,gz9,gz10,gz2,gz5,gz8,gz11;
        gz1 = C00z*gz0;
        gz3 = D00z*gz0;
        gz4 = B00*gz0+D00z*gz1;
        gz6 = B01*gz0+D00z*gz3;
        gz7 = B01*gz1+B00*gz3+D00z*gz4;
        gz2 = ABz*gz0+gz1;
        gz5 = ABz*gz3+gz4;
        gz8 = ABz*gz6+gz7;
        gz9 = CDz*gz3+gz6;
        gz10 = CDz*gz4+gz7;
        gz11 = CDz*gz5+gz8;
        gz6 = CDz*gz0+gz3;
        gz7 = CDz*gz1+gz4;
        gz8 = CDz*gz2+gz5;

        gout0_ix += (ai*gx10)*gy0*gz0;
        gout1_ix += (ai*gx4)*gy6*gz0;
        gout2_ix += (ai*gx4)*gy0*gz6;
        gout3_ix += (ai*gx7)*gy3*gz0;
        gout4_ix += (ai*gx1)*gy9*gz0;
        gout5_ix += (ai*gx1)*gy3*gz6;
        gout6_ix += (ai*gx7)*gy0*gz3;
        gout7_ix += (ai*gx1)*gy6*gz3;
        gout8_ix += (ai*gx1)*gy0*gz9;
        gout0_iy += gx9*(ai*gy1)*gz0;
        gout1_iy += gx3*(ai*gy7)*gz0;
        gout2_iy += gx3*(ai*gy1)*gz6;
        gout3_iy += gx6*(ai*gy4)*gz0;
        gout4_iy += gx0*(ai*gy10)*gz0;
        gout5_iy += gx0*(ai*gy4)*gz6;
        gout6_iy += gx6*(ai*gy1)*gz3;
        gout7_iy += gx0*(ai*gy7)*gz3;
        gout8_iy += gx0*(ai*gy1)*gz9;
        gout0_iz += gx9*gy0*(ai*gz1);
        gout1_iz += gx3*gy6*(ai*gz1);
        gout2_iz += gx3*gy0*(ai*gz7);
        gout3_iz += gx6*gy3*(ai*gz1);
        gout4_iz += gx0*gy9*(ai*gz1);
        gout5_iz += gx0*gy3*(ai*gz7);
        gout6_iz += gx6*gy0*(ai*gz4);
        gout7_iz += gx0*gy6*(ai*gz4);
        gout8_iz += gx0*gy0*(ai*gz10);
        gout0_jx += (aj*gx11)*gy0*gz0;
        gout1_jx += (aj*gx5)*gy6*gz0;
        gout2_jx += (aj*gx5)*gy0*gz6;
        gout3_jx += (aj*gx8)*gy3*gz0;
        gout4_jx += (aj*gx2)*gy9*gz0;
        gout5_jx += (aj*gx2)*gy3*gz6;
        gout6_jx += (aj*gx8)*gy0*gz3;
        gout7_jx += (aj*gx2)*gy6*gz3;
        gout8_jx += (aj*gx2)*gy0*gz9;
        gout0_jy += gx9*(aj*gy2)*gz0;
        gout1_jy += gx3*(aj*gy8)*gz0;
        gout2_jy += gx3*(aj*gy2)*gz6;
        gout3_jy += gx6*(aj*gy5)*gz0;
        gout4_jy += gx0*(aj*gy11)*gz0;
        gout5_jy += gx0*(aj*gy5)*gz6;
        gout6_jy += gx6*(aj*gy2)*gz3;
        gout7_jy += gx0*(aj*gy8)*gz3;
        gout8_jy += gx0*(aj*gy2)*gz9;
        gout0_jz += gx9*gy0*(aj*gz2);
        gout1_jz += gx3*gy6*(aj*gz2);
        gout2_jz += gx3*gy0*(aj*gz8);
        gout3_jz += gx6*gy3*(aj*gz2);
        gout4_jz += gx0*gy9*(aj*gz2);
        gout5_jz += gx0*gy3*(aj*gz8);
        gout6_jz += gx6*gy0*(aj*gz5);
        gout7_jz += gx0*gy6*(aj*gz5);
        gout8_jz += gx0*gy0*(aj*gz11);
      }
    } }

  int *ao_loc = c_bpcache.ao_loc;
  int i0 = ao_loc[ish];
  int j0 = ao_loc[jsh];
  int k0 = ao_loc[ksh];
  int l0 = ao_loc[lsh];
  int n_dm = jk.n_dm;
  int nao = jk.nao;
  double* __restrict__ dm = jk.dm;
  double *vj = jk.vj;
  double *vk = jk.vk;
  double d, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9;
  for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
    if(vj != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(j0+0)];
      d1 = dm[(k0+0)+nao*(l0+0)];
      d2 = dm[(k0+0)+nao*(l0+1)];
      d3 = dm[(k0+0)+nao*(l0+2)];
      d4 = dm[(k0+1)+nao*(l0+0)];
      d5 = dm[(k0+1)+nao*(l0+1)];
      d6 = dm[(k0+1)+nao*(l0+2)];
      d7 = dm[(k0+2)+nao*(l0+0)];
      d8 = dm[(k0+2)+nao*(l0+1)];
      d9 = dm[(k0+2)+nao*(l0+2)];

      d = d0*d1;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d0*d2;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d0*d3;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d0*d4;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d0*d5;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d0*d6;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      d = d0*d7;
      shell_ix += gout6_ix*d;
      shell_iy += gout6_iy*d;
      shell_iz += gout6_iz*d;
      shell_jx += gout6_jx*d;
      shell_jy += gout6_jy*d;
      shell_jz += gout6_jz*d;

      d = d0*d8;
      shell_ix += gout7_ix*d;
      shell_iy += gout7_iy*d;
      shell_iz += gout7_iz*d;
      shell_jx += gout7_jx*d;
      shell_jy += gout7_jy*d;
      shell_jz += gout7_jz*d;

      d = d0*d9;
      shell_ix += gout8_ix*d;
      shell_iy += gout8_iy*d;
      shell_iz += gout8_iz*d;
      shell_jx += gout8_jx*d;
      shell_jy += gout8_jy*d;
      shell_jz += gout8_jz*d;

      atomicAdd(vj+ish*3  , shell_ix);
      atomicAdd(vj+ish*3+1, shell_iy);
      atomicAdd(vj+ish*3+2, shell_iz);
      atomicAdd(vj+jsh*3  , shell_jx);
      atomicAdd(vj+jsh*3+1, shell_jy);
      atomicAdd(vj+jsh*3+2, shell_jz);
    }
    if(vk != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(k0+0)];
      d1 = dm[(i0+0)+nao*(k0+1)];
      d2 = dm[(i0+0)+nao*(k0+2)];
      d3 = dm[(j0+0)+nao*(l0+0)];
      d4 = dm[(j0+0)+nao*(l0+1)];
      d5 = dm[(j0+0)+nao*(l0+2)];

      d = d0*d3;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d0*d4;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d0*d5;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d1*d3;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d1*d4;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d1*d5;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      d = d2*d3;
      shell_ix += gout6_ix*d;
      shell_iy += gout6_iy*d;
      shell_iz += gout6_iz*d;
      shell_jx += gout6_jx*d;
      shell_jy += gout6_jy*d;
      shell_jz += gout6_jz*d;

      d = d2*d4;
      shell_ix += gout7_ix*d;
      shell_iy += gout7_iy*d;
      shell_iz += gout7_iz*d;
      shell_jx += gout7_jx*d;
      shell_jy += gout7_jy*d;
      shell_jz += gout7_jz*d;

      d = d2*d5;
      shell_ix += gout8_ix*d;
      shell_iy += gout8_iy*d;
      shell_iz += gout8_iz*d;
      shell_jx += gout8_jx*d;
      shell_jy += gout8_jy*d;
      shell_jz += gout8_jz*d;

      d0 = dm[(i0+0)+nao*(l0+0)];
      d1 = dm[(i0+0)+nao*(l0+1)];
      d2 = dm[(i0+0)+nao*(l0+2)];
      d3 = dm[(j0+0)+nao*(k0+0)];
      d4 = dm[(j0+0)+nao*(k0+1)];
      d5 = dm[(j0+0)+nao*(k0+2)];

      d = d0*d3;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d1*d3;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d2*d3;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d0*d4;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d1*d4;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d2*d4;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      d = d0*d5;
      shell_ix += gout6_ix*d;
      shell_iy += gout6_iy*d;
      shell_iz += gout6_iz*d;
      shell_jx += gout6_jx*d;
      shell_jy += gout6_jy*d;
      shell_jz += gout6_jz*d;

      d = d1*d5;
      shell_ix += gout7_ix*d;
      shell_iy += gout7_iy*d;
      shell_iz += gout7_iz*d;
      shell_jx += gout7_jx*d;
      shell_jy += gout7_jy*d;
      shell_jz += gout7_jz*d;

      d = d2*d5;
      shell_ix += gout8_ix*d;
      shell_iy += gout8_iy*d;
      shell_iz += gout8_iz*d;
      shell_jx += gout8_jx*d;
      shell_jy += gout8_jy*d;
      shell_jz += gout8_jz*d;

      atomicAdd(vk+ish*3  , shell_ix);
      atomicAdd(vk+ish*3+1, shell_iy);
      atomicAdd(vk+ish*3+2, shell_iz);
      atomicAdd(vk+jsh*3  , shell_jx);
      atomicAdd(vk+jsh*3+1, shell_jy);
      atomicAdd(vk+jsh*3+2, shell_jz);
    }
  }
}

__global__
static void GINTint2e_get_veff_ip1_kernel0020(GINTEnvVars envs,
                                              JKMatrix jk,
                                              BasisProdOffsets offsets)
{
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;
  double norm = envs.fac;
  double omega = envs.omega;
  int *bas_pair2bra = c_bpcache.bas_pair2bra;
  int *bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];
  int nprim_ij = envs.nprim_ij;
  int nprim_kl = envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

  double* __restrict__ a12 = c_bpcache.a12;
  double* __restrict__ e12 = c_bpcache.e12;
  double* __restrict__ x12 = c_bpcache.x12;
  double* __restrict__ y12 = c_bpcache.y12;
  double* __restrict__ z12 = c_bpcache.z12;
  double * __restrict__ i_exponent = c_bpcache.a1;
  double * __restrict__ j_exponent = c_bpcache.a2;

  int ij, kl;
  int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
  int nbas = c_bpcache.nbas;
  double* __restrict__ bas_x = c_bpcache.bas_coords;
  double* __restrict__ bas_y = bas_x + nbas;
  double* __restrict__ bas_z = bas_y + nbas;

  double gout0_ix = 0;
  double gout1_ix = 0;
  double gout2_ix = 0;
  double gout3_ix = 0;
  double gout4_ix = 0;
  double gout5_ix = 0;
  double gout0_iy = 0;
  double gout1_iy = 0;
  double gout2_iy = 0;
  double gout3_iy = 0;
  double gout4_iy = 0;
  double gout5_iy = 0;
  double gout0_iz = 0;
  double gout1_iz = 0;
  double gout2_iz = 0;
  double gout3_iz = 0;
  double gout4_iz = 0;
  double gout5_iz = 0;
  double gout0_jx = 0;
  double gout1_jx = 0;
  double gout2_jx = 0;
  double gout3_jx = 0;
  double gout4_jx = 0;
  double gout5_jx = 0;
  double gout0_jy = 0;
  double gout1_jy = 0;
  double gout2_jy = 0;
  double gout3_jy = 0;
  double gout4_jy = 0;
  double gout5_jy = 0;
  double gout0_jz = 0;
  double gout1_jz = 0;
  double gout2_jz = 0;
  double gout3_jz = 0;
  double gout4_jz = 0;
  double gout5_jz = 0;

  double xi = bas_x[ish];
  double yi = bas_y[ish];
  double zi = bas_z[ish];
  double ABx = xi - bas_x[jsh];
  double ABy = yi - bas_y[jsh];
  double ABz = zi - bas_z[jsh];
  double xk = bas_x[ksh];
  double yk = bas_y[ksh];
  double zk = bas_z[ksh];

  prim_ij0 = prim_ij;
  prim_ij1 = prim_ij + nprim_ij;
  prim_kl0 = prim_kl;
  prim_kl1 = prim_kl + nprim_kl;
  double rw[4];
  int irys;
  for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    double ai = i_exponent[ij] * 2.0;
    double aj = j_exponent[ij] * 2.0;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
      double akl = a12[kl];
      double ekl = e12[kl];
      double xkl = x12[kl];
      double ykl = y12[kl];
      double zkl = z12[kl];
      double xijxkl = xij - xkl;
      double yijykl = yij - ykl;
      double zijzkl = zij - zkl;
      double aijkl = aij + akl;
      double a1 = aij * akl;
      double a0 = a1 / aijkl;
      double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
      a0 *= theta;
      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      double fac = norm * eij * ekl * sqrt(a0 / (a1 * a1 * a1));
      //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

      GINTrys_root<2>(x, rw);
      GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
        double gz0 = rw[irys+2] * fac;
        double root0 = rw[irys];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double B00 = u2 * tmp4;
        double tmp1 = 2 * B00;

        double tmp2 = tmp1 * akl;
        double C00x = xij - xi - tmp2 * xijxkl;
        double C00y = yij - yi - tmp2 * yijykl;
        double C00z = zij - zi - tmp2 * zijzkl;
        double B01 = B00 + tmp4 * aij;
        double tmp3 = tmp1 * aij;
        double D00x = xkl - xk + tmp3 * xijxkl;
        double D00y = ykl - yk + tmp3 * yijykl;
        double D00z = zkl - zk + tmp3 * zijzkl;

        double gx0 = 1;
        double gy0 = 1;
        double gx1,gx3,gx4,gx6,gx7,gx2,gx5,gx8;
        gx1 = C00x*gx0;
        gx3 = D00x*gx0;
        gx4 = B00*gx0+D00x*gx1;
        gx6 = B01*gx0+D00x*gx3;
        gx7 = B01*gx1+B00*gx3+D00x*gx4;
        gx2 = ABx*gx0+gx1;
        gx5 = ABx*gx3+gx4;
        gx8 = ABx*gx6+gx7;
        double gy1,gy3,gy4,gy6,gy7,gy2,gy5,gy8;
        gy1 = C00y*gy0;
        gy3 = D00y*gy0;
        gy4 = B00*gy0+D00y*gy1;
        gy6 = B01*gy0+D00y*gy3;
        gy7 = B01*gy1+B00*gy3+D00y*gy4;
        gy2 = ABy*gy0+gy1;
        gy5 = ABy*gy3+gy4;
        gy8 = ABy*gy6+gy7;
        double gz1,gz3,gz4,gz6,gz7,gz2,gz5,gz8;
        gz1 = C00z*gz0;
        gz3 = D00z*gz0;
        gz4 = B00*gz0+D00z*gz1;
        gz6 = B01*gz0+D00z*gz3;
        gz7 = B01*gz1+B00*gz3+D00z*gz4;
        gz2 = ABz*gz0+gz1;
        gz5 = ABz*gz3+gz4;
        gz8 = ABz*gz6+gz7;

        gout0_ix += (ai*gx7)*gy0*gz0;
        gout1_ix += (ai*gx4)*gy3*gz0;
        gout2_ix += (ai*gx4)*gy0*gz3;
        gout3_ix += (ai*gx1)*gy6*gz0;
        gout4_ix += (ai*gx1)*gy3*gz3;
        gout5_ix += (ai*gx1)*gy0*gz6;
        gout0_iy += gx6*(ai*gy1)*gz0;
        gout1_iy += gx3*(ai*gy4)*gz0;
        gout2_iy += gx3*(ai*gy1)*gz3;
        gout3_iy += gx0*(ai*gy7)*gz0;
        gout4_iy += gx0*(ai*gy4)*gz3;
        gout5_iy += gx0*(ai*gy1)*gz6;
        gout0_iz += gx6*gy0*(ai*gz1);
        gout1_iz += gx3*gy3*(ai*gz1);
        gout2_iz += gx3*gy0*(ai*gz4);
        gout3_iz += gx0*gy6*(ai*gz1);
        gout4_iz += gx0*gy3*(ai*gz4);
        gout5_iz += gx0*gy0*(ai*gz7);
        gout0_jx += (aj*gx8)*gy0*gz0;
        gout1_jx += (aj*gx5)*gy3*gz0;
        gout2_jx += (aj*gx5)*gy0*gz3;
        gout3_jx += (aj*gx2)*gy6*gz0;
        gout4_jx += (aj*gx2)*gy3*gz3;
        gout5_jx += (aj*gx2)*gy0*gz6;
        gout0_jy += gx6*(aj*gy2)*gz0;
        gout1_jy += gx3*(aj*gy5)*gz0;
        gout2_jy += gx3*(aj*gy2)*gz3;
        gout3_jy += gx0*(aj*gy8)*gz0;
        gout4_jy += gx0*(aj*gy5)*gz3;
        gout5_jy += gx0*(aj*gy2)*gz6;
        gout0_jz += gx6*gy0*(aj*gz2);
        gout1_jz += gx3*gy3*(aj*gz2);
        gout2_jz += gx3*gy0*(aj*gz5);
        gout3_jz += gx0*gy6*(aj*gz2);
        gout4_jz += gx0*gy3*(aj*gz5);
        gout5_jz += gx0*gy0*(aj*gz8);
      }
    } }

  int *ao_loc = c_bpcache.ao_loc;
  int i0 = ao_loc[ish];
  int j0 = ao_loc[jsh];
  int k0 = ao_loc[ksh];
  int l0 = ao_loc[lsh];
  int n_dm = jk.n_dm;
  int nao = jk.nao;
  double* __restrict__ dm = jk.dm;
  double *vj = jk.vj;
  double *vk = jk.vk;
  double d, d0, d1, d2, d3, d4, d5, d6;
  for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
    if(vj != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(j0+0)];
      d1 = dm[(k0+0)+nao*(l0+0)];
      d2 = dm[(k0+1)+nao*(l0+0)];
      d3 = dm[(k0+2)+nao*(l0+0)];
      d4 = dm[(k0+3)+nao*(l0+0)];
      d5 = dm[(k0+4)+nao*(l0+0)];
      d6 = dm[(k0+5)+nao*(l0+0)];

      d = d0*d1;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d0*d2;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d0*d3;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d0*d4;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d0*d5;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d0*d6;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      atomicAdd(vj+ish*3  , shell_ix);
      atomicAdd(vj+ish*3+1, shell_iy);
      atomicAdd(vj+ish*3+2, shell_iz);
      atomicAdd(vj+jsh*3  , shell_jx);
      atomicAdd(vj+jsh*3+1, shell_jy);
      atomicAdd(vj+jsh*3+2, shell_jz);
    }
    if(vk != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(k0+0)];
      d1 = dm[(i0+0)+nao*(k0+1)];
      d2 = dm[(i0+0)+nao*(k0+2)];
      d3 = dm[(i0+0)+nao*(k0+3)];
      d4 = dm[(i0+0)+nao*(k0+4)];
      d5 = dm[(i0+0)+nao*(k0+5)];
      d6 = dm[(j0+0)+nao*(l0+0)];

      d = d0*d6;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d1*d6;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d2*d6;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d3*d6;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d4*d6;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d5*d6;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      d0 = dm[(i0+0)+nao*(l0+0)];
      d1 = dm[(j0+0)+nao*(k0+0)];
      d2 = dm[(j0+0)+nao*(k0+1)];
      d3 = dm[(j0+0)+nao*(k0+2)];
      d4 = dm[(j0+0)+nao*(k0+3)];
      d5 = dm[(j0+0)+nao*(k0+4)];
      d6 = dm[(j0+0)+nao*(k0+5)];

      d = d0*d1;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d0*d2;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d0*d3;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d0*d4;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d0*d5;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d0*d6;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      atomicAdd(vk+ish*3  , shell_ix);
      atomicAdd(vk+ish*3+1, shell_iy);
      atomicAdd(vk+ish*3+2, shell_iz);
      atomicAdd(vk+jsh*3  , shell_jx);
      atomicAdd(vk+jsh*3+1, shell_jy);
      atomicAdd(vk+jsh*3+2, shell_jz);
    }
  }
}

__global__
static void GINTint2e_get_veff_ip1_kernel1000(GINTEnvVars envs,
                                              JKMatrix jk,
                                              BasisProdOffsets offsets)
{
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;
  double norm = envs.fac;
  double omega = envs.omega;
  int *bas_pair2bra = c_bpcache.bas_pair2bra;
  int *bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];
  int nprim_ij = envs.nprim_ij;
  int nprim_kl = envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

  double* __restrict__ a12 = c_bpcache.a12;
  double* __restrict__ e12 = c_bpcache.e12;
  double* __restrict__ x12 = c_bpcache.x12;
  double* __restrict__ y12 = c_bpcache.y12;
  double* __restrict__ z12 = c_bpcache.z12;
  double * __restrict__ i_exponent = c_bpcache.a1;
  double * __restrict__ j_exponent = c_bpcache.a2;

  int ij, kl;
  int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
  int nbas = c_bpcache.nbas;
  double* __restrict__ bas_x = c_bpcache.bas_coords;
  double* __restrict__ bas_y = bas_x + nbas;
  double* __restrict__ bas_z = bas_y + nbas;

  double gout0_ix = 0;
  double gout1_ix = 0;
  double gout2_ix = 0;
  double gout0_iy = 0;
  double gout1_iy = 0;
  double gout2_iy = 0;
  double gout0_iz = 0;
  double gout1_iz = 0;
  double gout2_iz = 0;
  double gout0_jx = 0;
  double gout1_jx = 0;
  double gout2_jx = 0;
  double gout0_jy = 0;
  double gout1_jy = 0;
  double gout2_jy = 0;
  double gout0_jz = 0;
  double gout1_jz = 0;
  double gout2_jz = 0;

  double xi = bas_x[ish];
  double yi = bas_y[ish];
  double zi = bas_z[ish];
  double ABx = xi - bas_x[jsh];
  double ABy = yi - bas_y[jsh];
  double ABz = zi - bas_z[jsh];


  prim_ij0 = prim_ij;
  prim_ij1 = prim_ij + nprim_ij;
  prim_kl0 = prim_kl;
  prim_kl1 = prim_kl + nprim_kl;
  double rw[4];
  int irys;
  for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    double ai = i_exponent[ij] * 2.0;
    double aj = j_exponent[ij] * 2.0;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
      double akl = a12[kl];
      double ekl = e12[kl];
      double xkl = x12[kl];
      double ykl = y12[kl];
      double zkl = z12[kl];
      double xijxkl = xij - xkl;
      double yijykl = yij - ykl;
      double zijzkl = zij - zkl;
      double aijkl = aij + akl;
      double a1 = aij * akl;
      double a0 = a1 / aijkl;
      double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
      a0 *= theta;
      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      double fac = norm * eij * ekl * sqrt(a0 / (a1 * a1 * a1));
      //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

      GINTrys_root<2>(x, rw);
      GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
        double gz0 = rw[irys+2] * fac;
        double root0 = rw[irys];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double B00 = u2 * tmp4;
        double tmp1 = 2 * B00;
        double B10 = B00 + tmp4 * akl;
        double tmp2 = tmp1 * akl;
        double C00x = xij - xi - tmp2 * xijxkl;
        double C00y = yij - yi - tmp2 * yijykl;
        double C00z = zij - zi - tmp2 * zijzkl;



        double gx0 = 1;
        double gy0 = 1;
        double gx1,gx2,gx3,gx4;
        gx1 = C00x*gx0;
        gx2 = B10*gx0+C00x*gx1;
        gx4 = ABx*gx1+gx2;
        gx3 = ABx*gx0+gx1;
        double gy1,gy2,gy3,gy4;
        gy1 = C00y*gy0;
        gy2 = B10*gy0+C00y*gy1;
        gy4 = ABy*gy1+gy2;
        gy3 = ABy*gy0+gy1;
        double gz1,gz2,gz3,gz4;
        gz1 = C00z*gz0;
        gz2 = B10*gz0+C00z*gz1;
        gz4 = ABz*gz1+gz2;
        gz3 = ABz*gz0+gz1;

        gout0_ix += (-gx0+ai*gx2)*gy0*gz0;
        gout1_ix += (ai*gx1)*gy1*gz0;
        gout2_ix += (ai*gx1)*gy0*gz1;
        gout0_iy += gx1*(ai*gy1)*gz0;
        gout1_iy += gx0*(-gy0+ai*gy2)*gz0;
        gout2_iy += gx0*(ai*gy1)*gz1;
        gout0_iz += gx1*gy0*(ai*gz1);
        gout1_iz += gx0*gy1*(ai*gz1);
        gout2_iz += gx0*gy0*(-gz0+ai*gz2);
        gout0_jx += (aj*gx4)*gy0*gz0;
        gout1_jx += (aj*gx3)*gy1*gz0;
        gout2_jx += (aj*gx3)*gy0*gz1;
        gout0_jy += gx1*(aj*gy3)*gz0;
        gout1_jy += gx0*(aj*gy4)*gz0;
        gout2_jy += gx0*(aj*gy3)*gz1;
        gout0_jz += gx1*gy0*(aj*gz3);
        gout1_jz += gx0*gy1*(aj*gz3);
        gout2_jz += gx0*gy0*(aj*gz4);
      }
    } }

  int *ao_loc = c_bpcache.ao_loc;
  int i0 = ao_loc[ish];
  int j0 = ao_loc[jsh];
  int k0 = ao_loc[ksh];
  int l0 = ao_loc[lsh];
  int n_dm = jk.n_dm;
  int nao = jk.nao;
  double* __restrict__ dm = jk.dm;
  double *vj = jk.vj;
  double *vk = jk.vk;
  double d, d0, d1, d2, d3;
  for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
    if(vj != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(j0+0)];
      d1 = dm[(i0+1)+nao*(j0+0)];
      d2 = dm[(i0+2)+nao*(j0+0)];
      d3 = dm[(k0+0)+nao*(l0+0)];

      d = d0*d3;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d1*d3;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d2*d3;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      atomicAdd(vj+ish*3  , shell_ix);
      atomicAdd(vj+ish*3+1, shell_iy);
      atomicAdd(vj+ish*3+2, shell_iz);
      atomicAdd(vj+jsh*3  , shell_jx);
      atomicAdd(vj+jsh*3+1, shell_jy);
      atomicAdd(vj+jsh*3+2, shell_jz);
    }
    if(vk != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(k0+0)];
      d1 = dm[(i0+1)+nao*(k0+0)];
      d2 = dm[(i0+2)+nao*(k0+0)];
      d3 = dm[(j0+0)+nao*(l0+0)];

      d = d0*d3;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d1*d3;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d2*d3;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d0 = dm[(i0+0)+nao*(l0+0)];
      d1 = dm[(i0+1)+nao*(l0+0)];
      d2 = dm[(i0+2)+nao*(l0+0)];
      d3 = dm[(j0+0)+nao*(k0+0)];

      d = d0*d3;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d1*d3;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d2*d3;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      atomicAdd(vk+ish*3  , shell_ix);
      atomicAdd(vk+ish*3+1, shell_iy);
      atomicAdd(vk+ish*3+2, shell_iz);
      atomicAdd(vk+jsh*3  , shell_jx);
      atomicAdd(vk+jsh*3+1, shell_jy);
      atomicAdd(vk+jsh*3+2, shell_jz);
    }
  }
}

__global__
static void GINTint2e_get_veff_ip1_kernel1010(GINTEnvVars envs,
                                              JKMatrix jk,
                                              BasisProdOffsets offsets)
{
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;
  double norm = envs.fac;
  double omega = envs.omega;
  int *bas_pair2bra = c_bpcache.bas_pair2bra;
  int *bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];
  int nprim_ij = envs.nprim_ij;
  int nprim_kl = envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

  double* __restrict__ a12 = c_bpcache.a12;
  double* __restrict__ e12 = c_bpcache.e12;
  double* __restrict__ x12 = c_bpcache.x12;
  double* __restrict__ y12 = c_bpcache.y12;
  double* __restrict__ z12 = c_bpcache.z12;
  double * __restrict__ i_exponent = c_bpcache.a1;
  double * __restrict__ j_exponent = c_bpcache.a2;

  int ij, kl;
  int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
  int nbas = c_bpcache.nbas;
  double* __restrict__ bas_x = c_bpcache.bas_coords;
  double* __restrict__ bas_y = bas_x + nbas;
  double* __restrict__ bas_z = bas_y + nbas;

  double gout0_ix = 0;
  double gout1_ix = 0;
  double gout2_ix = 0;
  double gout3_ix = 0;
  double gout4_ix = 0;
  double gout5_ix = 0;
  double gout6_ix = 0;
  double gout7_ix = 0;
  double gout8_ix = 0;
  double gout0_iy = 0;
  double gout1_iy = 0;
  double gout2_iy = 0;
  double gout3_iy = 0;
  double gout4_iy = 0;
  double gout5_iy = 0;
  double gout6_iy = 0;
  double gout7_iy = 0;
  double gout8_iy = 0;
  double gout0_iz = 0;
  double gout1_iz = 0;
  double gout2_iz = 0;
  double gout3_iz = 0;
  double gout4_iz = 0;
  double gout5_iz = 0;
  double gout6_iz = 0;
  double gout7_iz = 0;
  double gout8_iz = 0;
  double gout0_jx = 0;
  double gout1_jx = 0;
  double gout2_jx = 0;
  double gout3_jx = 0;
  double gout4_jx = 0;
  double gout5_jx = 0;
  double gout6_jx = 0;
  double gout7_jx = 0;
  double gout8_jx = 0;
  double gout0_jy = 0;
  double gout1_jy = 0;
  double gout2_jy = 0;
  double gout3_jy = 0;
  double gout4_jy = 0;
  double gout5_jy = 0;
  double gout6_jy = 0;
  double gout7_jy = 0;
  double gout8_jy = 0;
  double gout0_jz = 0;
  double gout1_jz = 0;
  double gout2_jz = 0;
  double gout3_jz = 0;
  double gout4_jz = 0;
  double gout5_jz = 0;
  double gout6_jz = 0;
  double gout7_jz = 0;
  double gout8_jz = 0;

  double xi = bas_x[ish];
  double yi = bas_y[ish];
  double zi = bas_z[ish];
  double ABx = xi - bas_x[jsh];
  double ABy = yi - bas_y[jsh];
  double ABz = zi - bas_z[jsh];
  double xk = bas_x[ksh];
  double yk = bas_y[ksh];
  double zk = bas_z[ksh];

  prim_ij0 = prim_ij;
  prim_ij1 = prim_ij + nprim_ij;
  prim_kl0 = prim_kl;
  prim_kl1 = prim_kl + nprim_kl;
  double rw[4];
  int irys;
  for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    double ai = i_exponent[ij] * 2.0;
    double aj = j_exponent[ij] * 2.0;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
      double akl = a12[kl];
      double ekl = e12[kl];
      double xkl = x12[kl];
      double ykl = y12[kl];
      double zkl = z12[kl];
      double xijxkl = xij - xkl;
      double yijykl = yij - ykl;
      double zijzkl = zij - zkl;
      double aijkl = aij + akl;
      double a1 = aij * akl;
      double a0 = a1 / aijkl;
      double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
      a0 *= theta;
      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      double fac = norm * eij * ekl * sqrt(a0 / (a1 * a1 * a1));
      //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

      GINTrys_root<2>(x, rw);
      GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
        double gz0 = rw[irys+2] * fac;
        double root0 = rw[irys];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double B00 = u2 * tmp4;
        double tmp1 = 2 * B00;
        double B10 = B00 + tmp4 * akl;
        double tmp2 = tmp1 * akl;
        double C00x = xij - xi - tmp2 * xijxkl;
        double C00y = yij - yi - tmp2 * yijykl;
        double C00z = zij - zi - tmp2 * zijzkl;

        double tmp3 = tmp1 * aij;
        double D00x = xkl - xk + tmp3 * xijxkl;
        double D00y = ykl - yk + tmp3 * yijykl;
        double D00z = zkl - zk + tmp3 * zijzkl;

        double gx0 = 1;
        double gy0 = 1;
        double gx1,gx2,gx5,gx6,gx7,gx3,gx4,gx8,gx9;
        gx1 = C00x*gx0;
        gx2 = B10*gx0+C00x*gx1;
        gx5 = D00x*gx0;
        gx6 = B00*gx0+D00x*gx1;
        gx7 = 2*B00*gx1+D00x*gx2;
        gx4 = ABx*gx1+gx2;
        gx3 = ABx*gx0+gx1;
        gx9 = ABx*gx6+gx7;
        gx8 = ABx*gx5+gx6;
        double gy1,gy2,gy5,gy6,gy7,gy3,gy4,gy8,gy9;
        gy1 = C00y*gy0;
        gy2 = B10*gy0+C00y*gy1;
        gy5 = D00y*gy0;
        gy6 = B00*gy0+D00y*gy1;
        gy7 = 2*B00*gy1+D00y*gy2;
        gy4 = ABy*gy1+gy2;
        gy3 = ABy*gy0+gy1;
        gy9 = ABy*gy6+gy7;
        gy8 = ABy*gy5+gy6;
        double gz1,gz2,gz5,gz6,gz7,gz3,gz4,gz8,gz9;
        gz1 = C00z*gz0;
        gz2 = B10*gz0+C00z*gz1;
        gz5 = D00z*gz0;
        gz6 = B00*gz0+D00z*gz1;
        gz7 = 2*B00*gz1+D00z*gz2;
        gz4 = ABz*gz1+gz2;
        gz3 = ABz*gz0+gz1;
        gz9 = ABz*gz6+gz7;
        gz8 = ABz*gz5+gz6;

        gout0_ix += (-gx5+ai*gx7)*gy0*gz0;
        gout1_ix += (-gx0+ai*gx2)*gy5*gz0;
        gout2_ix += (-gx0+ai*gx2)*gy0*gz5;
        gout3_ix += (ai*gx6)*gy1*gz0;
        gout4_ix += (ai*gx1)*gy6*gz0;
        gout5_ix += (ai*gx1)*gy1*gz5;
        gout6_ix += (ai*gx6)*gy0*gz1;
        gout7_ix += (ai*gx1)*gy5*gz1;
        gout8_ix += (ai*gx1)*gy0*gz6;
        gout0_iy += gx6*(ai*gy1)*gz0;
        gout1_iy += gx1*(ai*gy6)*gz0;
        gout2_iy += gx1*(ai*gy1)*gz5;
        gout3_iy += gx5*(-gy0+ai*gy2)*gz0;
        gout4_iy += gx0*(-gy5+ai*gy7)*gz0;
        gout5_iy += gx0*(-gy0+ai*gy2)*gz5;
        gout6_iy += gx5*(ai*gy1)*gz1;
        gout7_iy += gx0*(ai*gy6)*gz1;
        gout8_iy += gx0*(ai*gy1)*gz6;
        gout0_iz += gx6*gy0*(ai*gz1);
        gout1_iz += gx1*gy5*(ai*gz1);
        gout2_iz += gx1*gy0*(ai*gz6);
        gout3_iz += gx5*gy1*(ai*gz1);
        gout4_iz += gx0*gy6*(ai*gz1);
        gout5_iz += gx0*gy1*(ai*gz6);
        gout6_iz += gx5*gy0*(-gz0+ai*gz2);
        gout7_iz += gx0*gy5*(-gz0+ai*gz2);
        gout8_iz += gx0*gy0*(-gz5+ai*gz7);
        gout0_jx += (aj*gx9)*gy0*gz0;
        gout1_jx += (aj*gx4)*gy5*gz0;
        gout2_jx += (aj*gx4)*gy0*gz5;
        gout3_jx += (aj*gx8)*gy1*gz0;
        gout4_jx += (aj*gx3)*gy6*gz0;
        gout5_jx += (aj*gx3)*gy1*gz5;
        gout6_jx += (aj*gx8)*gy0*gz1;
        gout7_jx += (aj*gx3)*gy5*gz1;
        gout8_jx += (aj*gx3)*gy0*gz6;
        gout0_jy += gx6*(aj*gy3)*gz0;
        gout1_jy += gx1*(aj*gy8)*gz0;
        gout2_jy += gx1*(aj*gy3)*gz5;
        gout3_jy += gx5*(aj*gy4)*gz0;
        gout4_jy += gx0*(aj*gy9)*gz0;
        gout5_jy += gx0*(aj*gy4)*gz5;
        gout6_jy += gx5*(aj*gy3)*gz1;
        gout7_jy += gx0*(aj*gy8)*gz1;
        gout8_jy += gx0*(aj*gy3)*gz6;
        gout0_jz += gx6*gy0*(aj*gz3);
        gout1_jz += gx1*gy5*(aj*gz3);
        gout2_jz += gx1*gy0*(aj*gz8);
        gout3_jz += gx5*gy1*(aj*gz3);
        gout4_jz += gx0*gy6*(aj*gz3);
        gout5_jz += gx0*gy1*(aj*gz8);
        gout6_jz += gx5*gy0*(aj*gz4);
        gout7_jz += gx0*gy5*(aj*gz4);
        gout8_jz += gx0*gy0*(aj*gz9);
      }
    } }

  int *ao_loc = c_bpcache.ao_loc;
  int i0 = ao_loc[ish];
  int j0 = ao_loc[jsh];
  int k0 = ao_loc[ksh];
  int l0 = ao_loc[lsh];
  int n_dm = jk.n_dm;
  int nao = jk.nao;
  double* __restrict__ dm = jk.dm;
  double *vj = jk.vj;
  double *vk = jk.vk;
  double d, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9;
  for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
    if(vj != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(j0+0)];
      d1 = dm[(i0+1)+nao*(j0+0)];
      d2 = dm[(i0+2)+nao*(j0+0)];
      d3 = dm[(k0+0)+nao*(l0+0)];
      d4 = dm[(k0+1)+nao*(l0+0)];
      d5 = dm[(k0+2)+nao*(l0+0)];

      d = d0*d3;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d0*d4;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d0*d5;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d1*d3;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d1*d4;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d1*d5;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      d = d2*d3;
      shell_ix += gout6_ix*d;
      shell_iy += gout6_iy*d;
      shell_iz += gout6_iz*d;
      shell_jx += gout6_jx*d;
      shell_jy += gout6_jy*d;
      shell_jz += gout6_jz*d;

      d = d2*d4;
      shell_ix += gout7_ix*d;
      shell_iy += gout7_iy*d;
      shell_iz += gout7_iz*d;
      shell_jx += gout7_jx*d;
      shell_jy += gout7_jy*d;
      shell_jz += gout7_jz*d;

      d = d2*d5;
      shell_ix += gout8_ix*d;
      shell_iy += gout8_iy*d;
      shell_iz += gout8_iz*d;
      shell_jx += gout8_jx*d;
      shell_jy += gout8_jy*d;
      shell_jz += gout8_jz*d;

      atomicAdd(vj+ish*3  , shell_ix);
      atomicAdd(vj+ish*3+1, shell_iy);
      atomicAdd(vj+ish*3+2, shell_iz);
      atomicAdd(vj+jsh*3  , shell_jx);
      atomicAdd(vj+jsh*3+1, shell_jy);
      atomicAdd(vj+jsh*3+2, shell_jz);
    }
    if(vk != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(k0+0)];
      d1 = dm[(i0+0)+nao*(k0+1)];
      d2 = dm[(i0+0)+nao*(k0+2)];
      d3 = dm[(i0+1)+nao*(k0+0)];
      d4 = dm[(i0+1)+nao*(k0+1)];
      d5 = dm[(i0+1)+nao*(k0+2)];
      d6 = dm[(i0+2)+nao*(k0+0)];
      d7 = dm[(i0+2)+nao*(k0+1)];
      d8 = dm[(i0+2)+nao*(k0+2)];
      d9 = dm[(j0+0)+nao*(l0+0)];

      d = d0*d9;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d1*d9;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d2*d9;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d3*d9;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d4*d9;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d5*d9;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      d = d6*d9;
      shell_ix += gout6_ix*d;
      shell_iy += gout6_iy*d;
      shell_iz += gout6_iz*d;
      shell_jx += gout6_jx*d;
      shell_jy += gout6_jy*d;
      shell_jz += gout6_jz*d;

      d = d7*d9;
      shell_ix += gout7_ix*d;
      shell_iy += gout7_iy*d;
      shell_iz += gout7_iz*d;
      shell_jx += gout7_jx*d;
      shell_jy += gout7_jy*d;
      shell_jz += gout7_jz*d;

      d = d8*d9;
      shell_ix += gout8_ix*d;
      shell_iy += gout8_iy*d;
      shell_iz += gout8_iz*d;
      shell_jx += gout8_jx*d;
      shell_jy += gout8_jy*d;
      shell_jz += gout8_jz*d;

      d0 = dm[(i0+0)+nao*(l0+0)];
      d1 = dm[(i0+1)+nao*(l0+0)];
      d2 = dm[(i0+2)+nao*(l0+0)];
      d3 = dm[(j0+0)+nao*(k0+0)];
      d4 = dm[(j0+0)+nao*(k0+1)];
      d5 = dm[(j0+0)+nao*(k0+2)];

      d = d0*d3;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d0*d4;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d0*d5;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d1*d3;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d1*d4;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d1*d5;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      d = d2*d3;
      shell_ix += gout6_ix*d;
      shell_iy += gout6_iy*d;
      shell_iz += gout6_iz*d;
      shell_jx += gout6_jx*d;
      shell_jy += gout6_jy*d;
      shell_jz += gout6_jz*d;

      d = d2*d4;
      shell_ix += gout7_ix*d;
      shell_iy += gout7_iy*d;
      shell_iz += gout7_iz*d;
      shell_jx += gout7_jx*d;
      shell_jy += gout7_jy*d;
      shell_jz += gout7_jz*d;

      d = d2*d5;
      shell_ix += gout8_ix*d;
      shell_iy += gout8_iy*d;
      shell_iz += gout8_iz*d;
      shell_jx += gout8_jx*d;
      shell_jy += gout8_jy*d;
      shell_jz += gout8_jz*d;

      atomicAdd(vk+ish*3  , shell_ix);
      atomicAdd(vk+ish*3+1, shell_iy);
      atomicAdd(vk+ish*3+2, shell_iz);
      atomicAdd(vk+jsh*3  , shell_jx);
      atomicAdd(vk+jsh*3+1, shell_jy);
      atomicAdd(vk+jsh*3+2, shell_jz);
    }
  }
}

__global__
static void GINTint2e_get_veff_ip1_kernel1100(GINTEnvVars envs,
                                              JKMatrix jk,
                                              BasisProdOffsets offsets)
{
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;
  double norm = envs.fac;
  double omega = envs.omega;
  int *bas_pair2bra = c_bpcache.bas_pair2bra;
  int *bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];
  int nprim_ij = envs.nprim_ij;
  int nprim_kl = envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

  double* __restrict__ a12 = c_bpcache.a12;
  double* __restrict__ e12 = c_bpcache.e12;
  double* __restrict__ x12 = c_bpcache.x12;
  double* __restrict__ y12 = c_bpcache.y12;
  double* __restrict__ z12 = c_bpcache.z12;
  double * __restrict__ i_exponent = c_bpcache.a1;
  double * __restrict__ j_exponent = c_bpcache.a2;

  int ij, kl;
  int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
  int nbas = c_bpcache.nbas;
  double* __restrict__ bas_x = c_bpcache.bas_coords;
  double* __restrict__ bas_y = bas_x + nbas;
  double* __restrict__ bas_z = bas_y + nbas;

  double gout0_ix = 0;
  double gout1_ix = 0;
  double gout2_ix = 0;
  double gout3_ix = 0;
  double gout4_ix = 0;
  double gout5_ix = 0;
  double gout6_ix = 0;
  double gout7_ix = 0;
  double gout8_ix = 0;
  double gout0_iy = 0;
  double gout1_iy = 0;
  double gout2_iy = 0;
  double gout3_iy = 0;
  double gout4_iy = 0;
  double gout5_iy = 0;
  double gout6_iy = 0;
  double gout7_iy = 0;
  double gout8_iy = 0;
  double gout0_iz = 0;
  double gout1_iz = 0;
  double gout2_iz = 0;
  double gout3_iz = 0;
  double gout4_iz = 0;
  double gout5_iz = 0;
  double gout6_iz = 0;
  double gout7_iz = 0;
  double gout8_iz = 0;
  double gout0_jx = 0;
  double gout1_jx = 0;
  double gout2_jx = 0;
  double gout3_jx = 0;
  double gout4_jx = 0;
  double gout5_jx = 0;
  double gout6_jx = 0;
  double gout7_jx = 0;
  double gout8_jx = 0;
  double gout0_jy = 0;
  double gout1_jy = 0;
  double gout2_jy = 0;
  double gout3_jy = 0;
  double gout4_jy = 0;
  double gout5_jy = 0;
  double gout6_jy = 0;
  double gout7_jy = 0;
  double gout8_jy = 0;
  double gout0_jz = 0;
  double gout1_jz = 0;
  double gout2_jz = 0;
  double gout3_jz = 0;
  double gout4_jz = 0;
  double gout5_jz = 0;
  double gout6_jz = 0;
  double gout7_jz = 0;
  double gout8_jz = 0;

  double xi = bas_x[ish];
  double yi = bas_y[ish];
  double zi = bas_z[ish];
  double ABx = xi - bas_x[jsh];
  double ABy = yi - bas_y[jsh];
  double ABz = zi - bas_z[jsh];


  prim_ij0 = prim_ij;
  prim_ij1 = prim_ij + nprim_ij;
  prim_kl0 = prim_kl;
  prim_kl1 = prim_kl + nprim_kl;
  double rw[4];
  int irys;
  for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    double ai = i_exponent[ij] * 2.0;
    double aj = j_exponent[ij] * 2.0;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
      double akl = a12[kl];
      double ekl = e12[kl];
      double xkl = x12[kl];
      double ykl = y12[kl];
      double zkl = z12[kl];
      double xijxkl = xij - xkl;
      double yijykl = yij - ykl;
      double zijzkl = zij - zkl;
      double aijkl = aij + akl;
      double a1 = aij * akl;
      double a0 = a1 / aijkl;
      double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
      a0 *= theta;
      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      double fac = norm * eij * ekl * sqrt(a0 / (a1 * a1 * a1));
      //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

      GINTrys_root<2>(x, rw);
      GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
        double gz0 = rw[irys+2] * fac;
        double root0 = rw[irys];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double B00 = u2 * tmp4;
        double tmp1 = 2 * B00;
        double B10 = B00 + tmp4 * akl;
        double tmp2 = tmp1 * akl;
        double C00x = xij - xi - tmp2 * xijxkl;
        double C00y = yij - yi - tmp2 * yijykl;
        double C00z = zij - zi - tmp2 * zijzkl;



        double gx0 = 1;
        double gy0 = 1;
        double gx1,gx2,gx3,gx4,gx5,gx6,gx7;
        gx1 = C00x*gx0;
        gx2 = B10*gx0+C00x*gx1;
        gx3 = 2*B10*gx1+C00x*gx2;
        gx5 = ABx*gx2+gx3;
        gx4 = ABx*gx1+gx2;
        gx3 = ABx*gx0+gx1;
        gx7 = ABx*gx4+gx5;
        gx6 = ABx*gx3+gx4;
        double gy1,gy2,gy3,gy4,gy5,gy6,gy7;
        gy1 = C00y*gy0;
        gy2 = B10*gy0+C00y*gy1;
        gy3 = 2*B10*gy1+C00y*gy2;
        gy5 = ABy*gy2+gy3;
        gy4 = ABy*gy1+gy2;
        gy3 = ABy*gy0+gy1;
        gy7 = ABy*gy4+gy5;
        gy6 = ABy*gy3+gy4;
        double gz1,gz2,gz3,gz4,gz5,gz6,gz7;
        gz1 = C00z*gz0;
        gz2 = B10*gz0+C00z*gz1;
        gz3 = 2*B10*gz1+C00z*gz2;
        gz5 = ABz*gz2+gz3;
        gz4 = ABz*gz1+gz2;
        gz3 = ABz*gz0+gz1;
        gz7 = ABz*gz4+gz5;
        gz6 = ABz*gz3+gz4;

        gout0_ix += (-gx3+ai*gx5)*gy0*gz0;
        gout1_ix += (-gx0+ai*gx2)*gy3*gz0;
        gout2_ix += (-gx0+ai*gx2)*gy0*gz3;
        gout3_ix += (ai*gx4)*gy1*gz0;
        gout4_ix += (ai*gx1)*gy4*gz0;
        gout5_ix += (ai*gx1)*gy1*gz3;
        gout6_ix += (ai*gx4)*gy0*gz1;
        gout7_ix += (ai*gx1)*gy3*gz1;
        gout8_ix += (ai*gx1)*gy0*gz4;
        gout0_iy += gx4*(ai*gy1)*gz0;
        gout1_iy += gx1*(ai*gy4)*gz0;
        gout2_iy += gx1*(ai*gy1)*gz3;
        gout3_iy += gx3*(-gy0+ai*gy2)*gz0;
        gout4_iy += gx0*(-gy3+ai*gy5)*gz0;
        gout5_iy += gx0*(-gy0+ai*gy2)*gz3;
        gout6_iy += gx3*(ai*gy1)*gz1;
        gout7_iy += gx0*(ai*gy4)*gz1;
        gout8_iy += gx0*(ai*gy1)*gz4;
        gout0_iz += gx4*gy0*(ai*gz1);
        gout1_iz += gx1*gy3*(ai*gz1);
        gout2_iz += gx1*gy0*(ai*gz4);
        gout3_iz += gx3*gy1*(ai*gz1);
        gout4_iz += gx0*gy4*(ai*gz1);
        gout5_iz += gx0*gy1*(ai*gz4);
        gout6_iz += gx3*gy0*(-gz0+ai*gz2);
        gout7_iz += gx0*gy3*(-gz0+ai*gz2);
        gout8_iz += gx0*gy0*(-gz3+ai*gz5);
        gout0_jx += (-gx1+aj*gx7)*gy0*gz0;
        gout1_jx += (aj*gx4)*gy3*gz0;
        gout2_jx += (aj*gx4)*gy0*gz3;
        gout3_jx += (-gx0+aj*gx6)*gy1*gz0;
        gout4_jx += (aj*gx3)*gy4*gz0;
        gout5_jx += (aj*gx3)*gy1*gz3;
        gout6_jx += (-gx0+aj*gx6)*gy0*gz1;
        gout7_jx += (aj*gx3)*gy3*gz1;
        gout8_jx += (aj*gx3)*gy0*gz4;
        gout0_jy += gx4*(aj*gy3)*gz0;
        gout1_jy += gx1*(-gy0+aj*gy6)*gz0;
        gout2_jy += gx1*(aj*gy3)*gz3;
        gout3_jy += gx3*(aj*gy4)*gz0;
        gout4_jy += gx0*(-gy1+aj*gy7)*gz0;
        gout5_jy += gx0*(aj*gy4)*gz3;
        gout6_jy += gx3*(aj*gy3)*gz1;
        gout7_jy += gx0*(-gy0+aj*gy6)*gz1;
        gout8_jy += gx0*(aj*gy3)*gz4;
        gout0_jz += gx4*gy0*(aj*gz3);
        gout1_jz += gx1*gy3*(aj*gz3);
        gout2_jz += gx1*gy0*(-gz0+aj*gz6);
        gout3_jz += gx3*gy1*(aj*gz3);
        gout4_jz += gx0*gy4*(aj*gz3);
        gout5_jz += gx0*gy1*(-gz0+aj*gz6);
        gout6_jz += gx3*gy0*(aj*gz4);
        gout7_jz += gx0*gy3*(aj*gz4);
        gout8_jz += gx0*gy0*(-gz1+aj*gz7);
      }
    } }

  int *ao_loc = c_bpcache.ao_loc;
  int i0 = ao_loc[ish];
  int j0 = ao_loc[jsh];
  int k0 = ao_loc[ksh];
  int l0 = ao_loc[lsh];
  int n_dm = jk.n_dm;
  int nao = jk.nao;
  double* __restrict__ dm = jk.dm;
  double *vj = jk.vj;
  double *vk = jk.vk;
  double d, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9;
  for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
    if(vj != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(j0+0)];
      d1 = dm[(i0+0)+nao*(j0+1)];
      d2 = dm[(i0+0)+nao*(j0+2)];
      d3 = dm[(i0+1)+nao*(j0+0)];
      d4 = dm[(i0+1)+nao*(j0+1)];
      d5 = dm[(i0+1)+nao*(j0+2)];
      d6 = dm[(i0+2)+nao*(j0+0)];
      d7 = dm[(i0+2)+nao*(j0+1)];
      d8 = dm[(i0+2)+nao*(j0+2)];
      d9 = dm[(k0+0)+nao*(l0+0)];

      d = d0*d9;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d1*d9;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d2*d9;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d3*d9;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d4*d9;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d5*d9;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      d = d6*d9;
      shell_ix += gout6_ix*d;
      shell_iy += gout6_iy*d;
      shell_iz += gout6_iz*d;
      shell_jx += gout6_jx*d;
      shell_jy += gout6_jy*d;
      shell_jz += gout6_jz*d;

      d = d7*d9;
      shell_ix += gout7_ix*d;
      shell_iy += gout7_iy*d;
      shell_iz += gout7_iz*d;
      shell_jx += gout7_jx*d;
      shell_jy += gout7_jy*d;
      shell_jz += gout7_jz*d;

      d = d8*d9;
      shell_ix += gout8_ix*d;
      shell_iy += gout8_iy*d;
      shell_iz += gout8_iz*d;
      shell_jx += gout8_jx*d;
      shell_jy += gout8_jy*d;
      shell_jz += gout8_jz*d;

      atomicAdd(vj+ish*3  , shell_ix);
      atomicAdd(vj+ish*3+1, shell_iy);
      atomicAdd(vj+ish*3+2, shell_iz);
      atomicAdd(vj+jsh*3  , shell_jx);
      atomicAdd(vj+jsh*3+1, shell_jy);
      atomicAdd(vj+jsh*3+2, shell_jz);
    }
    if(vk != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(k0+0)];
      d1 = dm[(i0+1)+nao*(k0+0)];
      d2 = dm[(i0+2)+nao*(k0+0)];
      d3 = dm[(j0+0)+nao*(l0+0)];
      d4 = dm[(j0+1)+nao*(l0+0)];
      d5 = dm[(j0+2)+nao*(l0+0)];

      d = d0*d3;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d0*d4;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d0*d5;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d1*d3;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d1*d4;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d1*d5;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      d = d2*d3;
      shell_ix += gout6_ix*d;
      shell_iy += gout6_iy*d;
      shell_iz += gout6_iz*d;
      shell_jx += gout6_jx*d;
      shell_jy += gout6_jy*d;
      shell_jz += gout6_jz*d;

      d = d2*d4;
      shell_ix += gout7_ix*d;
      shell_iy += gout7_iy*d;
      shell_iz += gout7_iz*d;
      shell_jx += gout7_jx*d;
      shell_jy += gout7_jy*d;
      shell_jz += gout7_jz*d;

      d = d2*d5;
      shell_ix += gout8_ix*d;
      shell_iy += gout8_iy*d;
      shell_iz += gout8_iz*d;
      shell_jx += gout8_jx*d;
      shell_jy += gout8_jy*d;
      shell_jz += gout8_jz*d;

      d0 = dm[(i0+0)+nao*(l0+0)];
      d1 = dm[(i0+1)+nao*(l0+0)];
      d2 = dm[(i0+2)+nao*(l0+0)];
      d3 = dm[(j0+0)+nao*(k0+0)];
      d4 = dm[(j0+1)+nao*(k0+0)];
      d5 = dm[(j0+2)+nao*(k0+0)];

      d = d0*d3;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d0*d4;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d0*d5;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d1*d3;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d1*d4;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d1*d5;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      d = d2*d3;
      shell_ix += gout6_ix*d;
      shell_iy += gout6_iy*d;
      shell_iz += gout6_iz*d;
      shell_jx += gout6_jx*d;
      shell_jy += gout6_jy*d;
      shell_jz += gout6_jz*d;

      d = d2*d4;
      shell_ix += gout7_ix*d;
      shell_iy += gout7_iy*d;
      shell_iz += gout7_iz*d;
      shell_jx += gout7_jx*d;
      shell_jy += gout7_jy*d;
      shell_jz += gout7_jz*d;

      d = d2*d5;
      shell_ix += gout8_ix*d;
      shell_iy += gout8_iy*d;
      shell_iz += gout8_iz*d;
      shell_jx += gout8_jx*d;
      shell_jy += gout8_jy*d;
      shell_jz += gout8_jz*d;

      atomicAdd(vk+ish*3  , shell_ix);
      atomicAdd(vk+ish*3+1, shell_iy);
      atomicAdd(vk+ish*3+2, shell_iz);
      atomicAdd(vk+jsh*3  , shell_jx);
      atomicAdd(vk+jsh*3+1, shell_jy);
      atomicAdd(vk+jsh*3+2, shell_jz);
    }
  }
}

__global__
static void GINTint2e_get_veff_ip1_kernel2000(GINTEnvVars envs,
                                              JKMatrix jk,
                                              BasisProdOffsets offsets)
{
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;
  double norm = envs.fac;
  double omega = envs.omega;
  int *bas_pair2bra = c_bpcache.bas_pair2bra;
  int *bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];
  int nprim_ij = envs.nprim_ij;
  int nprim_kl = envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;

  double* __restrict__ a12 = c_bpcache.a12;
  double* __restrict__ e12 = c_bpcache.e12;
  double* __restrict__ x12 = c_bpcache.x12;
  double* __restrict__ y12 = c_bpcache.y12;
  double* __restrict__ z12 = c_bpcache.z12;
  double * __restrict__ i_exponent = c_bpcache.a1;
  double * __restrict__ j_exponent = c_bpcache.a2;

  int ij, kl;
  int prim_ij0, prim_ij1, prim_kl0, prim_kl1;
  int nbas = c_bpcache.nbas;
  double* __restrict__ bas_x = c_bpcache.bas_coords;
  double* __restrict__ bas_y = bas_x + nbas;
  double* __restrict__ bas_z = bas_y + nbas;

  double gout0_ix = 0;
  double gout1_ix = 0;
  double gout2_ix = 0;
  double gout3_ix = 0;
  double gout4_ix = 0;
  double gout5_ix = 0;
  double gout0_iy = 0;
  double gout1_iy = 0;
  double gout2_iy = 0;
  double gout3_iy = 0;
  double gout4_iy = 0;
  double gout5_iy = 0;
  double gout0_iz = 0;
  double gout1_iz = 0;
  double gout2_iz = 0;
  double gout3_iz = 0;
  double gout4_iz = 0;
  double gout5_iz = 0;
  double gout0_jx = 0;
  double gout1_jx = 0;
  double gout2_jx = 0;
  double gout3_jx = 0;
  double gout4_jx = 0;
  double gout5_jx = 0;
  double gout0_jy = 0;
  double gout1_jy = 0;
  double gout2_jy = 0;
  double gout3_jy = 0;
  double gout4_jy = 0;
  double gout5_jy = 0;
  double gout0_jz = 0;
  double gout1_jz = 0;
  double gout2_jz = 0;
  double gout3_jz = 0;
  double gout4_jz = 0;
  double gout5_jz = 0;

  double xi = bas_x[ish];
  double yi = bas_y[ish];
  double zi = bas_z[ish];
  double ABx = xi - bas_x[jsh];
  double ABy = yi - bas_y[jsh];
  double ABz = zi - bas_z[jsh];


  prim_ij0 = prim_ij;
  prim_ij1 = prim_ij + nprim_ij;
  prim_kl0 = prim_kl;
  prim_kl1 = prim_kl + nprim_kl;
  double rw[4];
  int irys;
  for (ij = prim_ij0; ij < prim_ij1; ++ij) {
    double ai = i_exponent[ij] * 2.0;
    double aj = j_exponent[ij] * 2.0;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    for (kl = prim_kl0; kl < prim_kl1; ++kl) {
      double akl = a12[kl];
      double ekl = e12[kl];
      double xkl = x12[kl];
      double ykl = y12[kl];
      double zkl = z12[kl];
      double xijxkl = xij - xkl;
      double yijykl = yij - ykl;
      double zijzkl = zij - zkl;
      double aijkl = aij + akl;
      double a1 = aij * akl;
      double a0 = a1 / aijkl;
      double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0; 
      a0 *= theta;
      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      double fac = norm * eij * ekl * sqrt(a0 / (a1 * a1 * a1));
      //double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

      GINTrys_root<2>(x, rw);
      GINTscale_u<2>(rw, theta);
      for (irys = 0; irys < 2; ++irys) {
        double gz0 = rw[irys+2] * fac;
        double root0 = rw[irys];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double B00 = u2 * tmp4;
        double tmp1 = 2 * B00;
        double B10 = B00 + tmp4 * akl;
        double tmp2 = tmp1 * akl;
        double C00x = xij - xi - tmp2 * xijxkl;
        double C00y = yij - yi - tmp2 * yijykl;
        double C00z = zij - zi - tmp2 * zijzkl;



        double gx0 = 1;
        double gy0 = 1;
        double gx1,gx2,gx3,gx4,gx5,gx6;
        gx1 = C00x*gx0;
        gx2 = B10*gx0+C00x*gx1;
        gx3 = 2*B10*gx1+C00x*gx2;
        gx6 = ABx*gx2+gx3;
        gx5 = ABx*gx1+gx2;
        gx4 = ABx*gx0+gx1;
        double gy1,gy2,gy3,gy4,gy5,gy6;
        gy1 = C00y*gy0;
        gy2 = B10*gy0+C00y*gy1;
        gy3 = 2*B10*gy1+C00y*gy2;
        gy6 = ABy*gy2+gy3;
        gy5 = ABy*gy1+gy2;
        gy4 = ABy*gy0+gy1;
        double gz1,gz2,gz3,gz4,gz5,gz6;
        gz1 = C00z*gz0;
        gz2 = B10*gz0+C00z*gz1;
        gz3 = 2*B10*gz1+C00z*gz2;
        gz6 = ABz*gz2+gz3;
        gz5 = ABz*gz1+gz2;
        gz4 = ABz*gz0+gz1;

        gout0_ix += (-2*gx1+ai*gx3)*gy0*gz0;
        gout1_ix += (-gx0+ai*gx2)*gy1*gz0;
        gout2_ix += (-gx0+ai*gx2)*gy0*gz1;
        gout3_ix += (ai*gx1)*gy2*gz0;
        gout4_ix += (ai*gx1)*gy1*gz1;
        gout5_ix += (ai*gx1)*gy0*gz2;
        gout0_iy += gx2*(ai*gy1)*gz0;
        gout1_iy += gx1*(-gy0+ai*gy2)*gz0;
        gout2_iy += gx1*(ai*gy1)*gz1;
        gout3_iy += gx0*(-2*gy1+ai*gy3)*gz0;
        gout4_iy += gx0*(-gy0+ai*gy2)*gz1;
        gout5_iy += gx0*(ai*gy1)*gz2;
        gout0_iz += gx2*gy0*(ai*gz1);
        gout1_iz += gx1*gy1*(ai*gz1);
        gout2_iz += gx1*gy0*(-gz0+ai*gz2);
        gout3_iz += gx0*gy2*(ai*gz1);
        gout4_iz += gx0*gy1*(-gz0+ai*gz2);
        gout5_iz += gx0*gy0*(-2*gz1+ai*gz3);
        gout0_jx += (aj*gx6)*gy0*gz0;
        gout1_jx += (aj*gx5)*gy1*gz0;
        gout2_jx += (aj*gx5)*gy0*gz1;
        gout3_jx += (aj*gx4)*gy2*gz0;
        gout4_jx += (aj*gx4)*gy1*gz1;
        gout5_jx += (aj*gx4)*gy0*gz2;
        gout0_jy += gx2*(aj*gy4)*gz0;
        gout1_jy += gx1*(aj*gy5)*gz0;
        gout2_jy += gx1*(aj*gy4)*gz1;
        gout3_jy += gx0*(aj*gy6)*gz0;
        gout4_jy += gx0*(aj*gy5)*gz1;
        gout5_jy += gx0*(aj*gy4)*gz2;
        gout0_jz += gx2*gy0*(aj*gz4);
        gout1_jz += gx1*gy1*(aj*gz4);
        gout2_jz += gx1*gy0*(aj*gz5);
        gout3_jz += gx0*gy2*(aj*gz4);
        gout4_jz += gx0*gy1*(aj*gz5);
        gout5_jz += gx0*gy0*(aj*gz6);
      }
    } }

  int *ao_loc = c_bpcache.ao_loc;
  int i0 = ao_loc[ish];
  int j0 = ao_loc[jsh];
  int k0 = ao_loc[ksh];
  int l0 = ao_loc[lsh];
  int n_dm = jk.n_dm;
  int nao = jk.nao;
  double* __restrict__ dm = jk.dm;
  double *vj = jk.vj;
  double *vk = jk.vk;
  double d, d0, d1, d2, d3, d4, d5, d6;
  for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
    if(vj != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(j0+0)];
      d1 = dm[(i0+1)+nao*(j0+0)];
      d2 = dm[(i0+2)+nao*(j0+0)];
      d3 = dm[(i0+3)+nao*(j0+0)];
      d4 = dm[(i0+4)+nao*(j0+0)];
      d5 = dm[(i0+5)+nao*(j0+0)];
      d6 = dm[(k0+0)+nao*(l0+0)];

      d = d0*d6;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d1*d6;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d2*d6;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d3*d6;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d4*d6;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d5*d6;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      atomicAdd(vj+ish*3  , shell_ix);
      atomicAdd(vj+ish*3+1, shell_iy);
      atomicAdd(vj+ish*3+2, shell_iz);
      atomicAdd(vj+jsh*3  , shell_jx);
      atomicAdd(vj+jsh*3+1, shell_jy);
      atomicAdd(vj+jsh*3+2, shell_jz);
    }
    if(vk != NULL) {
      double shell_ix = 0, shell_iy = 0, shell_iz = 0, shell_jx = 0, shell_jy = 0, shell_jz = 0;
      d0 = dm[(i0+0)+nao*(k0+0)];
      d1 = dm[(i0+1)+nao*(k0+0)];
      d2 = dm[(i0+2)+nao*(k0+0)];
      d3 = dm[(i0+3)+nao*(k0+0)];
      d4 = dm[(i0+4)+nao*(k0+0)];
      d5 = dm[(i0+5)+nao*(k0+0)];
      d6 = dm[(j0+0)+nao*(l0+0)];

      d = d0*d6;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d1*d6;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d2*d6;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d3*d6;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d4*d6;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d5*d6;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      d0 = dm[(i0+0)+nao*(l0+0)];
      d1 = dm[(i0+1)+nao*(l0+0)];
      d2 = dm[(i0+2)+nao*(l0+0)];
      d3 = dm[(i0+3)+nao*(l0+0)];
      d4 = dm[(i0+4)+nao*(l0+0)];
      d5 = dm[(i0+5)+nao*(l0+0)];
      d6 = dm[(j0+0)+nao*(k0+0)];

      d = d0*d6;
      shell_ix += gout0_ix*d;
      shell_iy += gout0_iy*d;
      shell_iz += gout0_iz*d;
      shell_jx += gout0_jx*d;
      shell_jy += gout0_jy*d;
      shell_jz += gout0_jz*d;

      d = d1*d6;
      shell_ix += gout1_ix*d;
      shell_iy += gout1_iy*d;
      shell_iz += gout1_iz*d;
      shell_jx += gout1_jx*d;
      shell_jy += gout1_jy*d;
      shell_jz += gout1_jz*d;

      d = d2*d6;
      shell_ix += gout2_ix*d;
      shell_iy += gout2_iy*d;
      shell_iz += gout2_iz*d;
      shell_jx += gout2_jx*d;
      shell_jy += gout2_jy*d;
      shell_jz += gout2_jz*d;

      d = d3*d6;
      shell_ix += gout3_ix*d;
      shell_iy += gout3_iy*d;
      shell_iz += gout3_iz*d;
      shell_jx += gout3_jx*d;
      shell_jy += gout3_jy*d;
      shell_jz += gout3_jz*d;

      d = d4*d6;
      shell_ix += gout4_ix*d;
      shell_iy += gout4_iy*d;
      shell_iz += gout4_iz*d;
      shell_jx += gout4_jx*d;
      shell_jy += gout4_jy*d;
      shell_jz += gout4_jz*d;

      d = d5*d6;
      shell_ix += gout5_ix*d;
      shell_iy += gout5_iy*d;
      shell_iz += gout5_iz*d;
      shell_jx += gout5_jx*d;
      shell_jy += gout5_jy*d;
      shell_jz += gout5_jz*d;

      atomicAdd(vk+ish*3  , shell_ix);
      atomicAdd(vk+ish*3+1, shell_iy);
      atomicAdd(vk+ish*3+2, shell_iz);
      atomicAdd(vk+jsh*3  , shell_jx);
      atomicAdd(vk+jsh*3+1, shell_jy);
      atomicAdd(vk+jsh*3+2, shell_jz);
    }
  }
}
