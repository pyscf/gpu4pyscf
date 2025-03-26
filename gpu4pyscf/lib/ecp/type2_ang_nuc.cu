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
void type2_ang_nuc_l(double * __restrict__ omega, const int lc,
                    const int i, const int j, const int k,
                    double * __restrict__ unitr){
return;
}

template <> __device__
void type2_ang_nuc_l<0>(double * __restrict__ omega, const int lc,
                    const int i, const int j, const int k,
                    double * __restrict__ unitr){
    constexpr int l = 0;
    double rx[l+1], ry[l+1], rz[l+1];
    rx[0] = ry[0] = rz[0] = 1.0;
    for (int li = 1; li <= l; li++) {
        rx[li] = rx[li - 1] * unitr[0];
        ry[li] = ry[li - 1] * unitr[1];
        rz[li] = rz[li - 1] * unitr[2];
    }

    double c[2*l+1];
    for (int m = 0; m < 2*l+1; m++) c[m] = 0.0;
    c[0] += 0.28209479177387814*(rx[0]*ry[0]*rz[0]);;

    double buf[(ECP_LMAX+1)*(ECP_LMAX+2)/2];
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] = 0.0;
    double nuc;

    // l = 0, i = 0
    nuc = 0.0;
    nuc += c[0]*0.28209479177387814;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+0, k+pw+0);
    }

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] *= 4.0 * M_PI;
    cart2sph(omega, lc, buf);
}
template <> __device__
void type2_ang_nuc_l<1>(double * __restrict__ omega, const int lc,
                    const int i, const int j, const int k,
                    double * __restrict__ unitr){
    constexpr int l = 1;
    double rx[l+1], ry[l+1], rz[l+1];
    rx[0] = ry[0] = rz[0] = 1.0;
    for (int li = 1; li <= l; li++) {
        rx[li] = rx[li - 1] * unitr[0];
        ry[li] = ry[li - 1] * unitr[1];
        rz[li] = rz[li - 1] * unitr[2];
    }

    double c[2*l+1];
    for (int m = 0; m < 2*l+1; m++) c[m] = 0.0;
    c[0] += 0.4886025119029199*(rx[1]*ry[0]*rz[0]);
    c[1] += 0.4886025119029199*(rx[0]*ry[1]*rz[0]);
    c[2] += 0.4886025119029199*(rx[0]*ry[0]*rz[1]);;

    double buf[(ECP_LMAX+1)*(ECP_LMAX+2)/2];
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] = 0.0;
    double nuc;

    // l = 1, i = 0
    nuc = 0.0;
    nuc += c[0]*0.4886025119029199;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+0, k+pw+0);
    }
    // l = 1, i = 1
    nuc = 0.0;
    nuc += c[1]*0.4886025119029199;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+1, k+pw+0);
    }
    // l = 1, i = 2
    nuc = 0.0;
    nuc += c[2]*0.4886025119029199;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+0, k+pw+1);
    }

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] *= 4.0 * M_PI;
    cart2sph(omega, lc, buf);
}
template <> __device__
void type2_ang_nuc_l<2>(double * __restrict__ omega, const int lc,
                    const int i, const int j, const int k,
                    double * __restrict__ unitr){
    constexpr int l = 2;
    double rx[l+1], ry[l+1], rz[l+1];
    rx[0] = ry[0] = rz[0] = 1.0;
    for (int li = 1; li <= l; li++) {
        rx[li] = rx[li - 1] * unitr[0];
        ry[li] = ry[li - 1] * unitr[1];
        rz[li] = rz[li - 1] * unitr[2];
    }

    double c[2*l+1];
    for (int m = 0; m < 2*l+1; m++) c[m] = 0.0;
    c[0] += 1.0925484305920792*(rx[1]*ry[1]*rz[0]);
    c[1] += 1.0925484305920792*(rx[0]*ry[1]*rz[1]);
    c[2] += -0.31539156525252*(rx[2]*ry[0]*rz[0]);
    c[2] += -0.31539156525252*(rx[0]*ry[2]*rz[0]);
    c[2] += 0.63078313050504*(rx[0]*ry[0]*rz[2]);
    c[3] += 1.0925484305920792*(rx[1]*ry[0]*rz[1]);
    c[4] += 0.5462742152960396*(rx[2]*ry[0]*rz[0]);
    c[4] += -0.5462742152960396*(rx[0]*ry[2]*rz[0]);;

    double buf[(ECP_LMAX+1)*(ECP_LMAX+2)/2];
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] = 0.0;
    double nuc;

    // l = 2, i = 0
    nuc = 0.0;
    nuc += c[2]*-0.31539156525252;
    nuc += c[4]*0.5462742152960396;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+0, k+pw+0);
    }
    // l = 2, i = 1
    nuc = 0.0;
    nuc += c[0]*1.0925484305920792;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+1, k+pw+0);
    }
    // l = 2, i = 2
    nuc = 0.0;
    nuc += c[3]*1.0925484305920792;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+0, k+pw+1);
    }
    // l = 2, i = 3
    nuc = 0.0;
    nuc += c[2]*-0.31539156525252;
    nuc += c[4]*-0.5462742152960396;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+2, k+pw+0);
    }
    // l = 2, i = 4
    nuc = 0.0;
    nuc += c[1]*1.0925484305920792;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+1, k+pw+1);
    }
    // l = 2, i = 5
    nuc = 0.0;
    nuc += c[2]*0.63078313050504;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+0, k+pw+2);
    }

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] *= 4.0 * M_PI;
    cart2sph(omega, lc, buf);
}
template <> __device__
void type2_ang_nuc_l<3>(double * __restrict__ omega, const int lc,
                    const int i, const int j, const int k,
                    double * __restrict__ unitr){
    constexpr int l = 3;
    double rx[l+1], ry[l+1], rz[l+1];
    rx[0] = ry[0] = rz[0] = 1.0;
    for (int li = 1; li <= l; li++) {
        rx[li] = rx[li - 1] * unitr[0];
        ry[li] = ry[li - 1] * unitr[1];
        rz[li] = rz[li - 1] * unitr[2];
    }

    double c[2*l+1];
    for (int m = 0; m < 2*l+1; m++) c[m] = 0.0;
    c[0] += 1.7701307697799304*(rx[2]*ry[1]*rz[0]);
    c[0] += -0.5900435899266435*(rx[0]*ry[3]*rz[0]);
    c[1] += 2.8906114426405543*(rx[1]*ry[1]*rz[1]);
    c[2] += -0.4570457994644657*(rx[2]*ry[1]*rz[0]);
    c[2] += -0.4570457994644657*(rx[0]*ry[3]*rz[0]);
    c[2] += 1.8281831978578629*(rx[0]*ry[1]*rz[2]);
    c[3] += -1.1195289977703462*(rx[2]*ry[0]*rz[1]);
    c[3] += -1.1195289977703462*(rx[0]*ry[2]*rz[1]);
    c[3] += 0.7463526651802308*(rx[0]*ry[0]*rz[3]);
    c[4] += -0.4570457994644657*(rx[3]*ry[0]*rz[0]);
    c[4] += -0.4570457994644657*(rx[1]*ry[2]*rz[0]);
    c[4] += 1.8281831978578629*(rx[1]*ry[0]*rz[2]);
    c[5] += 1.4453057213202771*(rx[2]*ry[0]*rz[1]);
    c[5] += -1.4453057213202771*(rx[0]*ry[2]*rz[1]);
    c[6] += 0.5900435899266435*(rx[3]*ry[0]*rz[0]);
    c[6] += -1.7701307697799304*(rx[1]*ry[2]*rz[0]);;

    double buf[(ECP_LMAX+1)*(ECP_LMAX+2)/2];
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] = 0.0;
    double nuc;

    // l = 3, i = 0
    nuc = 0.0;
    nuc += c[4]*-0.4570457994644657;
    nuc += c[6]*0.5900435899266435;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+0, k+pw+0);
    }
    // l = 3, i = 1
    nuc = 0.0;
    nuc += c[0]*1.7701307697799304;
    nuc += c[2]*-0.4570457994644657;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+1, k+pw+0);
    }
    // l = 3, i = 2
    nuc = 0.0;
    nuc += c[3]*-1.1195289977703462;
    nuc += c[5]*1.4453057213202771;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+0, k+pw+1);
    }
    // l = 3, i = 3
    nuc = 0.0;
    nuc += c[4]*-0.4570457994644657;
    nuc += c[6]*-1.7701307697799304;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+2, k+pw+0);
    }
    // l = 3, i = 4
    nuc = 0.0;
    nuc += c[1]*2.8906114426405543;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+1, k+pw+1);
    }
    // l = 3, i = 5
    nuc = 0.0;
    nuc += c[4]*1.8281831978578629;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+0, k+pw+2);
    }
    // l = 3, i = 6
    nuc = 0.0;
    nuc += c[0]*-0.5900435899266435;
    nuc += c[2]*-0.4570457994644657;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+3, k+pw+0);
    }
    // l = 3, i = 7
    nuc = 0.0;
    nuc += c[3]*-1.1195289977703462;
    nuc += c[5]*-1.4453057213202771;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+2, k+pw+1);
    }
    // l = 3, i = 8
    nuc = 0.0;
    nuc += c[2]*1.8281831978578629;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+1, k+pw+2);
    }
    // l = 3, i = 9
    nuc = 0.0;
    nuc += c[3]*0.7463526651802308;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+0, k+pw+3);
    }

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] *= 4.0 * M_PI;
    cart2sph(omega, lc, buf);
}
template <> __device__
void type2_ang_nuc_l<4>(double * __restrict__ omega, const int lc,
                    const int i, const int j, const int k,
                    double * __restrict__ unitr){
    constexpr int l = 4;
    double rx[l+1], ry[l+1], rz[l+1];
    rx[0] = ry[0] = rz[0] = 1.0;
    for (int li = 1; li <= l; li++) {
        rx[li] = rx[li - 1] * unitr[0];
        ry[li] = ry[li - 1] * unitr[1];
        rz[li] = rz[li - 1] * unitr[2];
    }

    double c[2*l+1];
    for (int m = 0; m < 2*l+1; m++) c[m] = 0.0;
    c[0] += 2.5033429417967046*(rx[3]*ry[1]*rz[0]);
    c[0] += -2.5033429417967046*(rx[1]*ry[3]*rz[0]);
    c[1] += 5.310392309339791*(rx[2]*ry[1]*rz[1]);
    c[1] += -1.7701307697799304*(rx[0]*ry[3]*rz[1]);
    c[2] += -0.94617469575756*(rx[3]*ry[1]*rz[0]);
    c[2] += -0.94617469575756*(rx[1]*ry[3]*rz[0]);
    c[2] += 5.6770481745453605*(rx[1]*ry[1]*rz[2]);
    c[3] += -2.0071396306718676*(rx[2]*ry[1]*rz[1]);
    c[3] += -2.0071396306718676*(rx[0]*ry[3]*rz[1]);
    c[3] += 2.676186174229157*(rx[0]*ry[1]*rz[3]);
    c[4] += 0.31735664074561293*(rx[4]*ry[0]*rz[0]);
    c[4] += 0.6347132814912259*(rx[2]*ry[2]*rz[0]);
    c[4] += -2.5388531259649034*(rx[2]*ry[0]*rz[2]);
    c[4] += 0.31735664074561293*(rx[0]*ry[4]*rz[0]);
    c[4] += -2.5388531259649034*(rx[0]*ry[2]*rz[2]);
    c[4] += 0.8462843753216345*(rx[0]*ry[0]*rz[4]);
    c[5] += -2.0071396306718676*(rx[3]*ry[0]*rz[1]);
    c[5] += -2.0071396306718676*(rx[1]*ry[2]*rz[1]);
    c[5] += 2.676186174229157*(rx[1]*ry[0]*rz[3]);
    c[6] += -0.47308734787878*(rx[4]*ry[0]*rz[0]);
    c[6] += 2.8385240872726802*(rx[2]*ry[0]*rz[2]);
    c[6] += 0.47308734787878*(rx[0]*ry[4]*rz[0]);
    c[6] += -2.8385240872726802*(rx[0]*ry[2]*rz[2]);
    c[7] += 1.7701307697799304*(rx[3]*ry[0]*rz[1]);
    c[7] += -5.310392309339791*(rx[1]*ry[2]*rz[1]);
    c[8] += 0.6258357354491761*(rx[4]*ry[0]*rz[0]);
    c[8] += -3.755014412695057*(rx[2]*ry[2]*rz[0]);
    c[8] += 0.6258357354491761*(rx[0]*ry[4]*rz[0]);;

    double buf[(ECP_LMAX+1)*(ECP_LMAX+2)/2];
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] = 0.0;
    double nuc;

    // l = 4, i = 0
    nuc = 0.0;
    nuc += c[4]*0.31735664074561293;
    nuc += c[6]*-0.47308734787878;
    nuc += c[8]*0.6258357354491761;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+0, k+pw+0);
    }
    // l = 4, i = 1
    nuc = 0.0;
    nuc += c[0]*2.5033429417967046;
    nuc += c[2]*-0.94617469575756;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+1, k+pw+0);
    }
    // l = 4, i = 2
    nuc = 0.0;
    nuc += c[5]*-2.0071396306718676;
    nuc += c[7]*1.7701307697799304;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+0, k+pw+1);
    }
    // l = 4, i = 3
    nuc = 0.0;
    nuc += c[4]*0.6347132814912259;
    nuc += c[8]*-3.755014412695057;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+2, k+pw+0);
    }
    // l = 4, i = 4
    nuc = 0.0;
    nuc += c[1]*5.310392309339791;
    nuc += c[3]*-2.0071396306718676;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+1, k+pw+1);
    }
    // l = 4, i = 5
    nuc = 0.0;
    nuc += c[4]*-2.5388531259649034;
    nuc += c[6]*2.8385240872726802;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+0, k+pw+2);
    }
    // l = 4, i = 6
    nuc = 0.0;
    nuc += c[0]*-2.5033429417967046;
    nuc += c[2]*-0.94617469575756;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+3, k+pw+0);
    }
    // l = 4, i = 7
    nuc = 0.0;
    nuc += c[5]*-2.0071396306718676;
    nuc += c[7]*-5.310392309339791;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+2, k+pw+1);
    }
    // l = 4, i = 8
    nuc = 0.0;
    nuc += c[2]*5.6770481745453605;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+1, k+pw+2);
    }
    // l = 4, i = 9
    nuc = 0.0;
    nuc += c[5]*2.676186174229157;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+0, k+pw+3);
    }
    // l = 4, i = 10
    nuc = 0.0;
    nuc += c[4]*0.31735664074561293;
    nuc += c[6]*0.47308734787878;
    nuc += c[8]*0.6258357354491761;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+4, k+pw+0);
    }
    // l = 4, i = 11
    nuc = 0.0;
    nuc += c[1]*-1.7701307697799304;
    nuc += c[3]*-2.0071396306718676;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+3, k+pw+1);
    }
    // l = 4, i = 12
    nuc = 0.0;
    nuc += c[4]*-2.5388531259649034;
    nuc += c[6]*-2.8385240872726802;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+2, k+pw+2);
    }
    // l = 4, i = 13
    nuc = 0.0;
    nuc += c[3]*2.676186174229157;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+1, k+pw+3);
    }
    // l = 4, i = 14
    nuc = 0.0;
    nuc += c[4]*0.8462843753216345;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+0, k+pw+4);
    }

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] *= 4.0 * M_PI;
    cart2sph(omega, lc, buf);
}
template <> __device__
void type2_ang_nuc_l<5>(double * __restrict__ omega, const int lc,
                    const int i, const int j, const int k,
                    double * __restrict__ unitr){
    constexpr int l = 5;
    double rx[l+1], ry[l+1], rz[l+1];
    rx[0] = ry[0] = rz[0] = 1.0;
    for (int li = 1; li <= l; li++) {
        rx[li] = rx[li - 1] * unitr[0];
        ry[li] = ry[li - 1] * unitr[1];
        rz[li] = rz[li - 1] * unitr[2];
    }

    double c[2*l+1];
    for (int m = 0; m < 2*l+1; m++) c[m] = 0.0;
    c[0] += 3.2819102842008507*(rx[4]*ry[1]*rz[0]);
    c[0] += -6.563820568401701*(rx[2]*ry[3]*rz[0]);
    c[0] += 0.6563820568401701*(rx[0]*ry[5]*rz[0]);
    c[1] += 8.302649259524165*(rx[3]*ry[1]*rz[1]);
    c[1] += -8.302649259524165*(rx[1]*ry[3]*rz[1]);
    c[2] += -1.467714898305751*(rx[4]*ry[1]*rz[0]);
    c[2] += -0.9784765988705008*(rx[2]*ry[3]*rz[0]);
    c[2] += 11.741719186446009*(rx[2]*ry[1]*rz[2]);
    c[2] += 0.4892382994352504*(rx[0]*ry[5]*rz[0]);
    c[2] += -3.913906395482003*(rx[0]*ry[3]*rz[2]);
    c[3] += -4.793536784973324*(rx[3]*ry[1]*rz[1]);
    c[3] += -4.793536784973324*(rx[1]*ry[3]*rz[1]);
    c[3] += 9.587073569946648*(rx[1]*ry[1]*rz[3]);
    c[4] += 0.45294665119569694*(rx[4]*ry[1]*rz[0]);
    c[4] += 0.9058933023913939*(rx[2]*ry[3]*rz[0]);
    c[4] += -5.435359814348363*(rx[2]*ry[1]*rz[2]);
    c[4] += 0.45294665119569694*(rx[0]*ry[5]*rz[0]);
    c[4] += -5.435359814348363*(rx[0]*ry[3]*rz[2]);
    c[4] += 3.6235732095655755*(rx[0]*ry[1]*rz[4]);
    c[5] += 1.754254836801354*(rx[4]*ry[0]*rz[1]);
    c[5] += 3.508509673602708*(rx[2]*ry[2]*rz[1]);
    c[5] += -4.678012898136944*(rx[2]*ry[0]*rz[3]);
    c[5] += 1.754254836801354*(rx[0]*ry[4]*rz[1]);
    c[5] += -4.678012898136944*(rx[0]*ry[2]*rz[3]);
    c[5] += 0.9356025796273888*(rx[0]*ry[0]*rz[5]);
    c[6] += 0.45294665119569694*(rx[5]*ry[0]*rz[0]);
    c[6] += 0.9058933023913939*(rx[3]*ry[2]*rz[0]);
    c[6] += -5.435359814348363*(rx[3]*ry[0]*rz[2]);
    c[6] += 0.45294665119569694*(rx[1]*ry[4]*rz[0]);
    c[6] += -5.435359814348363*(rx[1]*ry[2]*rz[2]);
    c[6] += 3.6235732095655755*(rx[1]*ry[0]*rz[4]);
    c[7] += -2.396768392486662*(rx[4]*ry[0]*rz[1]);
    c[7] += 4.793536784973324*(rx[2]*ry[0]*rz[3]);
    c[7] += 2.396768392486662*(rx[0]*ry[4]*rz[1]);
    c[7] += -4.793536784973324*(rx[0]*ry[2]*rz[3]);
    c[8] += -0.4892382994352504*(rx[5]*ry[0]*rz[0]);
    c[8] += 0.9784765988705008*(rx[3]*ry[2]*rz[0]);
    c[8] += 3.913906395482003*(rx[3]*ry[0]*rz[2]);
    c[8] += 1.467714898305751*(rx[1]*ry[4]*rz[0]);
    c[8] += -11.741719186446009*(rx[1]*ry[2]*rz[2]);
    c[9] += 2.075662314881041*(rx[4]*ry[0]*rz[1]);
    c[9] += -12.453973889286248*(rx[2]*ry[2]*rz[1]);
    c[9] += 2.075662314881041*(rx[0]*ry[4]*rz[1]);
    c[10] += 0.6563820568401701*(rx[5]*ry[0]*rz[0]);
    c[10] += -6.563820568401701*(rx[3]*ry[2]*rz[0]);
    c[10] += 3.2819102842008507*(rx[1]*ry[4]*rz[0]);;

    double buf[(ECP_LMAX+1)*(ECP_LMAX+2)/2];
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] = 0.0;
    double nuc;

    // l = 5, i = 0
    nuc = 0.0;
    nuc += c[6]*0.45294665119569694;
    nuc += c[8]*-0.4892382994352504;
    nuc += c[10]*0.6563820568401701;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+0, k+pw+0);
    }
    // l = 5, i = 1
    nuc = 0.0;
    nuc += c[0]*3.2819102842008507;
    nuc += c[2]*-1.467714898305751;
    nuc += c[4]*0.45294665119569694;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+1, k+pw+0);
    }
    // l = 5, i = 2
    nuc = 0.0;
    nuc += c[5]*1.754254836801354;
    nuc += c[7]*-2.396768392486662;
    nuc += c[9]*2.075662314881041;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+0, k+pw+1);
    }
    // l = 5, i = 3
    nuc = 0.0;
    nuc += c[6]*0.9058933023913939;
    nuc += c[8]*0.9784765988705008;
    nuc += c[10]*-6.563820568401701;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+2, k+pw+0);
    }
    // l = 5, i = 4
    nuc = 0.0;
    nuc += c[1]*8.302649259524165;
    nuc += c[3]*-4.793536784973324;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+1, k+pw+1);
    }
    // l = 5, i = 5
    nuc = 0.0;
    nuc += c[6]*-5.435359814348363;
    nuc += c[8]*3.913906395482003;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+0, k+pw+2);
    }
    // l = 5, i = 6
    nuc = 0.0;
    nuc += c[0]*-6.563820568401701;
    nuc += c[2]*-0.9784765988705008;
    nuc += c[4]*0.9058933023913939;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+3, k+pw+0);
    }
    // l = 5, i = 7
    nuc = 0.0;
    nuc += c[5]*3.508509673602708;
    nuc += c[9]*-12.453973889286248;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+2, k+pw+1);
    }
    // l = 5, i = 8
    nuc = 0.0;
    nuc += c[2]*11.741719186446009;
    nuc += c[4]*-5.435359814348363;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+1, k+pw+2);
    }
    // l = 5, i = 9
    nuc = 0.0;
    nuc += c[5]*-4.678012898136944;
    nuc += c[7]*4.793536784973324;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+0, k+pw+3);
    }
    // l = 5, i = 10
    nuc = 0.0;
    nuc += c[6]*0.45294665119569694;
    nuc += c[8]*1.467714898305751;
    nuc += c[10]*3.2819102842008507;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+4, k+pw+0);
    }
    // l = 5, i = 11
    nuc = 0.0;
    nuc += c[1]*-8.302649259524165;
    nuc += c[3]*-4.793536784973324;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+3, k+pw+1);
    }
    // l = 5, i = 12
    nuc = 0.0;
    nuc += c[6]*-5.435359814348363;
    nuc += c[8]*-11.741719186446009;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+2, k+pw+2);
    }
    // l = 5, i = 13
    nuc = 0.0;
    nuc += c[3]*9.587073569946648;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+1, k+pw+3);
    }
    // l = 5, i = 14
    nuc = 0.0;
    nuc += c[6]*3.6235732095655755;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+0, k+pw+4);
    }
    // l = 5, i = 15
    nuc = 0.0;
    nuc += c[0]*0.6563820568401701;
    nuc += c[2]*0.4892382994352504;
    nuc += c[4]*0.45294665119569694;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+5, k+pw+0);
    }
    // l = 5, i = 16
    nuc = 0.0;
    nuc += c[5]*1.754254836801354;
    nuc += c[7]*2.396768392486662;
    nuc += c[9]*2.075662314881041;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+4, k+pw+1);
    }
    // l = 5, i = 17
    nuc = 0.0;
    nuc += c[2]*-3.913906395482003;
    nuc += c[4]*-5.435359814348363;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+3, k+pw+2);
    }
    // l = 5, i = 18
    nuc = 0.0;
    nuc += c[5]*-4.678012898136944;
    nuc += c[7]*-4.793536784973324;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+2, k+pw+3);
    }
    // l = 5, i = 19
    nuc = 0.0;
    nuc += c[4]*3.6235732095655755;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+1, k+pw+4);
    }
    // l = 5, i = 20
    nuc = 0.0;
    nuc += c[5]*0.9356025796273888;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+0, k+pw+5);
    }

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] *= 4.0 * M_PI;
    cart2sph(omega, lc, buf);
}
template <> __device__
void type2_ang_nuc_l<6>(double * __restrict__ omega, const int lc,
                    const int i, const int j, const int k,
                    double * __restrict__ unitr){
    constexpr int l = 6;
    double rx[l+1], ry[l+1], rz[l+1];
    rx[0] = ry[0] = rz[0] = 1.0;
    for (int li = 1; li <= l; li++) {
        rx[li] = rx[li - 1] * unitr[0];
        ry[li] = ry[li - 1] * unitr[1];
        rz[li] = rz[li - 1] * unitr[2];
    }

    double c[2*l+1];
    for (int m = 0; m < 2*l+1; m++) c[m] = 0.0;
    c[0] += 4.099104631151486*(rx[5]*ry[1]*rz[0]);
    c[0] += -13.663682103838289*(rx[3]*ry[3]*rz[0]);
    c[0] += 4.099104631151486*(rx[1]*ry[5]*rz[0]);
    c[1] += 11.833095811158763*(rx[4]*ry[1]*rz[1]);
    c[1] += -23.666191622317527*(rx[2]*ry[3]*rz[1]);
    c[1] += 2.3666191622317525*(rx[0]*ry[5]*rz[1]);
    c[2] += -2.0182596029148963*(rx[5]*ry[1]*rz[0]);
    c[2] += 20.182596029148968*(rx[3]*ry[1]*rz[2]);
    c[2] += 2.0182596029148963*(rx[1]*ry[5]*rz[0]);
    c[2] += -20.182596029148968*(rx[1]*ry[3]*rz[2]);
    c[3] += -8.29084733563431*(rx[4]*ry[1]*rz[1]);
    c[3] += -5.527231557089541*(rx[2]*ry[3]*rz[1]);
    c[3] += 22.108926228358165*(rx[2]*ry[1]*rz[3]);
    c[3] += 2.7636157785447706*(rx[0]*ry[5]*rz[1]);
    c[3] += -7.369642076119389*(rx[0]*ry[3]*rz[3]);
    c[4] += 0.9212052595149236*(rx[5]*ry[1]*rz[0]);
    c[4] += 1.8424105190298472*(rx[3]*ry[3]*rz[0]);
    c[4] += -14.739284152238778*(rx[3]*ry[1]*rz[2]);
    c[4] += 0.9212052595149236*(rx[1]*ry[5]*rz[0]);
    c[4] += -14.739284152238778*(rx[1]*ry[3]*rz[2]);
    c[4] += 14.739284152238778*(rx[1]*ry[1]*rz[4]);
    c[5] += 2.913106812593657*(rx[4]*ry[1]*rz[1]);
    c[5] += 5.826213625187314*(rx[2]*ry[3]*rz[1]);
    c[5] += -11.652427250374627*(rx[2]*ry[1]*rz[3]);
    c[5] += 2.913106812593657*(rx[0]*ry[5]*rz[1]);
    c[5] += -11.652427250374627*(rx[0]*ry[3]*rz[3]);
    c[5] += 4.6609709001498505*(rx[0]*ry[1]*rz[5]);
    c[6] += -0.3178460113381421*(rx[6]*ry[0]*rz[0]);
    c[6] += -0.9535380340144264*(rx[4]*ry[2]*rz[0]);
    c[6] += 5.721228204086558*(rx[4]*ry[0]*rz[2]);
    c[6] += -0.9535380340144264*(rx[2]*ry[4]*rz[0]);
    c[6] += 11.442456408173117*(rx[2]*ry[2]*rz[2]);
    c[6] += -7.628304272115411*(rx[2]*ry[0]*rz[4]);
    c[6] += -0.3178460113381421*(rx[0]*ry[6]*rz[0]);
    c[6] += 5.721228204086558*(rx[0]*ry[4]*rz[2]);
    c[6] += -7.628304272115411*(rx[0]*ry[2]*rz[4]);
    c[6] += 1.0171072362820548*(rx[0]*ry[0]*rz[6]);
    c[7] += 2.913106812593657*(rx[5]*ry[0]*rz[1]);
    c[7] += 5.826213625187314*(rx[3]*ry[2]*rz[1]);
    c[7] += -11.652427250374627*(rx[3]*ry[0]*rz[3]);
    c[7] += 2.913106812593657*(rx[1]*ry[4]*rz[1]);
    c[7] += -11.652427250374627*(rx[1]*ry[2]*rz[3]);
    c[7] += 4.6609709001498505*(rx[1]*ry[0]*rz[5]);
    c[8] += 0.4606026297574618*(rx[6]*ry[0]*rz[0]);
    c[8] += 0.4606026297574618*(rx[4]*ry[2]*rz[0]);
    c[8] += -7.369642076119389*(rx[4]*ry[0]*rz[2]);
    c[8] += -0.4606026297574618*(rx[2]*ry[4]*rz[0]);
    c[8] += 7.369642076119389*(rx[2]*ry[0]*rz[4]);
    c[8] += -0.4606026297574618*(rx[0]*ry[6]*rz[0]);
    c[8] += 7.369642076119389*(rx[0]*ry[4]*rz[2]);
    c[8] += -7.369642076119389*(rx[0]*ry[2]*rz[4]);
    c[9] += -2.7636157785447706*(rx[5]*ry[0]*rz[1]);
    c[9] += 5.527231557089541*(rx[3]*ry[2]*rz[1]);
    c[9] += 7.369642076119389*(rx[3]*ry[0]*rz[3]);
    c[9] += 8.29084733563431*(rx[1]*ry[4]*rz[1]);
    c[9] += -22.108926228358165*(rx[1]*ry[2]*rz[3]);
    c[10] += -0.5045649007287241*(rx[6]*ry[0]*rz[0]);
    c[10] += 2.52282450364362*(rx[4]*ry[2]*rz[0]);
    c[10] += 5.045649007287242*(rx[4]*ry[0]*rz[2]);
    c[10] += 2.52282450364362*(rx[2]*ry[4]*rz[0]);
    c[10] += -30.273894043723452*(rx[2]*ry[2]*rz[2]);
    c[10] += -0.5045649007287241*(rx[0]*ry[6]*rz[0]);
    c[10] += 5.045649007287242*(rx[0]*ry[4]*rz[2]);
    c[11] += 2.3666191622317525*(rx[5]*ry[0]*rz[1]);
    c[11] += -23.666191622317527*(rx[3]*ry[2]*rz[1]);
    c[11] += 11.833095811158763*(rx[1]*ry[4]*rz[1]);
    c[12] += 0.6831841051919144*(rx[6]*ry[0]*rz[0]);
    c[12] += -10.247761577878716*(rx[4]*ry[2]*rz[0]);
    c[12] += 10.247761577878716*(rx[2]*ry[4]*rz[0]);
    c[12] += -0.6831841051919144*(rx[0]*ry[6]*rz[0]);;

    double buf[(ECP_LMAX+1)*(ECP_LMAX+2)/2];
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] = 0.0;
    double nuc;

    // l = 6, i = 0
    nuc = 0.0;
    nuc += c[6]*-0.3178460113381421;
    nuc += c[8]*0.4606026297574618;
    nuc += c[10]*-0.5045649007287241;
    nuc += c[12]*0.6831841051919144;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+0, k+pw+0);
    }
    // l = 6, i = 1
    nuc = 0.0;
    nuc += c[0]*4.099104631151486;
    nuc += c[2]*-2.0182596029148963;
    nuc += c[4]*0.9212052595149236;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+1, k+pw+0);
    }
    // l = 6, i = 2
    nuc = 0.0;
    nuc += c[7]*2.913106812593657;
    nuc += c[9]*-2.7636157785447706;
    nuc += c[11]*2.3666191622317525;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+0, k+pw+1);
    }
    // l = 6, i = 3
    nuc = 0.0;
    nuc += c[6]*-0.9535380340144264;
    nuc += c[8]*0.4606026297574618;
    nuc += c[10]*2.52282450364362;
    nuc += c[12]*-10.247761577878716;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+2, k+pw+0);
    }
    // l = 6, i = 4
    nuc = 0.0;
    nuc += c[1]*11.833095811158763;
    nuc += c[3]*-8.29084733563431;
    nuc += c[5]*2.913106812593657;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+1, k+pw+1);
    }
    // l = 6, i = 5
    nuc = 0.0;
    nuc += c[6]*5.721228204086558;
    nuc += c[8]*-7.369642076119389;
    nuc += c[10]*5.045649007287242;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+0, k+pw+2);
    }
    // l = 6, i = 6
    nuc = 0.0;
    nuc += c[0]*-13.663682103838289;
    nuc += c[4]*1.8424105190298472;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+3, k+pw+0);
    }
    // l = 6, i = 7
    nuc = 0.0;
    nuc += c[7]*5.826213625187314;
    nuc += c[9]*5.527231557089541;
    nuc += c[11]*-23.666191622317527;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+2, k+pw+1);
    }
    // l = 6, i = 8
    nuc = 0.0;
    nuc += c[2]*20.182596029148968;
    nuc += c[4]*-14.739284152238778;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+1, k+pw+2);
    }
    // l = 6, i = 9
    nuc = 0.0;
    nuc += c[7]*-11.652427250374627;
    nuc += c[9]*7.369642076119389;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+0, k+pw+3);
    }
    // l = 6, i = 10
    nuc = 0.0;
    nuc += c[6]*-0.9535380340144264;
    nuc += c[8]*-0.4606026297574618;
    nuc += c[10]*2.52282450364362;
    nuc += c[12]*10.247761577878716;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+4, k+pw+0);
    }
    // l = 6, i = 11
    nuc = 0.0;
    nuc += c[1]*-23.666191622317527;
    nuc += c[3]*-5.527231557089541;
    nuc += c[5]*5.826213625187314;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+3, k+pw+1);
    }
    // l = 6, i = 12
    nuc = 0.0;
    nuc += c[6]*11.442456408173117;
    nuc += c[10]*-30.273894043723452;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+2, k+pw+2);
    }
    // l = 6, i = 13
    nuc = 0.0;
    nuc += c[3]*22.108926228358165;
    nuc += c[5]*-11.652427250374627;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+1, k+pw+3);
    }
    // l = 6, i = 14
    nuc = 0.0;
    nuc += c[6]*-7.628304272115411;
    nuc += c[8]*7.369642076119389;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+0, k+pw+4);
    }
    // l = 6, i = 15
    nuc = 0.0;
    nuc += c[0]*4.099104631151486;
    nuc += c[2]*2.0182596029148963;
    nuc += c[4]*0.9212052595149236;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+5, k+pw+0);
    }
    // l = 6, i = 16
    nuc = 0.0;
    nuc += c[7]*2.913106812593657;
    nuc += c[9]*8.29084733563431;
    nuc += c[11]*11.833095811158763;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+4, k+pw+1);
    }
    // l = 6, i = 17
    nuc = 0.0;
    nuc += c[2]*-20.182596029148968;
    nuc += c[4]*-14.739284152238778;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+3, k+pw+2);
    }
    // l = 6, i = 18
    nuc = 0.0;
    nuc += c[7]*-11.652427250374627;
    nuc += c[9]*-22.108926228358165;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+2, k+pw+3);
    }
    // l = 6, i = 19
    nuc = 0.0;
    nuc += c[4]*14.739284152238778;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+1, k+pw+4);
    }
    // l = 6, i = 20
    nuc = 0.0;
    nuc += c[7]*4.6609709001498505;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+0, k+pw+5);
    }
    // l = 6, i = 21
    nuc = 0.0;
    nuc += c[6]*-0.3178460113381421;
    nuc += c[8]*-0.4606026297574618;
    nuc += c[10]*-0.5045649007287241;
    nuc += c[12]*-0.6831841051919144;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+6, k+pw+0);
    }
    // l = 6, i = 22
    nuc = 0.0;
    nuc += c[1]*2.3666191622317525;
    nuc += c[3]*2.7636157785447706;
    nuc += c[5]*2.913106812593657;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+5, k+pw+1);
    }
    // l = 6, i = 23
    nuc = 0.0;
    nuc += c[6]*5.721228204086558;
    nuc += c[8]*7.369642076119389;
    nuc += c[10]*5.045649007287242;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+4, k+pw+2);
    }
    // l = 6, i = 24
    nuc = 0.0;
    nuc += c[3]*-7.369642076119389;
    nuc += c[5]*-11.652427250374627;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+3, k+pw+3);
    }
    // l = 6, i = 25
    nuc = 0.0;
    nuc += c[6]*-7.628304272115411;
    nuc += c[8]*-7.369642076119389;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+2, k+pw+4);
    }
    // l = 6, i = 26
    nuc = 0.0;
    nuc += c[5]*4.6609709001498505;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+1, k+pw+5);
    }
    // l = 6, i = 27
    nuc = 0.0;
    nuc += c[6]*1.0171072362820548;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+0, k+pw+6);
    }

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] *= 4.0 * M_PI;
    cart2sph(omega, lc, buf);
}
template <> __device__
void type2_ang_nuc_l<7>(double * __restrict__ omega, const int lc,
                    const int i, const int j, const int k,
                    double * __restrict__ unitr){
    constexpr int l = 7;
    double rx[l+1], ry[l+1], rz[l+1];
    rx[0] = ry[0] = rz[0] = 1.0;
    for (int li = 1; li <= l; li++) {
        rx[li] = rx[li - 1] * unitr[0];
        ry[li] = ry[li - 1] * unitr[1];
        rz[li] = rz[li - 1] * unitr[2];
    }

    double c[2*l+1];
    for (int m = 0; m < 2*l+1; m++) c[m] = 0.0;
    c[0] += 4.950139127672174*(rx[6]*ry[1]*rz[0]);
    c[0] += -24.75069563836087*(rx[4]*ry[3]*rz[0]);
    c[0] += 14.850417383016522*(rx[2]*ry[5]*rz[0]);
    c[0] += -0.7071627325245963*(rx[0]*ry[7]*rz[0]);
    c[1] += 15.8757639708114*(rx[5]*ry[1]*rz[1]);
    c[1] += -52.919213236038004*(rx[3]*ry[3]*rz[1]);
    c[1] += 15.8757639708114*(rx[1]*ry[5]*rz[1]);
    c[2] += -2.594577893601302*(rx[6]*ry[1]*rz[0]);
    c[2] += 2.594577893601302*(rx[4]*ry[3]*rz[0]);
    c[2] += 31.134934723215622*(rx[4]*ry[1]*rz[2]);
    c[2] += 4.670240208482344*(rx[2]*ry[5]*rz[0]);
    c[2] += -62.269869446431244*(rx[2]*ry[3]*rz[2]);
    c[2] += -0.5189155787202604*(rx[0]*ry[7]*rz[0]);
    c[2] += 6.226986944643125*(rx[0]*ry[5]*rz[2]);
    c[3] += -12.45397388928625*(rx[5]*ry[1]*rz[1]);
    c[3] += 41.51324629762083*(rx[3]*ry[1]*rz[3]);
    c[3] += 12.45397388928625*(rx[1]*ry[5]*rz[1]);
    c[3] += -41.51324629762083*(rx[1]*ry[3]*rz[3]);
    c[4] += 1.4081304047606462*(rx[6]*ry[1]*rz[0]);
    c[4] += 2.3468840079344107*(rx[4]*ry[3]*rz[0]);
    c[4] += -28.162608095212924*(rx[4]*ry[1]*rz[2]);
    c[4] += 0.4693768015868821*(rx[2]*ry[5]*rz[0]);
    c[4] += -18.77507206347528*(rx[2]*ry[3]*rz[2]);
    c[4] += 37.55014412695057*(rx[2]*ry[1]*rz[4]);
    c[4] += -0.4693768015868821*(rx[0]*ry[7]*rz[0]);
    c[4] += 9.38753603173764*(rx[0]*ry[5]*rz[2]);
    c[4] += -12.516714708983523*(rx[0]*ry[3]*rz[4]);
    c[5] += 6.637990386674741*(rx[5]*ry[1]*rz[1]);
    c[5] += 13.275980773349483*(rx[3]*ry[3]*rz[1]);
    c[5] += -35.402615395598616*(rx[3]*ry[1]*rz[3]);
    c[5] += 6.637990386674741*(rx[1]*ry[5]*rz[1]);
    c[5] += -35.402615395598616*(rx[1]*ry[3]*rz[3]);
    c[5] += 21.241569237359172*(rx[1]*ry[1]*rz[5]);
    c[6] += -0.4516580379125866*(rx[6]*ry[1]*rz[0]);
    c[6] += -1.35497411373776*(rx[4]*ry[3]*rz[0]);
    c[6] += 10.839792909902078*(rx[4]*ry[1]*rz[2]);
    c[6] += -1.35497411373776*(rx[2]*ry[5]*rz[0]);
    c[6] += 21.679585819804156*(rx[2]*ry[3]*rz[2]);
    c[6] += -21.679585819804156*(rx[2]*ry[1]*rz[4]);
    c[6] += -0.4516580379125866*(rx[0]*ry[7]*rz[0]);
    c[6] += 10.839792909902078*(rx[0]*ry[5]*rz[2]);
    c[6] += -21.679585819804156*(rx[0]*ry[3]*rz[4]);
    c[6] += 5.781222885281109*(rx[0]*ry[1]*rz[6]);
    c[7] += -2.389949691920173*(rx[6]*ry[0]*rz[1]);
    c[7] += -7.169849075760519*(rx[4]*ry[2]*rz[1]);
    c[7] += 14.339698151521036*(rx[4]*ry[0]*rz[3]);
    c[7] += -7.169849075760519*(rx[2]*ry[4]*rz[1]);
    c[7] += 28.679396303042072*(rx[2]*ry[2]*rz[3]);
    c[7] += -11.47175852121683*(rx[2]*ry[0]*rz[5]);
    c[7] += -2.389949691920173*(rx[0]*ry[6]*rz[1]);
    c[7] += 14.339698151521036*(rx[0]*ry[4]*rz[3]);
    c[7] += -11.47175852121683*(rx[0]*ry[2]*rz[5]);
    c[7] += 1.092548430592079*(rx[0]*ry[0]*rz[7]);
    c[8] += -0.4516580379125866*(rx[7]*ry[0]*rz[0]);
    c[8] += -1.35497411373776*(rx[5]*ry[2]*rz[0]);
    c[8] += 10.839792909902078*(rx[5]*ry[0]*rz[2]);
    c[8] += -1.35497411373776*(rx[3]*ry[4]*rz[0]);
    c[8] += 21.679585819804156*(rx[3]*ry[2]*rz[2]);
    c[8] += -21.679585819804156*(rx[3]*ry[0]*rz[4]);
    c[8] += -0.4516580379125866*(rx[1]*ry[6]*rz[0]);
    c[8] += 10.839792909902078*(rx[1]*ry[4]*rz[2]);
    c[8] += -21.679585819804156*(rx[1]*ry[2]*rz[4]);
    c[8] += 5.781222885281109*(rx[1]*ry[0]*rz[6]);
    c[9] += 3.3189951933373707*(rx[6]*ry[0]*rz[1]);
    c[9] += 3.3189951933373707*(rx[4]*ry[2]*rz[1]);
    c[9] += -17.701307697799308*(rx[4]*ry[0]*rz[3]);
    c[9] += -3.3189951933373707*(rx[2]*ry[4]*rz[1]);
    c[9] += 10.620784618679586*(rx[2]*ry[0]*rz[5]);
    c[9] += -3.3189951933373707*(rx[0]*ry[6]*rz[1]);
    c[9] += 17.701307697799308*(rx[0]*ry[4]*rz[3]);
    c[9] += -10.620784618679586*(rx[0]*ry[2]*rz[5]);
    c[10] += 0.4693768015868821*(rx[7]*ry[0]*rz[0]);
    c[10] += -0.4693768015868821*(rx[5]*ry[2]*rz[0]);
    c[10] += -9.38753603173764*(rx[5]*ry[0]*rz[2]);
    c[10] += -2.3468840079344107*(rx[3]*ry[4]*rz[0]);
    c[10] += 18.77507206347528*(rx[3]*ry[2]*rz[2]);
    c[10] += 12.516714708983523*(rx[3]*ry[0]*rz[4]);
    c[10] += -1.4081304047606462*(rx[1]*ry[6]*rz[0]);
    c[10] += 28.162608095212924*(rx[1]*ry[4]*rz[2]);
    c[10] += -37.55014412695057*(rx[1]*ry[2]*rz[4]);
    c[11] += -3.1134934723215624*(rx[6]*ry[0]*rz[1]);
    c[11] += 15.567467361607811*(rx[4]*ry[2]*rz[1]);
    c[11] += 10.378311574405208*(rx[4]*ry[0]*rz[3]);
    c[11] += 15.567467361607811*(rx[2]*ry[4]*rz[1]);
    c[11] += -62.269869446431244*(rx[2]*ry[2]*rz[3]);
    c[11] += -3.1134934723215624*(rx[0]*ry[6]*rz[1]);
    c[11] += 10.378311574405208*(rx[0]*ry[4]*rz[3]);
    c[12] += -0.5189155787202604*(rx[7]*ry[0]*rz[0]);
    c[12] += 4.670240208482344*(rx[5]*ry[2]*rz[0]);
    c[12] += 6.226986944643125*(rx[5]*ry[0]*rz[2]);
    c[12] += 2.594577893601302*(rx[3]*ry[4]*rz[0]);
    c[12] += -62.269869446431244*(rx[3]*ry[2]*rz[2]);
    c[12] += -2.594577893601302*(rx[1]*ry[6]*rz[0]);
    c[12] += 31.134934723215622*(rx[1]*ry[4]*rz[2]);
    c[13] += 2.6459606618019*(rx[6]*ry[0]*rz[1]);
    c[13] += -39.6894099270285*(rx[4]*ry[2]*rz[1]);
    c[13] += 39.6894099270285*(rx[2]*ry[4]*rz[1]);
    c[13] += -2.6459606618019*(rx[0]*ry[6]*rz[1]);
    c[14] += 0.7071627325245963*(rx[7]*ry[0]*rz[0]);
    c[14] += -14.850417383016522*(rx[5]*ry[2]*rz[0]);
    c[14] += 24.75069563836087*(rx[3]*ry[4]*rz[0]);
    c[14] += -4.950139127672174*(rx[1]*ry[6]*rz[0]);;

    double buf[(ECP_LMAX+1)*(ECP_LMAX+2)/2];
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] = 0.0;
    double nuc;

    // l = 7, i = 0
    nuc = 0.0;
    nuc += c[8]*-0.4516580379125866;
    nuc += c[10]*0.4693768015868821;
    nuc += c[12]*-0.5189155787202604;
    nuc += c[14]*0.7071627325245963;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+7, j+pv+0, k+pw+0);
    }
    // l = 7, i = 1
    nuc = 0.0;
    nuc += c[0]*4.950139127672174;
    nuc += c[2]*-2.594577893601302;
    nuc += c[4]*1.4081304047606462;
    nuc += c[6]*-0.4516580379125866;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+1, k+pw+0);
    }
    // l = 7, i = 2
    nuc = 0.0;
    nuc += c[7]*-2.389949691920173;
    nuc += c[9]*3.3189951933373707;
    nuc += c[11]*-3.1134934723215624;
    nuc += c[13]*2.6459606618019;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+0, k+pw+1);
    }
    // l = 7, i = 3
    nuc = 0.0;
    nuc += c[8]*-1.35497411373776;
    nuc += c[10]*-0.4693768015868821;
    nuc += c[12]*4.670240208482344;
    nuc += c[14]*-14.850417383016522;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+2, k+pw+0);
    }
    // l = 7, i = 4
    nuc = 0.0;
    nuc += c[1]*15.8757639708114;
    nuc += c[3]*-12.45397388928625;
    nuc += c[5]*6.637990386674741;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+1, k+pw+1);
    }
    // l = 7, i = 5
    nuc = 0.0;
    nuc += c[8]*10.839792909902078;
    nuc += c[10]*-9.38753603173764;
    nuc += c[12]*6.226986944643125;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+0, k+pw+2);
    }
    // l = 7, i = 6
    nuc = 0.0;
    nuc += c[0]*-24.75069563836087;
    nuc += c[2]*2.594577893601302;
    nuc += c[4]*2.3468840079344107;
    nuc += c[6]*-1.35497411373776;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+3, k+pw+0);
    }
    // l = 7, i = 7
    nuc = 0.0;
    nuc += c[7]*-7.169849075760519;
    nuc += c[9]*3.3189951933373707;
    nuc += c[11]*15.567467361607811;
    nuc += c[13]*-39.6894099270285;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+2, k+pw+1);
    }
    // l = 7, i = 8
    nuc = 0.0;
    nuc += c[2]*31.134934723215622;
    nuc += c[4]*-28.162608095212924;
    nuc += c[6]*10.839792909902078;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+1, k+pw+2);
    }
    // l = 7, i = 9
    nuc = 0.0;
    nuc += c[7]*14.339698151521036;
    nuc += c[9]*-17.701307697799308;
    nuc += c[11]*10.378311574405208;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+0, k+pw+3);
    }
    // l = 7, i = 10
    nuc = 0.0;
    nuc += c[8]*-1.35497411373776;
    nuc += c[10]*-2.3468840079344107;
    nuc += c[12]*2.594577893601302;
    nuc += c[14]*24.75069563836087;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+4, k+pw+0);
    }
    // l = 7, i = 11
    nuc = 0.0;
    nuc += c[1]*-52.919213236038004;
    nuc += c[5]*13.275980773349483;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+3, k+pw+1);
    }
    // l = 7, i = 12
    nuc = 0.0;
    nuc += c[8]*21.679585819804156;
    nuc += c[10]*18.77507206347528;
    nuc += c[12]*-62.269869446431244;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+2, k+pw+2);
    }
    // l = 7, i = 13
    nuc = 0.0;
    nuc += c[3]*41.51324629762083;
    nuc += c[5]*-35.402615395598616;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+1, k+pw+3);
    }
    // l = 7, i = 14
    nuc = 0.0;
    nuc += c[8]*-21.679585819804156;
    nuc += c[10]*12.516714708983523;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+0, k+pw+4);
    }
    // l = 7, i = 15
    nuc = 0.0;
    nuc += c[0]*14.850417383016522;
    nuc += c[2]*4.670240208482344;
    nuc += c[4]*0.4693768015868821;
    nuc += c[6]*-1.35497411373776;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+5, k+pw+0);
    }
    // l = 7, i = 16
    nuc = 0.0;
    nuc += c[7]*-7.169849075760519;
    nuc += c[9]*-3.3189951933373707;
    nuc += c[11]*15.567467361607811;
    nuc += c[13]*39.6894099270285;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+4, k+pw+1);
    }
    // l = 7, i = 17
    nuc = 0.0;
    nuc += c[2]*-62.269869446431244;
    nuc += c[4]*-18.77507206347528;
    nuc += c[6]*21.679585819804156;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+3, k+pw+2);
    }
    // l = 7, i = 18
    nuc = 0.0;
    nuc += c[7]*28.679396303042072;
    nuc += c[11]*-62.269869446431244;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+2, k+pw+3);
    }
    // l = 7, i = 19
    nuc = 0.0;
    nuc += c[4]*37.55014412695057;
    nuc += c[6]*-21.679585819804156;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+1, k+pw+4);
    }
    // l = 7, i = 20
    nuc = 0.0;
    nuc += c[7]*-11.47175852121683;
    nuc += c[9]*10.620784618679586;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+0, k+pw+5);
    }
    // l = 7, i = 21
    nuc = 0.0;
    nuc += c[8]*-0.4516580379125866;
    nuc += c[10]*-1.4081304047606462;
    nuc += c[12]*-2.594577893601302;
    nuc += c[14]*-4.950139127672174;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+6, k+pw+0);
    }
    // l = 7, i = 22
    nuc = 0.0;
    nuc += c[1]*15.8757639708114;
    nuc += c[3]*12.45397388928625;
    nuc += c[5]*6.637990386674741;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+5, k+pw+1);
    }
    // l = 7, i = 23
    nuc = 0.0;
    nuc += c[8]*10.839792909902078;
    nuc += c[10]*28.162608095212924;
    nuc += c[12]*31.134934723215622;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+4, k+pw+2);
    }
    // l = 7, i = 24
    nuc = 0.0;
    nuc += c[3]*-41.51324629762083;
    nuc += c[5]*-35.402615395598616;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+3, k+pw+3);
    }
    // l = 7, i = 25
    nuc = 0.0;
    nuc += c[8]*-21.679585819804156;
    nuc += c[10]*-37.55014412695057;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+2, k+pw+4);
    }
    // l = 7, i = 26
    nuc = 0.0;
    nuc += c[5]*21.241569237359172;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+1, k+pw+5);
    }
    // l = 7, i = 27
    nuc = 0.0;
    nuc += c[8]*5.781222885281109;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+0, k+pw+6);
    }
    // l = 7, i = 28
    nuc = 0.0;
    nuc += c[0]*-0.7071627325245963;
    nuc += c[2]*-0.5189155787202604;
    nuc += c[4]*-0.4693768015868821;
    nuc += c[6]*-0.4516580379125866;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+7, k+pw+0);
    }
    // l = 7, i = 29
    nuc = 0.0;
    nuc += c[7]*-2.389949691920173;
    nuc += c[9]*-3.3189951933373707;
    nuc += c[11]*-3.1134934723215624;
    nuc += c[13]*-2.6459606618019;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+6, k+pw+1);
    }
    // l = 7, i = 30
    nuc = 0.0;
    nuc += c[2]*6.226986944643125;
    nuc += c[4]*9.38753603173764;
    nuc += c[6]*10.839792909902078;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+5, k+pw+2);
    }
    // l = 7, i = 31
    nuc = 0.0;
    nuc += c[7]*14.339698151521036;
    nuc += c[9]*17.701307697799308;
    nuc += c[11]*10.378311574405208;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+4, k+pw+3);
    }
    // l = 7, i = 32
    nuc = 0.0;
    nuc += c[4]*-12.516714708983523;
    nuc += c[6]*-21.679585819804156;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+3, k+pw+4);
    }
    // l = 7, i = 33
    nuc = 0.0;
    nuc += c[7]*-11.47175852121683;
    nuc += c[9]*-10.620784618679586;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+2, k+pw+5);
    }
    // l = 7, i = 34
    nuc = 0.0;
    nuc += c[6]*5.781222885281109;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+1, k+pw+6);
    }
    // l = 7, i = 35
    nuc = 0.0;
    nuc += c[7]*1.092548430592079;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+0, k+pw+7);
    }

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] *= 4.0 * M_PI;
    cart2sph(omega, lc, buf);
}
template <> __device__
void type2_ang_nuc_l<8>(double * __restrict__ omega, const int lc,
                    const int i, const int j, const int k,
                    double * __restrict__ unitr){
    constexpr int l = 8;
    double rx[l+1], ry[l+1], rz[l+1];
    rx[0] = ry[0] = rz[0] = 1.0;
    for (int li = 1; li <= l; li++) {
        rx[li] = rx[li - 1] * unitr[0];
        ry[li] = ry[li - 1] * unitr[1];
        rz[li] = rz[li - 1] * unitr[2];
    }

    double c[2*l+1];
    for (int m = 0; m < 2*l+1; m++) c[m] = 0.0;
    c[0] += 5.83141328139864*(rx[7]*ry[1]*rz[0]);
    c[0] += -40.81989296979048*(rx[5]*ry[3]*rz[0]);
    c[0] += 40.81989296979048*(rx[3]*ry[5]*rz[0]);
    c[0] += -5.83141328139864*(rx[1]*ry[7]*rz[0]);
    c[1] += 20.40994648489524*(rx[6]*ry[1]*rz[1]);
    c[1] += -102.0497324244762*(rx[4]*ry[3]*rz[1]);
    c[1] += 61.22983945468572*(rx[2]*ry[5]*rz[1]);
    c[1] += -2.91570664069932*(rx[0]*ry[7]*rz[1]);
    c[2] += -3.193996596357255*(rx[7]*ry[1]*rz[0]);
    c[2] += 7.452658724833595*(rx[5]*ry[3]*rz[0]);
    c[2] += 44.71595234900157*(rx[5]*ry[1]*rz[2]);
    c[2] += 7.452658724833595*(rx[3]*ry[5]*rz[0]);
    c[2] += -149.0531744966719*(rx[3]*ry[3]*rz[2]);
    c[2] += -3.193996596357255*(rx[1]*ry[7]*rz[0]);
    c[2] += 44.71595234900157*(rx[1]*ry[5]*rz[2]);
    c[3] += -17.24955311049054*(rx[6]*ry[1]*rz[1]);
    c[3] += 17.24955311049054*(rx[4]*ry[3]*rz[1]);
    c[3] += 68.99821244196217*(rx[4]*ry[1]*rz[3]);
    c[3] += 31.04919559888297*(rx[2]*ry[5]*rz[1]);
    c[3] += -137.9964248839243*(rx[2]*ry[3]*rz[3]);
    c[3] += -3.449910622098108*(rx[0]*ry[7]*rz[1]);
    c[3] += 13.79964248839243*(rx[0]*ry[5]*rz[3]);
    c[4] += 1.913666099037323*(rx[7]*ry[1]*rz[0]);
    c[4] += 1.913666099037323*(rx[5]*ry[3]*rz[0]);
    c[4] += -45.92798637689575*(rx[5]*ry[1]*rz[2]);
    c[4] += -1.913666099037323*(rx[3]*ry[5]*rz[0]);
    c[4] += 76.54664396149292*(rx[3]*ry[1]*rz[4]);
    c[4] += -1.913666099037323*(rx[1]*ry[7]*rz[0]);
    c[4] += 45.92798637689575*(rx[1]*ry[5]*rz[2]);
    c[4] += -76.54664396149292*(rx[1]*ry[3]*rz[4]);
    c[5] += 11.1173953976599*(rx[6]*ry[1]*rz[1]);
    c[5] += 18.52899232943316*(rx[4]*ry[3]*rz[1]);
    c[5] += -74.11596931773265*(rx[4]*ry[1]*rz[3]);
    c[5] += 3.705798465886632*(rx[2]*ry[5]*rz[1]);
    c[5] += -49.41064621182176*(rx[2]*ry[3]*rz[3]);
    c[5] += 59.29277545418611*(rx[2]*ry[1]*rz[5]);
    c[5] += -3.705798465886632*(rx[0]*ry[7]*rz[1]);
    c[5] += 24.70532310591088*(rx[0]*ry[5]*rz[3]);
    c[5] += -19.7642584847287*(rx[0]*ry[3]*rz[5]);
    c[6] += -0.912304516869819*(rx[7]*ry[1]*rz[0]);
    c[6] += -2.736913550609457*(rx[5]*ry[3]*rz[0]);
    c[6] += 27.36913550609457*(rx[5]*ry[1]*rz[2]);
    c[6] += -2.736913550609457*(rx[3]*ry[5]*rz[0]);
    c[6] += 54.73827101218914*(rx[3]*ry[3]*rz[2]);
    c[6] += -72.98436134958553*(rx[3]*ry[1]*rz[4]);
    c[6] += -0.912304516869819*(rx[1]*ry[7]*rz[0]);
    c[6] += 27.36913550609457*(rx[1]*ry[5]*rz[2]);
    c[6] += -72.98436134958553*(rx[1]*ry[3]*rz[4]);
    c[6] += 29.19374453983421*(rx[1]*ry[1]*rz[6]);
    c[7] += -3.8164436064573*(rx[6]*ry[1]*rz[1]);
    c[7] += -11.4493308193719*(rx[4]*ry[3]*rz[1]);
    c[7] += 30.5315488516584*(rx[4]*ry[1]*rz[3]);
    c[7] += -11.4493308193719*(rx[2]*ry[5]*rz[1]);
    c[7] += 61.06309770331679*(rx[2]*ry[3]*rz[3]);
    c[7] += -36.63785862199007*(rx[2]*ry[1]*rz[5]);
    c[7] += -3.8164436064573*(rx[0]*ry[7]*rz[1]);
    c[7] += 30.5315488516584*(rx[0]*ry[5]*rz[3]);
    c[7] += -36.63785862199007*(rx[0]*ry[3]*rz[5]);
    c[7] += 6.978639737521918*(rx[0]*ry[1]*rz[7]);
    c[8] += 0.3180369672047749*(rx[8]*ry[0]*rz[0]);
    c[8] += 1.272147868819099*(rx[6]*ry[2]*rz[0]);
    c[8] += -10.1771829505528*(rx[6]*ry[0]*rz[2]);
    c[8] += 1.908221803228649*(rx[4]*ry[4]*rz[0]);
    c[8] += -30.53154885165839*(rx[4]*ry[2]*rz[2]);
    c[8] += 30.53154885165839*(rx[4]*ry[0]*rz[4]);
    c[8] += 1.272147868819099*(rx[2]*ry[6]*rz[0]);
    c[8] += -30.53154885165839*(rx[2]*ry[4]*rz[2]);
    c[8] += 61.06309770331677*(rx[2]*ry[2]*rz[4]);
    c[8] += -16.28349272088447*(rx[2]*ry[0]*rz[6]);
    c[8] += 0.3180369672047749*(rx[0]*ry[8]*rz[0]);
    c[8] += -10.1771829505528*(rx[0]*ry[6]*rz[2]);
    c[8] += 30.53154885165839*(rx[0]*ry[4]*rz[4]);
    c[8] += -16.28349272088447*(rx[0]*ry[2]*rz[6]);
    c[8] += 1.16310662292032*(rx[0]*ry[0]*rz[8]);
    c[9] += -3.8164436064573*(rx[7]*ry[0]*rz[1]);
    c[9] += -11.4493308193719*(rx[5]*ry[2]*rz[1]);
    c[9] += 30.5315488516584*(rx[5]*ry[0]*rz[3]);
    c[9] += -11.4493308193719*(rx[3]*ry[4]*rz[1]);
    c[9] += 61.06309770331679*(rx[3]*ry[2]*rz[3]);
    c[9] += -36.63785862199007*(rx[3]*ry[0]*rz[5]);
    c[9] += -3.8164436064573*(rx[1]*ry[6]*rz[1]);
    c[9] += 30.5315488516584*(rx[1]*ry[4]*rz[3]);
    c[9] += -36.63785862199007*(rx[1]*ry[2]*rz[5]);
    c[9] += 6.978639737521918*(rx[1]*ry[0]*rz[7]);
    c[10] += -0.4561522584349095*(rx[8]*ry[0]*rz[0]);
    c[10] += -0.912304516869819*(rx[6]*ry[2]*rz[0]);
    c[10] += 13.68456775304729*(rx[6]*ry[0]*rz[2]);
    c[10] += 13.68456775304729*(rx[4]*ry[2]*rz[2]);
    c[10] += -36.49218067479276*(rx[4]*ry[0]*rz[4]);
    c[10] += 0.912304516869819*(rx[2]*ry[6]*rz[0]);
    c[10] += -13.68456775304729*(rx[2]*ry[4]*rz[2]);
    c[10] += 14.5968722699171*(rx[2]*ry[0]*rz[6]);
    c[10] += 0.4561522584349095*(rx[0]*ry[8]*rz[0]);
    c[10] += -13.68456775304729*(rx[0]*ry[6]*rz[2]);
    c[10] += 36.49218067479276*(rx[0]*ry[4]*rz[4]);
    c[10] += -14.5968722699171*(rx[0]*ry[2]*rz[6]);
    c[11] += 3.705798465886632*(rx[7]*ry[0]*rz[1]);
    c[11] += -3.705798465886632*(rx[5]*ry[2]*rz[1]);
    c[11] += -24.70532310591088*(rx[5]*ry[0]*rz[3]);
    c[11] += -18.52899232943316*(rx[3]*ry[4]*rz[1]);
    c[11] += 49.41064621182176*(rx[3]*ry[2]*rz[3]);
    c[11] += 19.7642584847287*(rx[3]*ry[0]*rz[5]);
    c[11] += -11.1173953976599*(rx[1]*ry[6]*rz[1]);
    c[11] += 74.11596931773265*(rx[1]*ry[4]*rz[3]);
    c[11] += -59.29277545418611*(rx[1]*ry[2]*rz[5]);
    c[12] += 0.4784165247593308*(rx[8]*ry[0]*rz[0]);
    c[12] += -1.913666099037323*(rx[6]*ry[2]*rz[0]);
    c[12] += -11.48199659422394*(rx[6]*ry[0]*rz[2]);
    c[12] += -4.784165247593307*(rx[4]*ry[4]*rz[0]);
    c[12] += 57.40998297111968*(rx[4]*ry[2]*rz[2]);
    c[12] += 19.13666099037323*(rx[4]*ry[0]*rz[4]);
    c[12] += -1.913666099037323*(rx[2]*ry[6]*rz[0]);
    c[12] += 57.40998297111968*(rx[2]*ry[4]*rz[2]);
    c[12] += -114.8199659422394*(rx[2]*ry[2]*rz[4]);
    c[12] += 0.4784165247593308*(rx[0]*ry[8]*rz[0]);
    c[12] += -11.48199659422394*(rx[0]*ry[6]*rz[2]);
    c[12] += 19.13666099037323*(rx[0]*ry[4]*rz[4]);
    c[13] += -3.449910622098108*(rx[7]*ry[0]*rz[1]);
    c[13] += 31.04919559888297*(rx[5]*ry[2]*rz[1]);
    c[13] += 13.79964248839243*(rx[5]*ry[0]*rz[3]);
    c[13] += 17.24955311049054*(rx[3]*ry[4]*rz[1]);
    c[13] += -137.9964248839243*(rx[3]*ry[2]*rz[3]);
    c[13] += -17.24955311049054*(rx[1]*ry[6]*rz[1]);
    c[13] += 68.99821244196217*(rx[1]*ry[4]*rz[3]);
    c[14] += -0.5323327660595425*(rx[8]*ry[0]*rz[0]);
    c[14] += 7.452658724833595*(rx[6]*ry[2]*rz[0]);
    c[14] += 7.452658724833595*(rx[6]*ry[0]*rz[2]);
    c[14] += -111.7898808725039*(rx[4]*ry[2]*rz[2]);
    c[14] += -7.452658724833595*(rx[2]*ry[6]*rz[0]);
    c[14] += 111.7898808725039*(rx[2]*ry[4]*rz[2]);
    c[14] += 0.5323327660595425*(rx[0]*ry[8]*rz[0]);
    c[14] += -7.452658724833595*(rx[0]*ry[6]*rz[2]);
    c[15] += 2.91570664069932*(rx[7]*ry[0]*rz[1]);
    c[15] += -61.22983945468572*(rx[5]*ry[2]*rz[1]);
    c[15] += 102.0497324244762*(rx[3]*ry[4]*rz[1]);
    c[15] += -20.40994648489524*(rx[1]*ry[6]*rz[1]);
    c[16] += 0.72892666017483*(rx[8]*ry[0]*rz[0]);
    c[16] += -20.40994648489524*(rx[6]*ry[2]*rz[0]);
    c[16] += 51.0248662122381*(rx[4]*ry[4]*rz[0]);
    c[16] += -20.40994648489524*(rx[2]*ry[6]*rz[0]);
    c[16] += 0.72892666017483*(rx[0]*ry[8]*rz[0]);;

    double buf[(ECP_LMAX+1)*(ECP_LMAX+2)/2];
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] = 0.0;
    double nuc;

    // l = 8, i = 0
    nuc = 0.0;
    nuc += c[8]*0.3180369672047749;
    nuc += c[10]*-0.4561522584349095;
    nuc += c[12]*0.4784165247593308;
    nuc += c[14]*-0.5323327660595425;
    nuc += c[16]*0.72892666017483;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+8, j+pv+0, k+pw+0);
    }
    // l = 8, i = 1
    nuc = 0.0;
    nuc += c[0]*5.83141328139864;
    nuc += c[2]*-3.193996596357255;
    nuc += c[4]*1.913666099037323;
    nuc += c[6]*-0.912304516869819;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+7, j+pv+1, k+pw+0);
    }
    // l = 8, i = 2
    nuc = 0.0;
    nuc += c[9]*-3.8164436064573;
    nuc += c[11]*3.705798465886632;
    nuc += c[13]*-3.449910622098108;
    nuc += c[15]*2.91570664069932;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+7, j+pv+0, k+pw+1);
    }
    // l = 8, i = 3
    nuc = 0.0;
    nuc += c[8]*1.272147868819099;
    nuc += c[10]*-0.912304516869819;
    nuc += c[12]*-1.913666099037323;
    nuc += c[14]*7.452658724833595;
    nuc += c[16]*-20.40994648489524;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+2, k+pw+0);
    }
    // l = 8, i = 4
    nuc = 0.0;
    nuc += c[1]*20.40994648489524;
    nuc += c[3]*-17.24955311049054;
    nuc += c[5]*11.1173953976599;
    nuc += c[7]*-3.8164436064573;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+1, k+pw+1);
    }
    // l = 8, i = 5
    nuc = 0.0;
    nuc += c[8]*-10.1771829505528;
    nuc += c[10]*13.68456775304729;
    nuc += c[12]*-11.48199659422394;
    nuc += c[14]*7.452658724833595;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+0, k+pw+2);
    }
    // l = 8, i = 6
    nuc = 0.0;
    nuc += c[0]*-40.81989296979048;
    nuc += c[2]*7.452658724833595;
    nuc += c[4]*1.913666099037323;
    nuc += c[6]*-2.736913550609457;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+3, k+pw+0);
    }
    // l = 8, i = 7
    nuc = 0.0;
    nuc += c[9]*-11.4493308193719;
    nuc += c[11]*-3.705798465886632;
    nuc += c[13]*31.04919559888297;
    nuc += c[15]*-61.22983945468572;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+2, k+pw+1);
    }
    // l = 8, i = 8
    nuc = 0.0;
    nuc += c[2]*44.71595234900157;
    nuc += c[4]*-45.92798637689575;
    nuc += c[6]*27.36913550609457;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+1, k+pw+2);
    }
    // l = 8, i = 9
    nuc = 0.0;
    nuc += c[9]*30.5315488516584;
    nuc += c[11]*-24.70532310591088;
    nuc += c[13]*13.79964248839243;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+0, k+pw+3);
    }
    // l = 8, i = 10
    nuc = 0.0;
    nuc += c[8]*1.908221803228649;
    nuc += c[12]*-4.784165247593307;
    nuc += c[16]*51.0248662122381;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+4, k+pw+0);
    }
    // l = 8, i = 11
    nuc = 0.0;
    nuc += c[1]*-102.0497324244762;
    nuc += c[3]*17.24955311049054;
    nuc += c[5]*18.52899232943316;
    nuc += c[7]*-11.4493308193719;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+3, k+pw+1);
    }
    // l = 8, i = 12
    nuc = 0.0;
    nuc += c[8]*-30.53154885165839;
    nuc += c[10]*13.68456775304729;
    nuc += c[12]*57.40998297111968;
    nuc += c[14]*-111.7898808725039;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+2, k+pw+2);
    }
    // l = 8, i = 13
    nuc = 0.0;
    nuc += c[3]*68.99821244196217;
    nuc += c[5]*-74.11596931773265;
    nuc += c[7]*30.5315488516584;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+1, k+pw+3);
    }
    // l = 8, i = 14
    nuc = 0.0;
    nuc += c[8]*30.53154885165839;
    nuc += c[10]*-36.49218067479276;
    nuc += c[12]*19.13666099037323;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+0, k+pw+4);
    }
    // l = 8, i = 15
    nuc = 0.0;
    nuc += c[0]*40.81989296979048;
    nuc += c[2]*7.452658724833595;
    nuc += c[4]*-1.913666099037323;
    nuc += c[6]*-2.736913550609457;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+5, k+pw+0);
    }
    // l = 8, i = 16
    nuc = 0.0;
    nuc += c[9]*-11.4493308193719;
    nuc += c[11]*-18.52899232943316;
    nuc += c[13]*17.24955311049054;
    nuc += c[15]*102.0497324244762;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+4, k+pw+1);
    }
    // l = 8, i = 17
    nuc = 0.0;
    nuc += c[2]*-149.0531744966719;
    nuc += c[6]*54.73827101218914;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+3, k+pw+2);
    }
    // l = 8, i = 18
    nuc = 0.0;
    nuc += c[9]*61.06309770331679;
    nuc += c[11]*49.41064621182176;
    nuc += c[13]*-137.9964248839243;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+2, k+pw+3);
    }
    // l = 8, i = 19
    nuc = 0.0;
    nuc += c[4]*76.54664396149292;
    nuc += c[6]*-72.98436134958553;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+1, k+pw+4);
    }
    // l = 8, i = 20
    nuc = 0.0;
    nuc += c[9]*-36.63785862199007;
    nuc += c[11]*19.7642584847287;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+0, k+pw+5);
    }
    // l = 8, i = 21
    nuc = 0.0;
    nuc += c[8]*1.272147868819099;
    nuc += c[10]*0.912304516869819;
    nuc += c[12]*-1.913666099037323;
    nuc += c[14]*-7.452658724833595;
    nuc += c[16]*-20.40994648489524;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+6, k+pw+0);
    }
    // l = 8, i = 22
    nuc = 0.0;
    nuc += c[1]*61.22983945468572;
    nuc += c[3]*31.04919559888297;
    nuc += c[5]*3.705798465886632;
    nuc += c[7]*-11.4493308193719;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+5, k+pw+1);
    }
    // l = 8, i = 23
    nuc = 0.0;
    nuc += c[8]*-30.53154885165839;
    nuc += c[10]*-13.68456775304729;
    nuc += c[12]*57.40998297111968;
    nuc += c[14]*111.7898808725039;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+4, k+pw+2);
    }
    // l = 8, i = 24
    nuc = 0.0;
    nuc += c[3]*-137.9964248839243;
    nuc += c[5]*-49.41064621182176;
    nuc += c[7]*61.06309770331679;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+3, k+pw+3);
    }
    // l = 8, i = 25
    nuc = 0.0;
    nuc += c[8]*61.06309770331677;
    nuc += c[12]*-114.8199659422394;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+2, k+pw+4);
    }
    // l = 8, i = 26
    nuc = 0.0;
    nuc += c[5]*59.29277545418611;
    nuc += c[7]*-36.63785862199007;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+1, k+pw+5);
    }
    // l = 8, i = 27
    nuc = 0.0;
    nuc += c[8]*-16.28349272088447;
    nuc += c[10]*14.5968722699171;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+0, k+pw+6);
    }
    // l = 8, i = 28
    nuc = 0.0;
    nuc += c[0]*-5.83141328139864;
    nuc += c[2]*-3.193996596357255;
    nuc += c[4]*-1.913666099037323;
    nuc += c[6]*-0.912304516869819;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+7, k+pw+0);
    }
    // l = 8, i = 29
    nuc = 0.0;
    nuc += c[9]*-3.8164436064573;
    nuc += c[11]*-11.1173953976599;
    nuc += c[13]*-17.24955311049054;
    nuc += c[15]*-20.40994648489524;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+6, k+pw+1);
    }
    // l = 8, i = 30
    nuc = 0.0;
    nuc += c[2]*44.71595234900157;
    nuc += c[4]*45.92798637689575;
    nuc += c[6]*27.36913550609457;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+5, k+pw+2);
    }
    // l = 8, i = 31
    nuc = 0.0;
    nuc += c[9]*30.5315488516584;
    nuc += c[11]*74.11596931773265;
    nuc += c[13]*68.99821244196217;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+4, k+pw+3);
    }
    // l = 8, i = 32
    nuc = 0.0;
    nuc += c[4]*-76.54664396149292;
    nuc += c[6]*-72.98436134958553;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+3, k+pw+4);
    }
    // l = 8, i = 33
    nuc = 0.0;
    nuc += c[9]*-36.63785862199007;
    nuc += c[11]*-59.29277545418611;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+2, k+pw+5);
    }
    // l = 8, i = 34
    nuc = 0.0;
    nuc += c[6]*29.19374453983421;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+1, k+pw+6);
    }
    // l = 8, i = 35
    nuc = 0.0;
    nuc += c[9]*6.978639737521918;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+0, k+pw+7);
    }
    // l = 8, i = 36
    nuc = 0.0;
    nuc += c[8]*0.3180369672047749;
    nuc += c[10]*0.4561522584349095;
    nuc += c[12]*0.4784165247593308;
    nuc += c[14]*0.5323327660595425;
    nuc += c[16]*0.72892666017483;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+8, k+pw+0);
    }
    // l = 8, i = 37
    nuc = 0.0;
    nuc += c[1]*-2.91570664069932;
    nuc += c[3]*-3.449910622098108;
    nuc += c[5]*-3.705798465886632;
    nuc += c[7]*-3.8164436064573;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+7, k+pw+1);
    }
    // l = 8, i = 38
    nuc = 0.0;
    nuc += c[8]*-10.1771829505528;
    nuc += c[10]*-13.68456775304729;
    nuc += c[12]*-11.48199659422394;
    nuc += c[14]*-7.452658724833595;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+6, k+pw+2);
    }
    // l = 8, i = 39
    nuc = 0.0;
    nuc += c[3]*13.79964248839243;
    nuc += c[5]*24.70532310591088;
    nuc += c[7]*30.5315488516584;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+5, k+pw+3);
    }
    // l = 8, i = 40
    nuc = 0.0;
    nuc += c[8]*30.53154885165839;
    nuc += c[10]*36.49218067479276;
    nuc += c[12]*19.13666099037323;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+4, k+pw+4);
    }
    // l = 8, i = 41
    nuc = 0.0;
    nuc += c[5]*-19.7642584847287;
    nuc += c[7]*-36.63785862199007;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+3, k+pw+5);
    }
    // l = 8, i = 42
    nuc = 0.0;
    nuc += c[8]*-16.28349272088447;
    nuc += c[10]*-14.5968722699171;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+2, k+pw+6);
    }
    // l = 8, i = 43
    nuc = 0.0;
    nuc += c[7]*6.978639737521918;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+1, k+pw+7);
    }
    // l = 8, i = 44
    nuc = 0.0;
    nuc += c[8]*1.16310662292032;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+0, k+pw+8);
    }

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] *= 4.0 * M_PI;
    cart2sph(omega, lc, buf);
}
template <> __device__
void type2_ang_nuc_l<9>(double * __restrict__ omega, const int lc,
                    const int i, const int j, const int k,
                    double * __restrict__ unitr){
    constexpr int l = 9;
    double rx[l+1], ry[l+1], rz[l+1];
    rx[0] = ry[0] = rz[0] = 1.0;
    for (int li = 1; li <= l; li++) {
        rx[li] = rx[li - 1] * unitr[0];
        ry[li] = ry[li - 1] * unitr[1];
        rz[li] = rz[li - 1] * unitr[2];
    }

    double c[2*l+1];
    for (int m = 0; m < 2*l+1; m++) c[m] = 0.0;
    c[0] += 6.740108566678694*(rx[8]*ry[1]*rz[0]);
    c[0] += -62.9076799556678*(rx[6]*ry[3]*rz[0]);
    c[0] += 94.36151993350171*(rx[4]*ry[5]*rz[0]);
    c[0] += -26.96043426671477*(rx[2]*ry[7]*rz[0]);
    c[0] += 0.7489009518531882*(rx[0]*ry[9]*rz[0]);
    c[1] += 25.41854119163758*(rx[7]*ry[1]*rz[1]);
    c[1] += -177.9297883414631*(rx[5]*ry[3]*rz[1]);
    c[1] += 177.9297883414631*(rx[3]*ry[5]*rz[1]);
    c[1] += -25.41854119163758*(rx[1]*ry[7]*rz[1]);
    c[2] += -3.814338369408373*(rx[8]*ry[1]*rz[0]);
    c[2] += 15.25735347763349*(rx[6]*ry[3]*rz[0]);
    c[2] += 61.02941391053396*(rx[6]*ry[1]*rz[2]);
    c[2] += 7.628676738816745*(rx[4]*ry[5]*rz[0]);
    c[2] += -305.1470695526698*(rx[4]*ry[3]*rz[2]);
    c[2] += -10.89810962688107*(rx[2]*ry[7]*rz[0]);
    c[2] += 183.0882417316019*(rx[2]*ry[5]*rz[2]);
    c[2] += 0.5449054813440533*(rx[0]*ry[9]*rz[0]);
    c[2] += -8.718487701504852*(rx[0]*ry[7]*rz[2]);
    c[3] += -22.65129549625621*(rx[7]*ry[1]*rz[1]);
    c[3] += 52.85302282459782*(rx[5]*ry[3]*rz[1]);
    c[3] += 105.7060456491956*(rx[5]*ry[1]*rz[3]);
    c[3] += 52.85302282459782*(rx[3]*ry[5]*rz[1]);
    c[3] += -352.3534854973187*(rx[3]*ry[3]*rz[3]);
    c[3] += -22.65129549625621*(rx[1]*ry[7]*rz[1]);
    c[3] += 105.7060456491956*(rx[1]*ry[5]*rz[3]);
    c[4] += 2.436891395195093*(rx[8]*ry[1]*rz[0]);
    c[4] += -68.23295906546261*(rx[6]*ry[1]*rz[2]);
    c[4] += -6.82329590654626*(rx[4]*ry[5]*rz[0]);
    c[4] += 68.23295906546261*(rx[4]*ry[3]*rz[2]);
    c[4] += 136.4659181309252*(rx[4]*ry[1]*rz[4]);
    c[4] += -3.899026232312149*(rx[2]*ry[7]*rz[0]);
    c[4] += 122.8193263178327*(rx[2]*ry[5]*rz[2]);
    c[4] += -272.9318362618504*(rx[2]*ry[3]*rz[4]);
    c[4] += 0.4873782790390186*(rx[0]*ry[9]*rz[0]);
    c[4] += -13.64659181309252*(rx[0]*ry[7]*rz[2]);
    c[4] += 27.29318362618504*(rx[0]*ry[5]*rz[4]);
    c[5] += 16.31079695491669*(rx[7]*ry[1]*rz[1]);
    c[5] += 16.31079695491669*(rx[5]*ry[3]*rz[1]);
    c[5] += -130.4863756393335*(rx[5]*ry[1]*rz[3]);
    c[5] += -16.31079695491669*(rx[3]*ry[5]*rz[1]);
    c[5] += 130.4863756393335*(rx[3]*ry[1]*rz[5]);
    c[5] += -16.31079695491669*(rx[1]*ry[7]*rz[1]);
    c[5] += 130.4863756393335*(rx[1]*ry[5]*rz[3]);
    c[5] += -130.4863756393335*(rx[1]*ry[3]*rz[5]);
    c[6] += -1.385125560048583*(rx[8]*ry[1]*rz[0]);
    c[6] += -3.693668160129556*(rx[6]*ry[3]*rz[0]);
    c[6] += 49.864520161749*(rx[6]*ry[1]*rz[2]);
    c[6] += -2.770251120097167*(rx[4]*ry[5]*rz[0]);
    c[6] += 83.107533602915*(rx[4]*ry[3]*rz[2]);
    c[6] += -166.21506720583*(rx[4]*ry[1]*rz[4]);
    c[6] += 16.621506720583*(rx[2]*ry[5]*rz[2]);
    c[6] += -110.8100448038867*(rx[2]*ry[3]*rz[4]);
    c[6] += 88.64803584310934*(rx[2]*ry[1]*rz[6]);
    c[6] += 0.4617085200161945*(rx[0]*ry[9]*rz[0]);
    c[6] += -16.621506720583*(rx[0]*ry[7]*rz[2]);
    c[6] += 55.40502240194333*(rx[0]*ry[5]*rz[4]);
    c[6] += -29.54934528103645*(rx[0]*ry[3]*rz[6]);
    c[7] += -8.46325696792098*(rx[7]*ry[1]*rz[1]);
    c[7] += -25.38977090376294*(rx[5]*ry[3]*rz[1]);
    c[7] += 84.6325696792098*(rx[5]*ry[1]*rz[3]);
    c[7] += -25.38977090376294*(rx[3]*ry[5]*rz[1]);
    c[7] += 169.2651393584196*(rx[3]*ry[3]*rz[3]);
    c[7] += -135.4121114867357*(rx[3]*ry[1]*rz[5]);
    c[7] += -8.46325696792098*(rx[1]*ry[7]*rz[1]);
    c[7] += 84.6325696792098*(rx[1]*ry[5]*rz[3]);
    c[7] += -135.4121114867357*(rx[1]*ry[3]*rz[5]);
    c[7] += 38.68917471049591*(rx[1]*ry[1]*rz[7]);
    c[8] += 0.451093112065591*(rx[8]*ry[1]*rz[0]);
    c[8] += 1.804372448262364*(rx[6]*ry[3]*rz[0]);
    c[8] += -18.04372448262364*(rx[6]*ry[1]*rz[2]);
    c[8] += 2.706558672393546*(rx[4]*ry[5]*rz[0]);
    c[8] += -54.13117344787092*(rx[4]*ry[3]*rz[2]);
    c[8] += 72.17489793049457*(rx[4]*ry[1]*rz[4]);
    c[8] += 1.804372448262364*(rx[2]*ry[7]*rz[0]);
    c[8] += -54.13117344787092*(rx[2]*ry[5]*rz[2]);
    c[8] += 144.3497958609891*(rx[2]*ry[3]*rz[4]);
    c[8] += -57.73991834439565*(rx[2]*ry[1]*rz[6]);
    c[8] += 0.451093112065591*(rx[0]*ry[9]*rz[0]);
    c[8] += -18.04372448262364*(rx[0]*ry[7]*rz[2]);
    c[8] += 72.17489793049457*(rx[0]*ry[5]*rz[4]);
    c[8] += -57.73991834439565*(rx[0]*ry[3]*rz[6]);
    c[8] += 8.248559763485094*(rx[0]*ry[1]*rz[8]);
    c[9] += 3.026024588281776*(rx[8]*ry[0]*rz[1]);
    c[9] += 12.1040983531271*(rx[6]*ry[2]*rz[1]);
    c[9] += -32.27759560833895*(rx[6]*ry[0]*rz[3]);
    c[9] += 18.15614752969066*(rx[4]*ry[4]*rz[1]);
    c[9] += -96.83278682501685*(rx[4]*ry[2]*rz[3]);
    c[9] += 58.0996720950101*(rx[4]*ry[0]*rz[5]);
    c[9] += 12.1040983531271*(rx[2]*ry[6]*rz[1]);
    c[9] += -96.83278682501685*(rx[2]*ry[4]*rz[3]);
    c[9] += 116.1993441900202*(rx[2]*ry[2]*rz[5]);
    c[9] += -22.1332084171467*(rx[2]*ry[0]*rz[7]);
    c[9] += 3.026024588281776*(rx[0]*ry[8]*rz[1]);
    c[9] += -32.27759560833895*(rx[0]*ry[6]*rz[3]);
    c[9] += 58.0996720950101*(rx[0]*ry[4]*rz[5]);
    c[9] += -22.1332084171467*(rx[0]*ry[2]*rz[7]);
    c[9] += 1.229622689841484*(rx[0]*ry[0]*rz[9]);
    c[10] += 0.451093112065591*(rx[9]*ry[0]*rz[0]);
    c[10] += 1.804372448262364*(rx[7]*ry[2]*rz[0]);
    c[10] += -18.04372448262364*(rx[7]*ry[0]*rz[2]);
    c[10] += 2.706558672393546*(rx[5]*ry[4]*rz[0]);
    c[10] += -54.13117344787092*(rx[5]*ry[2]*rz[2]);
    c[10] += 72.17489793049457*(rx[5]*ry[0]*rz[4]);
    c[10] += 1.804372448262364*(rx[3]*ry[6]*rz[0]);
    c[10] += -54.13117344787092*(rx[3]*ry[4]*rz[2]);
    c[10] += 144.3497958609891*(rx[3]*ry[2]*rz[4]);
    c[10] += -57.73991834439565*(rx[3]*ry[0]*rz[6]);
    c[10] += 0.451093112065591*(rx[1]*ry[8]*rz[0]);
    c[10] += -18.04372448262364*(rx[1]*ry[6]*rz[2]);
    c[10] += 72.17489793049457*(rx[1]*ry[4]*rz[4]);
    c[10] += -57.73991834439565*(rx[1]*ry[2]*rz[6]);
    c[10] += 8.248559763485094*(rx[1]*ry[0]*rz[8]);
    c[11] += -4.23162848396049*(rx[8]*ry[0]*rz[1]);
    c[11] += -8.46325696792098*(rx[6]*ry[2]*rz[1]);
    c[11] += 42.3162848396049*(rx[6]*ry[0]*rz[3]);
    c[11] += 42.3162848396049*(rx[4]*ry[2]*rz[3]);
    c[11] += -67.70605574336784*(rx[4]*ry[0]*rz[5]);
    c[11] += 8.46325696792098*(rx[2]*ry[6]*rz[1]);
    c[11] += -42.3162848396049*(rx[2]*ry[4]*rz[3]);
    c[11] += 19.34458735524795*(rx[2]*ry[0]*rz[7]);
    c[11] += 4.23162848396049*(rx[0]*ry[8]*rz[1]);
    c[11] += -42.3162848396049*(rx[0]*ry[6]*rz[3]);
    c[11] += 67.70605574336784*(rx[0]*ry[4]*rz[5]);
    c[11] += -19.34458735524795*(rx[0]*ry[2]*rz[7]);
    c[12] += -0.4617085200161945*(rx[9]*ry[0]*rz[0]);
    c[12] += 16.621506720583*(rx[7]*ry[0]*rz[2]);
    c[12] += 2.770251120097167*(rx[5]*ry[4]*rz[0]);
    c[12] += -16.621506720583*(rx[5]*ry[2]*rz[2]);
    c[12] += -55.40502240194333*(rx[5]*ry[0]*rz[4]);
    c[12] += 3.693668160129556*(rx[3]*ry[6]*rz[0]);
    c[12] += -83.107533602915*(rx[3]*ry[4]*rz[2]);
    c[12] += 110.8100448038867*(rx[3]*ry[2]*rz[4]);
    c[12] += 29.54934528103645*(rx[3]*ry[0]*rz[6]);
    c[12] += 1.385125560048583*(rx[1]*ry[8]*rz[0]);
    c[12] += -49.864520161749*(rx[1]*ry[6]*rz[2]);
    c[12] += 166.21506720583*(rx[1]*ry[4]*rz[4]);
    c[12] += -88.64803584310934*(rx[1]*ry[2]*rz[6]);
    c[13] += 4.077699238729173*(rx[8]*ry[0]*rz[1]);
    c[13] += -16.31079695491669*(rx[6]*ry[2]*rz[1]);
    c[13] += -32.62159390983339*(rx[6]*ry[0]*rz[3]);
    c[13] += -40.77699238729173*(rx[4]*ry[4]*rz[1]);
    c[13] += 163.1079695491669*(rx[4]*ry[2]*rz[3]);
    c[13] += 32.62159390983339*(rx[4]*ry[0]*rz[5]);
    c[13] += -16.31079695491669*(rx[2]*ry[6]*rz[1]);
    c[13] += 163.1079695491669*(rx[2]*ry[4]*rz[3]);
    c[13] += -195.7295634590003*(rx[2]*ry[2]*rz[5]);
    c[13] += 4.077699238729173*(rx[0]*ry[8]*rz[1]);
    c[13] += -32.62159390983339*(rx[0]*ry[6]*rz[3]);
    c[13] += 32.62159390983339*(rx[0]*ry[4]*rz[5]);
    c[14] += 0.4873782790390186*(rx[9]*ry[0]*rz[0]);
    c[14] += -3.899026232312149*(rx[7]*ry[2]*rz[0]);
    c[14] += -13.64659181309252*(rx[7]*ry[0]*rz[2]);
    c[14] += -6.82329590654626*(rx[5]*ry[4]*rz[0]);
    c[14] += 122.8193263178327*(rx[5]*ry[2]*rz[2]);
    c[14] += 27.29318362618504*(rx[5]*ry[0]*rz[4]);
    c[14] += 68.23295906546261*(rx[3]*ry[4]*rz[2]);
    c[14] += -272.9318362618504*(rx[3]*ry[2]*rz[4]);
    c[14] += 2.436891395195093*(rx[1]*ry[8]*rz[0]);
    c[14] += -68.23295906546261*(rx[1]*ry[6]*rz[2]);
    c[14] += 136.4659181309252*(rx[1]*ry[4]*rz[4]);
    c[15] += -3.775215916042701*(rx[8]*ry[0]*rz[1]);
    c[15] += 52.85302282459782*(rx[6]*ry[2]*rz[1]);
    c[15] += 17.61767427486594*(rx[6]*ry[0]*rz[3]);
    c[15] += -264.2651141229891*(rx[4]*ry[2]*rz[3]);
    c[15] += -52.85302282459782*(rx[2]*ry[6]*rz[1]);
    c[15] += 264.2651141229891*(rx[2]*ry[4]*rz[3]);
    c[15] += 3.775215916042701*(rx[0]*ry[8]*rz[1]);
    c[15] += -17.61767427486594*(rx[0]*ry[6]*rz[3]);
    c[16] += -0.5449054813440533*(rx[9]*ry[0]*rz[0]);
    c[16] += 10.89810962688107*(rx[7]*ry[2]*rz[0]);
    c[16] += 8.718487701504852*(rx[7]*ry[0]*rz[2]);
    c[16] += -7.628676738816745*(rx[5]*ry[4]*rz[0]);
    c[16] += -183.0882417316019*(rx[5]*ry[2]*rz[2]);
    c[16] += -15.25735347763349*(rx[3]*ry[6]*rz[0]);
    c[16] += 305.1470695526698*(rx[3]*ry[4]*rz[2]);
    c[16] += 3.814338369408373*(rx[1]*ry[8]*rz[0]);
    c[16] += -61.02941391053396*(rx[1]*ry[6]*rz[2]);
    c[17] += 3.177317648954698*(rx[8]*ry[0]*rz[1]);
    c[17] += -88.96489417073154*(rx[6]*ry[2]*rz[1]);
    c[17] += 222.4122354268289*(rx[4]*ry[4]*rz[1]);
    c[17] += -88.96489417073154*(rx[2]*ry[6]*rz[1]);
    c[17] += 3.177317648954698*(rx[0]*ry[8]*rz[1]);
    c[18] += 0.7489009518531882*(rx[9]*ry[0]*rz[0]);
    c[18] += -26.96043426671477*(rx[7]*ry[2]*rz[0]);
    c[18] += 94.36151993350171*(rx[5]*ry[4]*rz[0]);
    c[18] += -62.9076799556678*(rx[3]*ry[6]*rz[0]);
    c[18] += 6.740108566678694*(rx[1]*ry[8]*rz[0]);;

    double buf[(ECP_LMAX+1)*(ECP_LMAX+2)/2];
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] = 0.0;
    double nuc;

    // l = 9, i = 0
    nuc = 0.0;
    nuc += c[10]*0.451093112065591;
    nuc += c[12]*-0.4617085200161945;
    nuc += c[14]*0.4873782790390186;
    nuc += c[16]*-0.5449054813440533;
    nuc += c[18]*0.7489009518531882;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+9, j+pv+0, k+pw+0);
    }
    // l = 9, i = 1
    nuc = 0.0;
    nuc += c[0]*6.740108566678694;
    nuc += c[2]*-3.814338369408373;
    nuc += c[4]*2.436891395195093;
    nuc += c[6]*-1.385125560048583;
    nuc += c[8]*0.451093112065591;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+8, j+pv+1, k+pw+0);
    }
    // l = 9, i = 2
    nuc = 0.0;
    nuc += c[9]*3.026024588281776;
    nuc += c[11]*-4.23162848396049;
    nuc += c[13]*4.077699238729173;
    nuc += c[15]*-3.775215916042701;
    nuc += c[17]*3.177317648954698;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+8, j+pv+0, k+pw+1);
    }
    // l = 9, i = 3
    nuc = 0.0;
    nuc += c[10]*1.804372448262364;
    nuc += c[14]*-3.899026232312149;
    nuc += c[16]*10.89810962688107;
    nuc += c[18]*-26.96043426671477;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+7, j+pv+2, k+pw+0);
    }
    // l = 9, i = 4
    nuc = 0.0;
    nuc += c[1]*25.41854119163758;
    nuc += c[3]*-22.65129549625621;
    nuc += c[5]*16.31079695491669;
    nuc += c[7]*-8.46325696792098;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+7, j+pv+1, k+pw+1);
    }
    // l = 9, i = 5
    nuc = 0.0;
    nuc += c[10]*-18.04372448262364;
    nuc += c[12]*16.621506720583;
    nuc += c[14]*-13.64659181309252;
    nuc += c[16]*8.718487701504852;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+7, j+pv+0, k+pw+2);
    }
    // l = 9, i = 6
    nuc = 0.0;
    nuc += c[0]*-62.9076799556678;
    nuc += c[2]*15.25735347763349;
    nuc += c[6]*-3.693668160129556;
    nuc += c[8]*1.804372448262364;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+3, k+pw+0);
    }
    // l = 9, i = 7
    nuc = 0.0;
    nuc += c[9]*12.1040983531271;
    nuc += c[11]*-8.46325696792098;
    nuc += c[13]*-16.31079695491669;
    nuc += c[15]*52.85302282459782;
    nuc += c[17]*-88.96489417073154;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+2, k+pw+1);
    }
    // l = 9, i = 8
    nuc = 0.0;
    nuc += c[2]*61.02941391053396;
    nuc += c[4]*-68.23295906546261;
    nuc += c[6]*49.864520161749;
    nuc += c[8]*-18.04372448262364;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+1, k+pw+2);
    }
    // l = 9, i = 9
    nuc = 0.0;
    nuc += c[9]*-32.27759560833895;
    nuc += c[11]*42.3162848396049;
    nuc += c[13]*-32.62159390983339;
    nuc += c[15]*17.61767427486594;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+0, k+pw+3);
    }
    // l = 9, i = 10
    nuc = 0.0;
    nuc += c[10]*2.706558672393546;
    nuc += c[12]*2.770251120097167;
    nuc += c[14]*-6.82329590654626;
    nuc += c[16]*-7.628676738816745;
    nuc += c[18]*94.36151993350171;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+4, k+pw+0);
    }
    // l = 9, i = 11
    nuc = 0.0;
    nuc += c[1]*-177.9297883414631;
    nuc += c[3]*52.85302282459782;
    nuc += c[5]*16.31079695491669;
    nuc += c[7]*-25.38977090376294;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+3, k+pw+1);
    }
    // l = 9, i = 12
    nuc = 0.0;
    nuc += c[10]*-54.13117344787092;
    nuc += c[12]*-16.621506720583;
    nuc += c[14]*122.8193263178327;
    nuc += c[16]*-183.0882417316019;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+2, k+pw+2);
    }
    // l = 9, i = 13
    nuc = 0.0;
    nuc += c[3]*105.7060456491956;
    nuc += c[5]*-130.4863756393335;
    nuc += c[7]*84.6325696792098;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+1, k+pw+3);
    }
    // l = 9, i = 14
    nuc = 0.0;
    nuc += c[10]*72.17489793049457;
    nuc += c[12]*-55.40502240194333;
    nuc += c[14]*27.29318362618504;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+0, k+pw+4);
    }
    // l = 9, i = 15
    nuc = 0.0;
    nuc += c[0]*94.36151993350171;
    nuc += c[2]*7.628676738816745;
    nuc += c[4]*-6.82329590654626;
    nuc += c[6]*-2.770251120097167;
    nuc += c[8]*2.706558672393546;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+5, k+pw+0);
    }
    // l = 9, i = 16
    nuc = 0.0;
    nuc += c[9]*18.15614752969066;
    nuc += c[13]*-40.77699238729173;
    nuc += c[17]*222.4122354268289;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+4, k+pw+1);
    }
    // l = 9, i = 17
    nuc = 0.0;
    nuc += c[2]*-305.1470695526698;
    nuc += c[4]*68.23295906546261;
    nuc += c[6]*83.107533602915;
    nuc += c[8]*-54.13117344787092;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+3, k+pw+2);
    }
    // l = 9, i = 18
    nuc = 0.0;
    nuc += c[9]*-96.83278682501685;
    nuc += c[11]*42.3162848396049;
    nuc += c[13]*163.1079695491669;
    nuc += c[15]*-264.2651141229891;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+2, k+pw+3);
    }
    // l = 9, i = 19
    nuc = 0.0;
    nuc += c[4]*136.4659181309252;
    nuc += c[6]*-166.21506720583;
    nuc += c[8]*72.17489793049457;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+1, k+pw+4);
    }
    // l = 9, i = 20
    nuc = 0.0;
    nuc += c[9]*58.0996720950101;
    nuc += c[11]*-67.70605574336784;
    nuc += c[13]*32.62159390983339;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+0, k+pw+5);
    }
    // l = 9, i = 21
    nuc = 0.0;
    nuc += c[10]*1.804372448262364;
    nuc += c[12]*3.693668160129556;
    nuc += c[16]*-15.25735347763349;
    nuc += c[18]*-62.9076799556678;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+6, k+pw+0);
    }
    // l = 9, i = 22
    nuc = 0.0;
    nuc += c[1]*177.9297883414631;
    nuc += c[3]*52.85302282459782;
    nuc += c[5]*-16.31079695491669;
    nuc += c[7]*-25.38977090376294;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+5, k+pw+1);
    }
    // l = 9, i = 23
    nuc = 0.0;
    nuc += c[10]*-54.13117344787092;
    nuc += c[12]*-83.107533602915;
    nuc += c[14]*68.23295906546261;
    nuc += c[16]*305.1470695526698;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+4, k+pw+2);
    }
    // l = 9, i = 24
    nuc = 0.0;
    nuc += c[3]*-352.3534854973187;
    nuc += c[7]*169.2651393584196;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+3, k+pw+3);
    }
    // l = 9, i = 25
    nuc = 0.0;
    nuc += c[10]*144.3497958609891;
    nuc += c[12]*110.8100448038867;
    nuc += c[14]*-272.9318362618504;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+2, k+pw+4);
    }
    // l = 9, i = 26
    nuc = 0.0;
    nuc += c[5]*130.4863756393335;
    nuc += c[7]*-135.4121114867357;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+1, k+pw+5);
    }
    // l = 9, i = 27
    nuc = 0.0;
    nuc += c[10]*-57.73991834439565;
    nuc += c[12]*29.54934528103645;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+0, k+pw+6);
    }
    // l = 9, i = 28
    nuc = 0.0;
    nuc += c[0]*-26.96043426671477;
    nuc += c[2]*-10.89810962688107;
    nuc += c[4]*-3.899026232312149;
    nuc += c[8]*1.804372448262364;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+7, k+pw+0);
    }
    // l = 9, i = 29
    nuc = 0.0;
    nuc += c[9]*12.1040983531271;
    nuc += c[11]*8.46325696792098;
    nuc += c[13]*-16.31079695491669;
    nuc += c[15]*-52.85302282459782;
    nuc += c[17]*-88.96489417073154;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+6, k+pw+1);
    }
    // l = 9, i = 30
    nuc = 0.0;
    nuc += c[2]*183.0882417316019;
    nuc += c[4]*122.8193263178327;
    nuc += c[6]*16.621506720583;
    nuc += c[8]*-54.13117344787092;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+5, k+pw+2);
    }
    // l = 9, i = 31
    nuc = 0.0;
    nuc += c[9]*-96.83278682501685;
    nuc += c[11]*-42.3162848396049;
    nuc += c[13]*163.1079695491669;
    nuc += c[15]*264.2651141229891;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+4, k+pw+3);
    }
    // l = 9, i = 32
    nuc = 0.0;
    nuc += c[4]*-272.9318362618504;
    nuc += c[6]*-110.8100448038867;
    nuc += c[8]*144.3497958609891;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+3, k+pw+4);
    }
    // l = 9, i = 33
    nuc = 0.0;
    nuc += c[9]*116.1993441900202;
    nuc += c[13]*-195.7295634590003;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+2, k+pw+5);
    }
    // l = 9, i = 34
    nuc = 0.0;
    nuc += c[6]*88.64803584310934;
    nuc += c[8]*-57.73991834439565;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+1, k+pw+6);
    }
    // l = 9, i = 35
    nuc = 0.0;
    nuc += c[9]*-22.1332084171467;
    nuc += c[11]*19.34458735524795;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+0, k+pw+7);
    }
    // l = 9, i = 36
    nuc = 0.0;
    nuc += c[10]*0.451093112065591;
    nuc += c[12]*1.385125560048583;
    nuc += c[14]*2.436891395195093;
    nuc += c[16]*3.814338369408373;
    nuc += c[18]*6.740108566678694;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+8, k+pw+0);
    }
    // l = 9, i = 37
    nuc = 0.0;
    nuc += c[1]*-25.41854119163758;
    nuc += c[3]*-22.65129549625621;
    nuc += c[5]*-16.31079695491669;
    nuc += c[7]*-8.46325696792098;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+7, k+pw+1);
    }
    // l = 9, i = 38
    nuc = 0.0;
    nuc += c[10]*-18.04372448262364;
    nuc += c[12]*-49.864520161749;
    nuc += c[14]*-68.23295906546261;
    nuc += c[16]*-61.02941391053396;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+6, k+pw+2);
    }
    // l = 9, i = 39
    nuc = 0.0;
    nuc += c[3]*105.7060456491956;
    nuc += c[5]*130.4863756393335;
    nuc += c[7]*84.6325696792098;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+5, k+pw+3);
    }
    // l = 9, i = 40
    nuc = 0.0;
    nuc += c[10]*72.17489793049457;
    nuc += c[12]*166.21506720583;
    nuc += c[14]*136.4659181309252;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+4, k+pw+4);
    }
    // l = 9, i = 41
    nuc = 0.0;
    nuc += c[5]*-130.4863756393335;
    nuc += c[7]*-135.4121114867357;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+3, k+pw+5);
    }
    // l = 9, i = 42
    nuc = 0.0;
    nuc += c[10]*-57.73991834439565;
    nuc += c[12]*-88.64803584310934;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+2, k+pw+6);
    }
    // l = 9, i = 43
    nuc = 0.0;
    nuc += c[7]*38.68917471049591;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+1, k+pw+7);
    }
    // l = 9, i = 44
    nuc = 0.0;
    nuc += c[10]*8.248559763485094;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+0, k+pw+8);
    }
    // l = 9, i = 45
    nuc = 0.0;
    nuc += c[0]*0.7489009518531882;
    nuc += c[2]*0.5449054813440533;
    nuc += c[4]*0.4873782790390186;
    nuc += c[6]*0.4617085200161945;
    nuc += c[8]*0.451093112065591;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+9, k+pw+0);
    }
    // l = 9, i = 46
    nuc = 0.0;
    nuc += c[9]*3.026024588281776;
    nuc += c[11]*4.23162848396049;
    nuc += c[13]*4.077699238729173;
    nuc += c[15]*3.775215916042701;
    nuc += c[17]*3.177317648954698;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+8, k+pw+1);
    }
    // l = 9, i = 47
    nuc = 0.0;
    nuc += c[2]*-8.718487701504852;
    nuc += c[4]*-13.64659181309252;
    nuc += c[6]*-16.621506720583;
    nuc += c[8]*-18.04372448262364;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+7, k+pw+2);
    }
    // l = 9, i = 48
    nuc = 0.0;
    nuc += c[9]*-32.27759560833895;
    nuc += c[11]*-42.3162848396049;
    nuc += c[13]*-32.62159390983339;
    nuc += c[15]*-17.61767427486594;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+6, k+pw+3);
    }
    // l = 9, i = 49
    nuc = 0.0;
    nuc += c[4]*27.29318362618504;
    nuc += c[6]*55.40502240194333;
    nuc += c[8]*72.17489793049457;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+5, k+pw+4);
    }
    // l = 9, i = 50
    nuc = 0.0;
    nuc += c[9]*58.0996720950101;
    nuc += c[11]*67.70605574336784;
    nuc += c[13]*32.62159390983339;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+4, k+pw+5);
    }
    // l = 9, i = 51
    nuc = 0.0;
    nuc += c[6]*-29.54934528103645;
    nuc += c[8]*-57.73991834439565;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+3, k+pw+6);
    }
    // l = 9, i = 52
    nuc = 0.0;
    nuc += c[9]*-22.1332084171467;
    nuc += c[11]*-19.34458735524795;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+2, k+pw+7);
    }
    // l = 9, i = 53
    nuc = 0.0;
    nuc += c[8]*8.248559763485094;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+1, k+pw+8);
    }
    // l = 9, i = 54
    nuc = 0.0;
    nuc += c[9]*1.229622689841484;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+0, k+pw+9);
    }

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] *= 4.0 * M_PI;
    cart2sph(omega, lc, buf);
}
template <> __device__
void type2_ang_nuc_l<10>(double * __restrict__ omega, const int lc,
                    const int i, const int j, const int k,
                    double * __restrict__ unitr){
    constexpr int l = 10;
    double rx[l+1], ry[l+1], rz[l+1];
    rx[0] = ry[0] = rz[0] = 1.0;
    for (int li = 1; li <= l; li++) {
        rx[li] = rx[li - 1] * unitr[0];
        ry[li] = ry[li - 1] * unitr[1];
        rz[li] = rz[li - 1] * unitr[2];
    }

    double c[2*l+1];
    for (int m = 0; m < 2*l+1; m++) c[m] = 0.0;
    c[0] += 7.673951182219901*(rx[9]*ry[1]*rz[0]);
    c[0] += -92.08741418663881*(rx[7]*ry[3]*rz[0]);
    c[0] += 193.3835697919415*(rx[5]*ry[5]*rz[0]);
    c[0] += -92.08741418663881*(rx[3]*ry[7]*rz[0]);
    c[0] += 7.673951182219901*(rx[1]*ry[9]*rz[0]);
    c[1] += 30.88705769902543*(rx[8]*ry[1]*rz[1]);
    c[1] += -288.2792051909041*(rx[6]*ry[3]*rz[1]);
    c[1] += 432.4188077863561*(rx[4]*ry[5]*rz[1]);
    c[1] += -123.5482307961017*(rx[2]*ry[7]*rz[1]);
    c[1] += 3.431895299891715*(rx[0]*ry[9]*rz[1]);
    c[2] += -4.453815461763347*(rx[9]*ry[1]*rz[0]);
    c[2] += 26.72289277058008*(rx[7]*ry[3]*rz[0]);
    c[2] += 80.16867831174027*(rx[7]*ry[1]*rz[2]);
    c[2] += -561.1807481821819*(rx[5]*ry[3]*rz[2]);
    c[2] += -26.72289277058008*(rx[3]*ry[7]*rz[0]);
    c[2] += 561.1807481821819*(rx[3]*ry[5]*rz[2]);
    c[2] += 4.453815461763347*(rx[1]*ry[9]*rz[0]);
    c[2] += -80.16867831174027*(rx[1]*ry[7]*rz[2]);
    c[3] += -28.63763513582592*(rx[8]*ry[1]*rz[1]);
    c[3] += 114.5505405433037*(rx[6]*ry[3]*rz[1]);
    c[3] += 152.7340540577382*(rx[6]*ry[1]*rz[3]);
    c[3] += 57.27527027165184*(rx[4]*ry[5]*rz[1]);
    c[3] += -763.6702702886912*(rx[4]*ry[3]*rz[3]);
    c[3] += -81.82181467378834*(rx[2]*ry[7]*rz[1]);
    c[3] += 458.2021621732147*(rx[2]*ry[5]*rz[3]);
    c[3] += 4.091090733689417*(rx[0]*ry[9]*rz[1]);
    c[3] += -21.81915057967689*(rx[0]*ry[7]*rz[3]);
    c[4] += 2.976705744527138*(rx[9]*ry[1]*rz[0]);
    c[4] += -3.968940992702851*(rx[7]*ry[3]*rz[0]);
    c[4] += -95.25458382486842*(rx[7]*ry[1]*rz[2]);
    c[4] += -13.89129347445998*(rx[5]*ry[5]*rz[0]);
    c[4] += 222.2606955913596*(rx[5]*ry[3]*rz[2]);
    c[4] += 222.2606955913597*(rx[5]*ry[1]*rz[4]);
    c[4] += -3.968940992702851*(rx[3]*ry[7]*rz[0]);
    c[4] += 222.2606955913596*(rx[3]*ry[5]*rz[2]);
    c[4] += -740.8689853045323*(rx[3]*ry[3]*rz[4]);
    c[4] += 2.976705744527138*(rx[1]*ry[9]*rz[0]);
    c[4] += -95.25458382486842*(rx[1]*ry[7]*rz[2]);
    c[4] += 222.2606955913597*(rx[1]*ry[5]*rz[4]);
    c[5] += 22.18705464592268*(rx[8]*ry[1]*rz[1]);
    c[5] += -207.0791766952783*(rx[6]*ry[1]*rz[3]);
    c[5] += -62.12375300858349*(rx[4]*ry[5]*rz[1]);
    c[5] += 207.0791766952783*(rx[4]*ry[3]*rz[3]);
    c[5] += 248.495012034334*(rx[4]*ry[1]*rz[5]);
    c[5] += -35.49928743347628*(rx[2]*ry[7]*rz[1]);
    c[5] += 372.742518051501*(rx[2]*ry[5]*rz[3]);
    c[5] += -496.990024068668*(rx[2]*ry[3]*rz[5]);
    c[5] += 4.437410929184535*(rx[0]*ry[9]*rz[1]);
    c[5] += -41.41583533905566*(rx[0]*ry[7]*rz[3]);
    c[5] += 49.6990024068668*(rx[0]*ry[5]*rz[5]);
    c[6] += -1.870976726712969*(rx[9]*ry[1]*rz[0]);
    c[6] += -3.741953453425937*(rx[7]*ry[3]*rz[0]);
    c[6] += 78.58102252194469*(rx[7]*ry[1]*rz[2]);
    c[6] += 78.58102252194469*(rx[5]*ry[3]*rz[2]);
    c[6] += -314.3240900877788*(rx[5]*ry[1]*rz[4]);
    c[6] += 3.741953453425937*(rx[3]*ry[7]*rz[0]);
    c[6] += -78.58102252194469*(rx[3]*ry[5]*rz[2]);
    c[6] += 209.5493933918525*(rx[3]*ry[1]*rz[6]);
    c[6] += 1.870976726712969*(rx[1]*ry[9]*rz[0]);
    c[6] += -78.58102252194469*(rx[1]*ry[7]*rz[2]);
    c[6] += 314.3240900877788*(rx[1]*ry[5]*rz[4]);
    c[6] += -209.5493933918525*(rx[1]*ry[3]*rz[6]);
    c[7] += -13.89129347445998*(rx[8]*ry[1]*rz[1]);
    c[7] += -37.04344926522661*(rx[6]*ry[3]*rz[1]);
    c[7] += 166.6955216935197*(rx[6]*ry[1]*rz[3]);
    c[7] += -27.78258694891996*(rx[4]*ry[5]*rz[1]);
    c[7] += 277.8258694891996*(rx[4]*ry[3]*rz[3]);
    c[7] += -333.3910433870395*(rx[4]*ry[1]*rz[5]);
    c[7] += 55.56517389783991*(rx[2]*ry[5]*rz[3]);
    c[7] += -222.2606955913596*(rx[2]*ry[3]*rz[5]);
    c[7] += 127.0061117664912*(rx[2]*ry[1]*rz[7]);
    c[7] += 4.630431158153326*(rx[0]*ry[9]*rz[1]);
    c[7] += -55.56517389783991*(rx[0]*ry[7]*rz[3]);
    c[7] += 111.1303477956798*(rx[0]*ry[5]*rz[5]);
    c[7] += -42.33537058883041*(rx[0]*ry[3]*rz[7]);
    c[8] += 0.9081022627604556*(rx[9]*ry[1]*rz[0]);
    c[8] += 3.632409051041822*(rx[7]*ry[3]*rz[0]);
    c[8] += -43.58890861250187*(rx[7]*ry[1]*rz[2]);
    c[8] += 5.448613576562733*(rx[5]*ry[5]*rz[0]);
    c[8] += -130.7667258375056*(rx[5]*ry[3]*rz[2]);
    c[8] += 217.9445430625093*(rx[5]*ry[1]*rz[4]);
    c[8] += 3.632409051041822*(rx[3]*ry[7]*rz[0]);
    c[8] += -130.7667258375056*(rx[3]*ry[5]*rz[2]);
    c[8] += 435.8890861250187*(rx[3]*ry[3]*rz[4]);
    c[8] += -232.4741792666766*(rx[3]*ry[1]*rz[6]);
    c[8] += 0.9081022627604556*(rx[1]*ry[9]*rz[0]);
    c[8] += -43.58890861250187*(rx[1]*ry[7]*rz[2]);
    c[8] += 217.9445430625093*(rx[1]*ry[5]*rz[4]);
    c[8] += -232.4741792666766*(rx[1]*ry[3]*rz[6]);
    c[8] += 49.815895557145*(rx[1]*ry[1]*rz[8]);
    c[9] += 4.718637772708116*(rx[8]*ry[1]*rz[1]);
    c[9] += 18.87455109083247*(rx[6]*ry[3]*rz[1]);
    c[9] += -62.91517030277488*(rx[6]*ry[1]*rz[3]);
    c[9] += 28.3118266362487*(rx[4]*ry[5]*rz[1]);
    c[9] += -188.7455109083247*(rx[4]*ry[3]*rz[3]);
    c[9] += 150.9964087266597*(rx[4]*ry[1]*rz[5]);
    c[9] += 18.87455109083247*(rx[2]*ry[7]*rz[1]);
    c[9] += -188.7455109083247*(rx[2]*ry[5]*rz[3]);
    c[9] += 301.9928174533194*(rx[2]*ry[3]*rz[5]);
    c[9] += -86.28366212951984*(rx[2]*ry[1]*rz[7]);
    c[9] += 4.718637772708116*(rx[0]*ry[9]*rz[1]);
    c[9] += -62.91517030277488*(rx[0]*ry[7]*rz[3]);
    c[9] += 150.9964087266597*(rx[0]*ry[5]*rz[5]);
    c[9] += -86.28366212951984*(rx[0]*ry[3]*rz[7]);
    c[9] += 9.587073569946648*(rx[0]*ry[1]*rz[9]);
    c[10] += -0.3181304937373671*(rx[10]*ry[0]*rz[0]);
    c[10] += -1.590652468686835*(rx[8]*ry[2]*rz[0]);
    c[10] += 15.90652468686835*(rx[8]*ry[0]*rz[2]);
    c[10] += -3.181304937373671*(rx[6]*ry[4]*rz[0]);
    c[10] += 63.62609874747341*(rx[6]*ry[2]*rz[2]);
    c[10] += -84.83479832996456*(rx[6]*ry[0]*rz[4]);
    c[10] += -3.181304937373671*(rx[4]*ry[6]*rz[0]);
    c[10] += 95.43914812121012*(rx[4]*ry[4]*rz[2]);
    c[10] += -254.5043949898937*(rx[4]*ry[2]*rz[4]);
    c[10] += 101.8017579959575*(rx[4]*ry[0]*rz[6]);
    c[10] += -1.590652468686835*(rx[2]*ry[8]*rz[0]);
    c[10] += 63.62609874747341*(rx[2]*ry[6]*rz[2]);
    c[10] += -254.5043949898937*(rx[2]*ry[4]*rz[4]);
    c[10] += 203.6035159919149*(rx[2]*ry[2]*rz[6]);
    c[10] += -29.08621657027356*(rx[2]*ry[0]*rz[8]);
    c[10] += -0.3181304937373671*(rx[0]*ry[10]*rz[0]);
    c[10] += 15.90652468686835*(rx[0]*ry[8]*rz[2]);
    c[10] += -84.83479832996456*(rx[0]*ry[6]*rz[4]);
    c[10] += 101.8017579959575*(rx[0]*ry[4]*rz[6]);
    c[10] += -29.08621657027356*(rx[0]*ry[2]*rz[8]);
    c[10] += 1.292720736456603*(rx[0]*ry[0]*rz[10]);
    c[11] += 4.718637772708116*(rx[9]*ry[0]*rz[1]);
    c[11] += 18.87455109083247*(rx[7]*ry[2]*rz[1]);
    c[11] += -62.91517030277488*(rx[7]*ry[0]*rz[3]);
    c[11] += 28.3118266362487*(rx[5]*ry[4]*rz[1]);
    c[11] += -188.7455109083247*(rx[5]*ry[2]*rz[3]);
    c[11] += 150.9964087266597*(rx[5]*ry[0]*rz[5]);
    c[11] += 18.87455109083247*(rx[3]*ry[6]*rz[1]);
    c[11] += -188.7455109083247*(rx[3]*ry[4]*rz[3]);
    c[11] += 301.9928174533194*(rx[3]*ry[2]*rz[5]);
    c[11] += -86.28366212951984*(rx[3]*ry[0]*rz[7]);
    c[11] += 4.718637772708116*(rx[1]*ry[8]*rz[1]);
    c[11] += -62.91517030277488*(rx[1]*ry[6]*rz[3]);
    c[11] += 150.9964087266597*(rx[1]*ry[4]*rz[5]);
    c[11] += -86.28366212951984*(rx[1]*ry[2]*rz[7]);
    c[11] += 9.587073569946648*(rx[1]*ry[0]*rz[9]);
    c[12] += 0.4540511313802278*(rx[10]*ry[0]*rz[0]);
    c[12] += 1.362153394140683*(rx[8]*ry[2]*rz[0]);
    c[12] += -21.79445430625093*(rx[8]*ry[0]*rz[2]);
    c[12] += 0.9081022627604556*(rx[6]*ry[4]*rz[0]);
    c[12] += -43.58890861250187*(rx[6]*ry[2]*rz[2]);
    c[12] += 108.9722715312547*(rx[6]*ry[0]*rz[4]);
    c[12] += -0.9081022627604556*(rx[4]*ry[6]*rz[0]);
    c[12] += 108.9722715312547*(rx[4]*ry[2]*rz[4]);
    c[12] += -116.2370896333383*(rx[4]*ry[0]*rz[6]);
    c[12] += -1.362153394140683*(rx[2]*ry[8]*rz[0]);
    c[12] += 43.58890861250187*(rx[2]*ry[6]*rz[2]);
    c[12] += -108.9722715312547*(rx[2]*ry[4]*rz[4]);
    c[12] += 24.9079477785725*(rx[2]*ry[0]*rz[8]);
    c[12] += -0.4540511313802278*(rx[0]*ry[10]*rz[0]);
    c[12] += 21.79445430625093*(rx[0]*ry[8]*rz[2]);
    c[12] += -108.9722715312547*(rx[0]*ry[6]*rz[4]);
    c[12] += 116.2370896333383*(rx[0]*ry[4]*rz[6]);
    c[12] += -24.9079477785725*(rx[0]*ry[2]*rz[8]);
    c[13] += -4.630431158153326*(rx[9]*ry[0]*rz[1]);
    c[13] += 55.56517389783991*(rx[7]*ry[0]*rz[3]);
    c[13] += 27.78258694891996*(rx[5]*ry[4]*rz[1]);
    c[13] += -55.56517389783991*(rx[5]*ry[2]*rz[3]);
    c[13] += -111.1303477956798*(rx[5]*ry[0]*rz[5]);
    c[13] += 37.04344926522661*(rx[3]*ry[6]*rz[1]);
    c[13] += -277.8258694891996*(rx[3]*ry[4]*rz[3]);
    c[13] += 222.2606955913596*(rx[3]*ry[2]*rz[5]);
    c[13] += 42.33537058883041*(rx[3]*ry[0]*rz[7]);
    c[13] += 13.89129347445998*(rx[1]*ry[8]*rz[1]);
    c[13] += -166.6955216935197*(rx[1]*ry[6]*rz[3]);
    c[13] += 333.3910433870395*(rx[1]*ry[4]*rz[5]);
    c[13] += -127.0061117664912*(rx[1]*ry[2]*rz[7]);
    c[14] += -0.4677441816782422*(rx[10]*ry[0]*rz[0]);
    c[14] += 1.403232545034726*(rx[8]*ry[2]*rz[0]);
    c[14] += 19.64525563048617*(rx[8]*ry[0]*rz[2]);
    c[14] += 6.548418543495391*(rx[6]*ry[4]*rz[0]);
    c[14] += -78.58102252194469*(rx[6]*ry[2]*rz[2]);
    c[14] += -78.58102252194469*(rx[6]*ry[0]*rz[4]);
    c[14] += 6.548418543495391*(rx[4]*ry[6]*rz[0]);
    c[14] += -196.4525563048617*(rx[4]*ry[4]*rz[2]);
    c[14] += 392.9051126097235*(rx[4]*ry[2]*rz[4]);
    c[14] += 52.38734834796313*(rx[4]*ry[0]*rz[6]);
    c[14] += 1.403232545034726*(rx[2]*ry[8]*rz[0]);
    c[14] += -78.58102252194469*(rx[2]*ry[6]*rz[2]);
    c[14] += 392.9051126097235*(rx[2]*ry[4]*rz[4]);
    c[14] += -314.3240900877788*(rx[2]*ry[2]*rz[6]);
    c[14] += -0.4677441816782422*(rx[0]*ry[10]*rz[0]);
    c[14] += 19.64525563048617*(rx[0]*ry[8]*rz[2]);
    c[14] += -78.58102252194469*(rx[0]*ry[6]*rz[4]);
    c[14] += 52.38734834796313*(rx[0]*ry[4]*rz[6]);
    c[15] += 4.437410929184535*(rx[9]*ry[0]*rz[1]);
    c[15] += -35.49928743347628*(rx[7]*ry[2]*rz[1]);
    c[15] += -41.41583533905566*(rx[7]*ry[0]*rz[3]);
    c[15] += -62.12375300858349*(rx[5]*ry[4]*rz[1]);
    c[15] += 372.742518051501*(rx[5]*ry[2]*rz[3]);
    c[15] += 49.6990024068668*(rx[5]*ry[0]*rz[5]);
    c[15] += 207.0791766952783*(rx[3]*ry[4]*rz[3]);
    c[15] += -496.990024068668*(rx[3]*ry[2]*rz[5]);
    c[15] += 22.18705464592268*(rx[1]*ry[8]*rz[1]);
    c[15] += -207.0791766952783*(rx[1]*ry[6]*rz[3]);
    c[15] += 248.495012034334*(rx[1]*ry[4]*rz[5]);
    c[16] += 0.4961176240878564*(rx[10]*ry[0]*rz[0]);
    c[16] += -6.449529113142133*(rx[8]*ry[2]*rz[0]);
    c[16] += -15.8757639708114*(rx[8]*ry[0]*rz[2]);
    c[16] += -6.945646737229989*(rx[6]*ry[4]*rz[0]);
    c[16] += 222.2606955913596*(rx[6]*ry[2]*rz[2]);
    c[16] += 37.04344926522661*(rx[6]*ry[0]*rz[4]);
    c[16] += 6.945646737229989*(rx[4]*ry[6]*rz[0]);
    c[16] += -555.6517389783992*(rx[4]*ry[2]*rz[4]);
    c[16] += 6.449529113142133*(rx[2]*ry[8]*rz[0]);
    c[16] += -222.2606955913596*(rx[2]*ry[6]*rz[2]);
    c[16] += 555.6517389783992*(rx[2]*ry[4]*rz[4]);
    c[16] += -0.4961176240878564*(rx[0]*ry[10]*rz[0]);
    c[16] += 15.8757639708114*(rx[0]*ry[8]*rz[2]);
    c[16] += -37.04344926522661*(rx[0]*ry[6]*rz[4]);
    c[17] += -4.091090733689417*(rx[9]*ry[0]*rz[1]);
    c[17] += 81.82181467378834*(rx[7]*ry[2]*rz[1]);
    c[17] += 21.81915057967689*(rx[7]*ry[0]*rz[3]);
    c[17] += -57.27527027165184*(rx[5]*ry[4]*rz[1]);
    c[17] += -458.2021621732147*(rx[5]*ry[2]*rz[3]);
    c[17] += -114.5505405433037*(rx[3]*ry[6]*rz[1]);
    c[17] += 763.6702702886912*(rx[3]*ry[4]*rz[3]);
    c[17] += 28.63763513582592*(rx[1]*ry[8]*rz[1]);
    c[17] += -152.7340540577382*(rx[1]*ry[6]*rz[3]);
    c[18] += -0.5567269327204184*(rx[10]*ry[0]*rz[0]);
    c[18] += 15.0316271834513*(rx[8]*ry[2]*rz[0]);
    c[18] += 10.02108478896753*(rx[8]*ry[0]*rz[2]);
    c[18] += -23.38253117425757*(rx[6]*ry[4]*rz[0]);
    c[18] += -280.590374091091*(rx[6]*ry[2]*rz[2]);
    c[18] += -23.38253117425757*(rx[4]*ry[6]*rz[0]);
    c[18] += 701.4759352277273*(rx[4]*ry[4]*rz[2]);
    c[18] += 15.0316271834513*(rx[2]*ry[8]*rz[0]);
    c[18] += -280.590374091091*(rx[2]*ry[6]*rz[2]);
    c[18] += -0.5567269327204184*(rx[0]*ry[10]*rz[0]);
    c[18] += 10.02108478896753*(rx[0]*ry[8]*rz[2]);
    c[19] += 3.431895299891715*(rx[9]*ry[0]*rz[1]);
    c[19] += -123.5482307961017*(rx[7]*ry[2]*rz[1]);
    c[19] += 432.4188077863561*(rx[5]*ry[4]*rz[1]);
    c[19] += -288.2792051909041*(rx[3]*ry[6]*rz[1]);
    c[19] += 30.88705769902543*(rx[1]*ry[8]*rz[1]);
    c[20] += 0.7673951182219901*(rx[10]*ry[0]*rz[0]);
    c[20] += -34.53278031998956*(rx[8]*ry[2]*rz[0]);
    c[20] += 161.1529748266179*(rx[6]*ry[4]*rz[0]);
    c[20] += -161.1529748266179*(rx[4]*ry[6]*rz[0]);
    c[20] += 34.53278031998956*(rx[2]*ry[8]*rz[0]);
    c[20] += -0.7673951182219901*(rx[0]*ry[10]*rz[0]);;

    double buf[(ECP_LMAX+1)*(ECP_LMAX+2)/2];
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] = 0.0;
    double nuc;

    // l = 10, i = 0
    nuc = 0.0;
    nuc += c[10]*-0.3181304937373671;
    nuc += c[12]*0.4540511313802278;
    nuc += c[14]*-0.4677441816782422;
    nuc += c[16]*0.4961176240878564;
    nuc += c[18]*-0.5567269327204184;
    nuc += c[20]*0.7673951182219901;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+10, j+pv+0, k+pw+0);
    }
    // l = 10, i = 1
    nuc = 0.0;
    nuc += c[0]*7.673951182219901;
    nuc += c[2]*-4.453815461763347;
    nuc += c[4]*2.976705744527138;
    nuc += c[6]*-1.870976726712969;
    nuc += c[8]*0.9081022627604556;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+9, j+pv+1, k+pw+0);
    }
    // l = 10, i = 2
    nuc = 0.0;
    nuc += c[11]*4.718637772708116;
    nuc += c[13]*-4.630431158153326;
    nuc += c[15]*4.437410929184535;
    nuc += c[17]*-4.091090733689417;
    nuc += c[19]*3.431895299891715;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+9, j+pv+0, k+pw+1);
    }
    // l = 10, i = 3
    nuc = 0.0;
    nuc += c[10]*-1.590652468686835;
    nuc += c[12]*1.362153394140683;
    nuc += c[14]*1.403232545034726;
    nuc += c[16]*-6.449529113142133;
    nuc += c[18]*15.0316271834513;
    nuc += c[20]*-34.53278031998956;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+8, j+pv+2, k+pw+0);
    }
    // l = 10, i = 4
    nuc = 0.0;
    nuc += c[1]*30.88705769902543;
    nuc += c[3]*-28.63763513582592;
    nuc += c[5]*22.18705464592268;
    nuc += c[7]*-13.89129347445998;
    nuc += c[9]*4.718637772708116;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+8, j+pv+1, k+pw+1);
    }
    // l = 10, i = 5
    nuc = 0.0;
    nuc += c[10]*15.90652468686835;
    nuc += c[12]*-21.79445430625093;
    nuc += c[14]*19.64525563048617;
    nuc += c[16]*-15.8757639708114;
    nuc += c[18]*10.02108478896753;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+8, j+pv+0, k+pw+2);
    }
    // l = 10, i = 6
    nuc = 0.0;
    nuc += c[0]*-92.08741418663881;
    nuc += c[2]*26.72289277058008;
    nuc += c[4]*-3.968940992702851;
    nuc += c[6]*-3.741953453425937;
    nuc += c[8]*3.632409051041822;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+7, j+pv+3, k+pw+0);
    }
    // l = 10, i = 7
    nuc = 0.0;
    nuc += c[11]*18.87455109083247;
    nuc += c[15]*-35.49928743347628;
    nuc += c[17]*81.82181467378834;
    nuc += c[19]*-123.5482307961017;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+7, j+pv+2, k+pw+1);
    }
    // l = 10, i = 8
    nuc = 0.0;
    nuc += c[2]*80.16867831174027;
    nuc += c[4]*-95.25458382486842;
    nuc += c[6]*78.58102252194469;
    nuc += c[8]*-43.58890861250187;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+7, j+pv+1, k+pw+2);
    }
    // l = 10, i = 9
    nuc = 0.0;
    nuc += c[11]*-62.91517030277488;
    nuc += c[13]*55.56517389783991;
    nuc += c[15]*-41.41583533905566;
    nuc += c[17]*21.81915057967689;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+7, j+pv+0, k+pw+3);
    }
    // l = 10, i = 10
    nuc = 0.0;
    nuc += c[10]*-3.181304937373671;
    nuc += c[12]*0.9081022627604556;
    nuc += c[14]*6.548418543495391;
    nuc += c[16]*-6.945646737229989;
    nuc += c[18]*-23.38253117425757;
    nuc += c[20]*161.1529748266179;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+4, k+pw+0);
    }
    // l = 10, i = 11
    nuc = 0.0;
    nuc += c[1]*-288.2792051909041;
    nuc += c[3]*114.5505405433037;
    nuc += c[7]*-37.04344926522661;
    nuc += c[9]*18.87455109083247;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+3, k+pw+1);
    }
    // l = 10, i = 12
    nuc = 0.0;
    nuc += c[10]*63.62609874747341;
    nuc += c[12]*-43.58890861250187;
    nuc += c[14]*-78.58102252194469;
    nuc += c[16]*222.2606955913596;
    nuc += c[18]*-280.590374091091;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+2, k+pw+2);
    }
    // l = 10, i = 13
    nuc = 0.0;
    nuc += c[3]*152.7340540577382;
    nuc += c[5]*-207.0791766952783;
    nuc += c[7]*166.6955216935197;
    nuc += c[9]*-62.91517030277488;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+1, k+pw+3);
    }
    // l = 10, i = 14
    nuc = 0.0;
    nuc += c[10]*-84.83479832996456;
    nuc += c[12]*108.9722715312547;
    nuc += c[14]*-78.58102252194469;
    nuc += c[16]*37.04344926522661;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+6, j+pv+0, k+pw+4);
    }
    // l = 10, i = 15
    nuc = 0.0;
    nuc += c[0]*193.3835697919415;
    nuc += c[4]*-13.89129347445998;
    nuc += c[8]*5.448613576562733;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+5, k+pw+0);
    }
    // l = 10, i = 16
    nuc = 0.0;
    nuc += c[11]*28.3118266362487;
    nuc += c[13]*27.78258694891996;
    nuc += c[15]*-62.12375300858349;
    nuc += c[17]*-57.27527027165184;
    nuc += c[19]*432.4188077863561;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+4, k+pw+1);
    }
    // l = 10, i = 17
    nuc = 0.0;
    nuc += c[2]*-561.1807481821819;
    nuc += c[4]*222.2606955913596;
    nuc += c[6]*78.58102252194469;
    nuc += c[8]*-130.7667258375056;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+3, k+pw+2);
    }
    // l = 10, i = 18
    nuc = 0.0;
    nuc += c[11]*-188.7455109083247;
    nuc += c[13]*-55.56517389783991;
    nuc += c[15]*372.742518051501;
    nuc += c[17]*-458.2021621732147;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+2, k+pw+3);
    }
    // l = 10, i = 19
    nuc = 0.0;
    nuc += c[4]*222.2606955913597;
    nuc += c[6]*-314.3240900877788;
    nuc += c[8]*217.9445430625093;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+1, k+pw+4);
    }
    // l = 10, i = 20
    nuc = 0.0;
    nuc += c[11]*150.9964087266597;
    nuc += c[13]*-111.1303477956798;
    nuc += c[15]*49.6990024068668;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+5, j+pv+0, k+pw+5);
    }
    // l = 10, i = 21
    nuc = 0.0;
    nuc += c[10]*-3.181304937373671;
    nuc += c[12]*-0.9081022627604556;
    nuc += c[14]*6.548418543495391;
    nuc += c[16]*6.945646737229989;
    nuc += c[18]*-23.38253117425757;
    nuc += c[20]*-161.1529748266179;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+6, k+pw+0);
    }
    // l = 10, i = 22
    nuc = 0.0;
    nuc += c[1]*432.4188077863561;
    nuc += c[3]*57.27527027165184;
    nuc += c[5]*-62.12375300858349;
    nuc += c[7]*-27.78258694891996;
    nuc += c[9]*28.3118266362487;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+5, k+pw+1);
    }
    // l = 10, i = 23
    nuc = 0.0;
    nuc += c[10]*95.43914812121012;
    nuc += c[14]*-196.4525563048617;
    nuc += c[18]*701.4759352277273;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+4, k+pw+2);
    }
    // l = 10, i = 24
    nuc = 0.0;
    nuc += c[3]*-763.6702702886912;
    nuc += c[5]*207.0791766952783;
    nuc += c[7]*277.8258694891996;
    nuc += c[9]*-188.7455109083247;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+3, k+pw+3);
    }
    // l = 10, i = 25
    nuc = 0.0;
    nuc += c[10]*-254.5043949898937;
    nuc += c[12]*108.9722715312547;
    nuc += c[14]*392.9051126097235;
    nuc += c[16]*-555.6517389783992;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+2, k+pw+4);
    }
    // l = 10, i = 26
    nuc = 0.0;
    nuc += c[5]*248.495012034334;
    nuc += c[7]*-333.3910433870395;
    nuc += c[9]*150.9964087266597;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+1, k+pw+5);
    }
    // l = 10, i = 27
    nuc = 0.0;
    nuc += c[10]*101.8017579959575;
    nuc += c[12]*-116.2370896333383;
    nuc += c[14]*52.38734834796313;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+4, j+pv+0, k+pw+6);
    }
    // l = 10, i = 28
    nuc = 0.0;
    nuc += c[0]*-92.08741418663881;
    nuc += c[2]*-26.72289277058008;
    nuc += c[4]*-3.968940992702851;
    nuc += c[6]*3.741953453425937;
    nuc += c[8]*3.632409051041822;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+7, k+pw+0);
    }
    // l = 10, i = 29
    nuc = 0.0;
    nuc += c[11]*18.87455109083247;
    nuc += c[13]*37.04344926522661;
    nuc += c[17]*-114.5505405433037;
    nuc += c[19]*-288.2792051909041;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+6, k+pw+1);
    }
    // l = 10, i = 30
    nuc = 0.0;
    nuc += c[2]*561.1807481821819;
    nuc += c[4]*222.2606955913596;
    nuc += c[6]*-78.58102252194469;
    nuc += c[8]*-130.7667258375056;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+5, k+pw+2);
    }
    // l = 10, i = 31
    nuc = 0.0;
    nuc += c[11]*-188.7455109083247;
    nuc += c[13]*-277.8258694891996;
    nuc += c[15]*207.0791766952783;
    nuc += c[17]*763.6702702886912;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+4, k+pw+3);
    }
    // l = 10, i = 32
    nuc = 0.0;
    nuc += c[4]*-740.8689853045323;
    nuc += c[8]*435.8890861250187;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+3, k+pw+4);
    }
    // l = 10, i = 33
    nuc = 0.0;
    nuc += c[11]*301.9928174533194;
    nuc += c[13]*222.2606955913596;
    nuc += c[15]*-496.990024068668;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+2, k+pw+5);
    }
    // l = 10, i = 34
    nuc = 0.0;
    nuc += c[6]*209.5493933918525;
    nuc += c[8]*-232.4741792666766;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+1, k+pw+6);
    }
    // l = 10, i = 35
    nuc = 0.0;
    nuc += c[11]*-86.28366212951984;
    nuc += c[13]*42.33537058883041;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+3, j+pv+0, k+pw+7);
    }
    // l = 10, i = 36
    nuc = 0.0;
    nuc += c[10]*-1.590652468686835;
    nuc += c[12]*-1.362153394140683;
    nuc += c[14]*1.403232545034726;
    nuc += c[16]*6.449529113142133;
    nuc += c[18]*15.0316271834513;
    nuc += c[20]*34.53278031998956;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+8, k+pw+0);
    }
    // l = 10, i = 37
    nuc = 0.0;
    nuc += c[1]*-123.5482307961017;
    nuc += c[3]*-81.82181467378834;
    nuc += c[5]*-35.49928743347628;
    nuc += c[9]*18.87455109083247;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+7, k+pw+1);
    }
    // l = 10, i = 38
    nuc = 0.0;
    nuc += c[10]*63.62609874747341;
    nuc += c[12]*43.58890861250187;
    nuc += c[14]*-78.58102252194469;
    nuc += c[16]*-222.2606955913596;
    nuc += c[18]*-280.590374091091;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+6, k+pw+2);
    }
    // l = 10, i = 39
    nuc = 0.0;
    nuc += c[3]*458.2021621732147;
    nuc += c[5]*372.742518051501;
    nuc += c[7]*55.56517389783991;
    nuc += c[9]*-188.7455109083247;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+5, k+pw+3);
    }
    // l = 10, i = 40
    nuc = 0.0;
    nuc += c[10]*-254.5043949898937;
    nuc += c[12]*-108.9722715312547;
    nuc += c[14]*392.9051126097235;
    nuc += c[16]*555.6517389783992;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+4, k+pw+4);
    }
    // l = 10, i = 41
    nuc = 0.0;
    nuc += c[5]*-496.990024068668;
    nuc += c[7]*-222.2606955913596;
    nuc += c[9]*301.9928174533194;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+3, k+pw+5);
    }
    // l = 10, i = 42
    nuc = 0.0;
    nuc += c[10]*203.6035159919149;
    nuc += c[14]*-314.3240900877788;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+2, k+pw+6);
    }
    // l = 10, i = 43
    nuc = 0.0;
    nuc += c[7]*127.0061117664912;
    nuc += c[9]*-86.28366212951984;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+1, k+pw+7);
    }
    // l = 10, i = 44
    nuc = 0.0;
    nuc += c[10]*-29.08621657027356;
    nuc += c[12]*24.9079477785725;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+2, j+pv+0, k+pw+8);
    }
    // l = 10, i = 45
    nuc = 0.0;
    nuc += c[0]*7.673951182219901;
    nuc += c[2]*4.453815461763347;
    nuc += c[4]*2.976705744527138;
    nuc += c[6]*1.870976726712969;
    nuc += c[8]*0.9081022627604556;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+9, k+pw+0);
    }
    // l = 10, i = 46
    nuc = 0.0;
    nuc += c[11]*4.718637772708116;
    nuc += c[13]*13.89129347445998;
    nuc += c[15]*22.18705464592268;
    nuc += c[17]*28.63763513582592;
    nuc += c[19]*30.88705769902543;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+8, k+pw+1);
    }
    // l = 10, i = 47
    nuc = 0.0;
    nuc += c[2]*-80.16867831174027;
    nuc += c[4]*-95.25458382486842;
    nuc += c[6]*-78.58102252194469;
    nuc += c[8]*-43.58890861250187;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+7, k+pw+2);
    }
    // l = 10, i = 48
    nuc = 0.0;
    nuc += c[11]*-62.91517030277488;
    nuc += c[13]*-166.6955216935197;
    nuc += c[15]*-207.0791766952783;
    nuc += c[17]*-152.7340540577382;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+6, k+pw+3);
    }
    // l = 10, i = 49
    nuc = 0.0;
    nuc += c[4]*222.2606955913597;
    nuc += c[6]*314.3240900877788;
    nuc += c[8]*217.9445430625093;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+5, k+pw+4);
    }
    // l = 10, i = 50
    nuc = 0.0;
    nuc += c[11]*150.9964087266597;
    nuc += c[13]*333.3910433870395;
    nuc += c[15]*248.495012034334;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+4, k+pw+5);
    }
    // l = 10, i = 51
    nuc = 0.0;
    nuc += c[6]*-209.5493933918525;
    nuc += c[8]*-232.4741792666766;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+3, k+pw+6);
    }
    // l = 10, i = 52
    nuc = 0.0;
    nuc += c[11]*-86.28366212951984;
    nuc += c[13]*-127.0061117664912;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+2, k+pw+7);
    }
    // l = 10, i = 53
    nuc = 0.0;
    nuc += c[8]*49.815895557145;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+1, k+pw+8);
    }
    // l = 10, i = 54
    nuc = 0.0;
    nuc += c[11]*9.587073569946648;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+1, j+pv+0, k+pw+9);
    }
    // l = 10, i = 55
    nuc = 0.0;
    nuc += c[10]*-0.3181304937373671;
    nuc += c[12]*-0.4540511313802278;
    nuc += c[14]*-0.4677441816782422;
    nuc += c[16]*-0.4961176240878564;
    nuc += c[18]*-0.5567269327204184;
    nuc += c[20]*-0.7673951182219901;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+10, k+pw+0);
    }
    // l = 10, i = 56
    nuc = 0.0;
    nuc += c[1]*3.431895299891715;
    nuc += c[3]*4.091090733689417;
    nuc += c[5]*4.437410929184535;
    nuc += c[7]*4.630431158153326;
    nuc += c[9]*4.718637772708116;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+9, k+pw+1);
    }
    // l = 10, i = 57
    nuc = 0.0;
    nuc += c[10]*15.90652468686835;
    nuc += c[12]*21.79445430625093;
    nuc += c[14]*19.64525563048617;
    nuc += c[16]*15.8757639708114;
    nuc += c[18]*10.02108478896753;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+8, k+pw+2);
    }
    // l = 10, i = 58
    nuc = 0.0;
    nuc += c[3]*-21.81915057967689;
    nuc += c[5]*-41.41583533905566;
    nuc += c[7]*-55.56517389783991;
    nuc += c[9]*-62.91517030277488;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+7, k+pw+3);
    }
    // l = 10, i = 59
    nuc = 0.0;
    nuc += c[10]*-84.83479832996456;
    nuc += c[12]*-108.9722715312547;
    nuc += c[14]*-78.58102252194469;
    nuc += c[16]*-37.04344926522661;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+6, k+pw+4);
    }
    // l = 10, i = 60
    nuc = 0.0;
    nuc += c[5]*49.6990024068668;
    nuc += c[7]*111.1303477956798;
    nuc += c[9]*150.9964087266597;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+5, k+pw+5);
    }
    // l = 10, i = 61
    nuc = 0.0;
    nuc += c[10]*101.8017579959575;
    nuc += c[12]*116.2370896333383;
    nuc += c[14]*52.38734834796313;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+4, k+pw+6);
    }
    // l = 10, i = 62
    nuc = 0.0;
    nuc += c[7]*-42.33537058883041;
    nuc += c[9]*-86.28366212951984;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+3, k+pw+7);
    }
    // l = 10, i = 63
    nuc = 0.0;
    nuc += c[10]*-29.08621657027356;
    nuc += c[12]*-24.9079477785725;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+2, k+pw+8);
    }
    // l = 10, i = 64
    nuc = 0.0;
    nuc += c[9]*9.587073569946648;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+1, k+pw+9);
    }
    // l = 10, i = 65
    nuc = 0.0;
    nuc += c[10]*1.292720736456603;

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+0, j+pv+0, k+pw+10);
    }

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] *= 4.0 * M_PI;
    cart2sph(omega, lc, buf);
}