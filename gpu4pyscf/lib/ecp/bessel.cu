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

__constant__
static double _factorial[] = {
    1.0, 1.0, 2.0, 6.0, 24.,
    1.2e+2, 7.2e+2, 5.04e+3, 4.032e+4, 3.6288e+5,
    3.6288e+6, 3.99168e+7, 4.790016e+8, 6.2270208e+9, 8.71782912e+10,
    1.307674368e+12, 2.0922789888e+13, 3.55687428096e+14,
    6.402373705728e+15, 1.21645100408832e+17,
    2.43290200817664e+18, 5.109094217170944e+19,
    1.1240007277776077e+21, 2.5852016738884978e+22,
};

// ijk+1 < LI+LC (<=10) + LI(<=6) + LC (<=4) + 1 <= 21
__constant__
static double _factorial2[] = {
    1., 1., 2., 3., 8.,
    15., 48., 105., 384., 945.,
    3840., 10395., 46080., 135135., 645120.,
    2027025., 10321920., 34459425., 185794560., 654729075.,
    3715891200., 13749310575., 81749606400., 316234143225., 1961990553600.,
    //7905853580625., 51011754393600., 213458046676875.,
    //1428329123020800., 6190283353629376.,
    //42849873690624000., 1.9189878396251069e+17,
    //1.371195958099968e+18, 6.3326598707628524e+18,
    //4.6620662575398912e+19, 2.2164309547669976e+20,
    //1.6783438527143608e+21, 8.2007945326378929e+21,
    //6.3777066403145712e+22, 3.1983098677287775e+23,
};

__device__ __forceinline__
static double factorial2(int n){
    return (n < 0) ? 1.0 : _factorial2[n];
}


__device__ __forceinline__
static double int_unit_xyz(const int i, const int j, const int k){
    // i % 2 and j % 2 and k % 2
    const int even = 1 - (((i & 1) | (j & 1)) | (k & 1));
    const double fi = factorial2(i-1);
    const double fj = factorial2(j-1);
    const double fk = factorial2(k-1);
    const int ijk = i + j + k;
    const double fijk = factorial2(ijk+1);
    return even * (fi * fj * fk) / fijk;
}

/*
 * exponentially scaled modified spherical Bessel function of the first kind
 * scipy.special.sph_in(order, z) * numpy.exp(-z)
 *
 * JCC, 27, 1009
 */
__device__
static void _ine(double *out, const int order, const double z)
{
    if (z < 1e-7) {
        // (1-z) * z^l / (2l+1)!!
        out[0] = 1. - z;
        for (int i = 1; i <= order; i++) {
            out[i] = out[i-1] * z / (i*2+1);
        }
    } else if (z > 16) {
        // R_l(z) = \sum_k (l+k)!/(k!(l-k)!(2x)^k)
        const double z2 = -.5 / z;
        for (int i = 0; i <= order; i++) {
            double ti = .5 / z;
            double s = ti;
            for (int k = 1; k <= i; k++) {
                ti *= z2;
                s += ti * _factorial[i+k] / (_factorial[k] * _factorial[i-k]);
            }
            out[i] = s;
        }
    } else {
        // z^l e^{-z} \sum (z^2/2)^k/(k!(2k+2l+1)!!)
        const double z2 = .5 * z * z;
        double t0 = exp(-z);
        for (int i = 0; i <= order; i++) {
            double ti = t0;
            double s = ti;
            for (int k = 1;; k++) {
                ti *= z2 / (k * (k*2+i*2+1));
                double next = s + ti;
                if (next == s) {
                    break;
                } else {
                    s = next;
                }
            }
            t0 *= z/(i*2+3);  // k = 0
            out[i] = s;
        }
    }
}

