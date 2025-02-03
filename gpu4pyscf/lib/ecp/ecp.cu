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

#include "ecp.h"
#include "bessel.cu"

__device__
static double r99[] = {
    7.49149775547408580678e-09,2.39390171574704879731e-07,1.81360303841415770876e-06,
    7.61741913646307722274e-06,2.31485030718348028245e-05,5.73044613240147882038e-05,
    1.23105600339346032968e-04,2.38337096238994128328e-04,4.26098829770071851897e-04,
    7.15253566223816861225e-04,1.14076690801112601292e-03,1.74393542073036922346e-03,
    2.57250241525242007157e-03,3.68066393021282411979e-03,5.12897037092585605933e-03,
    6.98413189926139210684e-03,9.31873792827664360061e-03,1.22109028731325341965e-02,
    1.57438515926779931675e-02,2.00054587018748675220e-02,2.50877561523923375830e-02,
    3.10864232033296605806e-02,3.81002721950566280995e-02,4.62307424720755921754e-02,
    5.55814134610747023757e-02,6.62575463867129954565e-02,7.83656624864419448784e-02,
    9.20131639458153793854e-02,1.07308002185481865531e-01,1.24358396645808944037e-01,
    1.43272605876474390385e-01,1.64158751574681183172e-01,1.87124695242669281114e-01,
    2.12277966358133562963e-01,2.39725740366322215280e-01,2.69574864399791169767e-01,
    3.01931928396128146375e-01,3.36903379197803665157e-01,3.74595675262858196497e-01,
    4.15115479771713591362e-01,4.58569890166489968486e-01,5.05066702489160856970e-01,
    5.54714709280482720644e-01,6.07624030252335245450e-01,6.63906475444308141753e-01,
    7.23675941116369569883e-01,7.87048839211608508570e-01,8.54144561847640537700e-01,
    9.25085982966693309848e-01,1.00000000000000000000e+00,1.07901811919271173323e+00,
    1.16227708910521165819e+00,1.24991958777366174438e+00,1.34209497009947353874e+00,
    1.43896008327213209554e+00,1.54068015944685710039e+00,1.64742979654017807079e+00,
    1.75939403992477760852e+00,1.87676958006419325464e+00,1.99976608380819209643e+00,
    2.12860768027057822849e+00,2.26353462605831046162e+00,2.40480517927247383625e+00,
    2.55269771735934725143e+00,2.70751314081141725154e+00,2.86957761323600735182e+00,
    3.03924569885439943562e+00,3.21690397163072372422e+00,3.40297518669977883121e+00,
    3.59792312555240823002e+00,3.80225825286600205288e+00,4.01654435671977161348e+00,
    4.24140638764222543955e+00,4.47753976885303650590e+00,4.72572152485034457925e+00,
    4.98682367473171250793e+00,5.26182946969805254156e+00,5.55185323462325097665e+00,
    5.85816482123891901779e+00,6.18222002494881728296e+00,6.52569880330852480910e+00,
    6.89055383078500405247e+00,7.27907294007241034706e+00,7.69396050945778942065e+00,
    8.13844514586779865795e+00,8.61642457049883248033e+00,9.13266428605907520932e+00,
    9.69307592048363630965e+00,1.03051169664600053011e+01,1.09783815787195599967e+01,
    1.17255037073349104304e+01,1.25635944005437458770e+01,1.35166439102318687304e+01,
    1.46197880380327820404e+01,1.59274995188788217604e+01,1.75310364710511947806e+01,
    1.96014771359075226087e+01,2.25228992353034804808e+01,2.75208650628360445012e+01
};

__device__
static double w99[] = {
    3.74504465546597569538e-08,5.98025425874844469710e-07,3.01755904037220807623e-06,
    9.49315435270622856210e-06,2.30398689788116110507e-05,4.74313676460523959283e-05,
    8.71259201682535134800e-05,1.47179566351102945714e-04,2.33148687103363678368e-04,
    3.50984582880628406387e-04,5.06922938043571566502e-04,7.07371219265206943941e-04,
    9.58797101661354954881e-04,1.26762093081784132584e-03,1.64011501514828858875e-03,
    2.08231221338544105273e-03,2.59992585711949202726e-03,3.19828255500335511402e-03,
    3.88226889403487296012e-03,4.65629251553667869445e-03,5.52425752862067010601e-03,
    6.48955375754613900119e-03,7.55505892130658475436e-03,8.72315252730691305383e-03,
    9.99574003267965187358e-03,1.13742856869007685078e-02,1.28598524127975146619e-02,
    1.44531471005464283441e-02,1.61545697688108974566e-02,1.79642651752900291140e-02,
    1.98821756219053555337e-02,2.19080938846085149230e-02,2.40417153927295589033e-02,
    2.62826889781383951639e-02,2.86306657025656371984e-02,3.10853454466156094160e-02,
    3.36465211026480970347e-02,3.63141203538897561209e-02,3.90882451433852487477e-02,
    4.19692090394069017290e-02,4.49575727902660482460e-02,4.80541784332910032473e-02,
    5.12601823826394700778e-02,5.45770879714215714773e-02,5.80067779682611125991e-02,
    6.15515476299037800345e-02,6.52141388927818116406e-02,6.89977763505640606656e-02,
    7.29062057147122910550e-02,7.69437355140780432361e-02,8.11152828609839615659e-02,
    8.54264241987284650426e-02,8.98834520532275083049e-02,9.44934389444379158052e-02,
    9.92643097771207100211e-02,1.04204924232482093460e-01,1.09325170931016332765e-01,
    1.14636075443408688712e-01,1.20149924604410915374e-01,1.25880410051722568809e-01,
    1.31842794490549808373e-01,1.38054104903617747002e-01,1.44533357823557057076e-01,
    1.51301822908832206416e-01,1.58383332480235811124e-01,1.65804646467863830983e-01,
    1.73595884502318081877e-01,1.81791039811778243340e-01,1.90428593366022247402e-01,
    1.99552251623036053241e-01,2.09211837674749323579e-01,2.19464374100552722657e-01,
    2.30375407187545538923e-01,2.42020637456367176954e-01,2.54487942212678397436e-01,
    2.67879904418947323297e-01,2.82317001937544220791e-01,2.97941667216647110283e-01,
    3.14923507520003143068e-01,3.33466091844559053836e-01,3.53815881680646926455e-01,
    3.76274139285664444010e-01,4.01213039460866560670e-01,4.29097823819136747758e-01,
    4.60517817328173983960e-01,4.96230738223049638869e-01,5.37227459212045510561e-01,
    5.84829149853474561382e-01,6.40837406259777031536e-01,7.07774459213704520977e-01,
    7.89283521639834395600e-01,8.90829397324601646169e-01,1.02099961208170930682e+00,
    1.19410667684690441348e+00,1.43591306733334556078e+00,1.79794367421340983704e+00,
    2.40042315475528500457e+00,3.60402534195157464580e+00,7.21211908055371253568e+00
};

#define NGAUSS  99

__device__
static int _cart_pow_y[] = {
        0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1,
        0, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0, 8, 7, 6, 5,
        4, 3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,10, 9, 8, 7, 6,
        5, 4, 3, 2, 1, 0,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,12,11,
       10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,13,12,11,10, 9, 8, 7, 6, 5,
        4, 3, 2, 1, 0,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
};

__device__
static int _cart_pow_z[] = {
        0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
        5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3,
        4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 0, 1,
        2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 0, 1, 2, 3, 4, 5, 6, 7, 8,
        9,10,11,12,13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
};

template<int L> __device__
static void ang_nuc_part(double *omega, double rx, double ry, double rz){
    if (L == 0){
        omega[0] = 0.282094791773878143;
    } else if (L == 1){
        omega[0] = 0.488602511902919921 * rx;
        omega[1] = 0.488602511902919921 * ry;
        omega[2] = 0.488602511902919921 * rz;
    } else if (L == 2){
        double g0 = rx * rx;
        double g1 = rx * ry;
        double g2 = rx * rz;
        double g3 = ry * ry;
        double g4 = ry * rz;
        double g5 = rz * rz;
        omega[0] = 1.092548430592079070 * g1;
        omega[1] = 1.092548430592079070 * g4;
        omega[2] = 0.630783130505040012 * g5 - 0.315391565252520002 * (g0 + g3);
        omega[3] = 1.092548430592079070 * g2;
        omega[4] = 0.546274215296039535 * (g0 - g3);
    } else if (L == 3){
        double g0 = rx * rx * rx;
        double g1 = rx * rx * ry;
        double g2 = rx * rx * rz;
        double g3 = rx * ry * ry;
        double g4 = rx * ry * rz;
        double g5 = rx * rz * rz;
        double g6 = ry * ry * ry;
        double g7 = ry * ry * rz;
        double g8 = ry * rz * rz;
        double g9 = rz * rz * rz;
        omega[0] = 1.770130769779930531 * g1 - 0.590043589926643510 * g6;
        omega[1] = 2.890611442640554055 * g4;
        omega[2] = 1.828183197857862944 * g8 - 0.457045799464465739 * (g1 + g6);
        omega[3] = 0.746352665180230782 * g9 - 1.119528997770346170 * (g2 + g7);
        omega[4] = 1.828183197857862944 * g5 - 0.457045799464465739 * (g0 + g3);
        omega[5] = 1.445305721320277020 * (g2 - g7);
        omega[6] = 0.590043589926643510 * g0 - 1.770130769779930530 * g3;
    } else if (L == 4){
        double g0  = rx * rx * rx * rx;
        double g1  = rx * rx * rx * ry;
        double g2  = rx * rx * rx * rz;
        double g3  = rx * rx * ry * ry;
        double g4  = rx * rx * ry * rz;
        double g5  = rx * rx * rz * rz;
        double g6  = rx * ry * ry * ry;
        double g7  = rx * ry * ry * rz;
        double g8  = rx * ry * rz * rz;
        double g9  = rx * rz * rz * rz;
        double g10 = ry * ry * ry * ry;
        double g11 = ry * ry * ry * rz;
        double g12 = ry * ry * rz * rz;
        double g13 = ry * rz * rz * rz;
        double g14 = rz * rz * rz * rz;
        omega[0] = 2.503342941796704538 * (g1 - g6);
        omega[1] = 5.310392309339791593 * g4 - 1.770130769779930530 * g11;
        omega[2] = 5.677048174545360108 * g8 - 0.946174695757560014 * (g1 + g6);
        omega[3] = 2.676186174229156671 * g13- 2.007139630671867500 * (g4 + g11);
        omega[4] = 0.317356640745612911 * (g0 + g10) + 0.634713281491225822 * g3 - 2.538853125964903290 * (g5 + g12) + 0.846284375321634430 * g14;
        omega[5] = 2.676186174229156671 * g9 - 2.007139630671867500 * (g2 + g7);
        omega[6] = 2.838524087272680054 * (g5 - g12) + 0.473087347878780009 * (g10 - g0);
        omega[7] = 1.770130769779930531 * g2 - 5.310392309339791590 * g7 ;
        omega[8] = 0.625835735449176134 * (g0  + g10) - 3.755014412695056800 * g3;
    }
}

template <int L>
__global__
void _ang_nuc_part(double *omega, double *x, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n){
        return;
    }
    int offset = idx * (2*L+1);
    ang_nuc_part<L>(omega+offset, x[3*n], x[3*n+1], x[3*n+2]);
}

template <int LMAX> __device__
void type1_rad_part(double *rad_all, double k, double aij,
                    double *ur, int inc)
{
    constexpr int LMAX1 = LMAX + 1;
    double rur[LMAX1];
    double bval[NGAUSS*LMAX1];

    double kaij = k / (2*aij);
    double fac = kaij * kaij * aij;
    for (int n = 0; n < NGAUSS; n++){
        double tmp = r99[n*inc] - kaij;
        tmp = fac - aij*tmp*tmp;
        if (ur[n] == 0 || tmp > CUTOFF || tmp < -(EXPCUTOFF+6.+30.)) {
            rur[n] = 0;
            for (int i = 0; i < LMAX1; i++){
                bval[n*LMAX1 + i] = 0;
            }
        } else {
            rur[n] = ur[n] * exp(tmp);
            _ine(bval+n*LMAX1, LMAX, k*r99[n*inc]);
        }
    }

    for (int lab = 0; lab <= LMAX; lab++){
        if (lab > 0){
            for (int n = 0; n < NGAUSS; n++){
                rur[n] *= r99[n];
            }
        }
        double *prad = rad_all + lab * LMAX1;
        for (int i = lab%2; i <= LMAX; i+=2){
            double s = prad[i];
            for (int n = 0; n < NGAUSS; n++){
                s += rur[n] * bval[n*LMAX1+i];
            }
            prad[i] = s;
        }
    }
}

/*
template <int LMAX> __device__
void type1_rad_ang(double *rad_ang, double *r, double *rad_all)
{
    double unitr[3];
    if (r[0] == 0 && r[1] == 0 && r[2] == 0) {
        unitr[0] = 0;
        unitr[1] = 0;
        unitr[2] = 0;
    } else {
        double norm_r = -1/sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        unitr[0] = r[0] * norm_r;
        unitr[1] = r[1] * norm_r;
        unitr[2] = r[2] * norm_r;
    }

    double omega_nuc[CART_CUM];
    double *pnuc;
    for (int i = 0; i <= LMAX; i++) {
        pnuc = omega_nuc + _offset_cart[i];
        ang_nuc_in_cart(pnuc, i, unitr);
    }

    const int d1 = LMAX + 1;
    const int d2 = d1 * d1;
    const int d3 = d2 * d1;

    double *pout, *prad;
    for (int i = 0; i < d3; i++) { rad_ang[i] = 0; }
    for (int i = 0; i <= LMAX; i++) {
    for (int j = 0; j <= LMAX-i; j++) {
    for (int k = 0; k <= LMAX-i-j; k++) {
        pout = rad_ang + i*d2+j*d1+k;
        prad = rad_all + (i+j+k)*d1;
        // need_even to ensure (a+b+c+lmb) is even
        int need_even = (i+j+k)%2;
        for (int lmb = need_even; lmb <= LMAX; lmb+=2) {
            double tmp = 0;
            pnuc = omega_nuc + _offset_cart[lmb];
            for (int n = 0; n < (lmb+1)*(lmb+2)/2; n++){
                int ps = _cart_pow_y[n];
                int pt = _cart_pow_z[n];
                int pr = lmb - ps - pt;
                tmp += pnuc[n] * int_unit_xyz(i+pr, j+ps, k+pt);
            }
            *pout += prad[lmb] * tmp;
        }
    } } }
}
*/

/*
template <int LI, int LJ> __device__
static  void ECPtype1_cart(double *gctr, int *ecpbas, int necpbas,
                            int *atm, int natm,
                            int *bas, int nbas, double *env)
{
    if (necpbas == 0){
        return 0;
    }

    const int ish = blockIdx.x * blockDim.x + threadIdx.x;
    const int jsh = blockIdx.y * blockDim.y + threadIdx.y;
    const int npi = bas[NPRIM_OF+ish*BAS_SLOTS];
    const int npj = bas[NPRIM_OF+jsh*BAS_SLOTS];
    const int nfi = (LI+1) * (LI+2) / 2;
    const int nfj = (LI+1) * (LJ+2) / 2;
    const double *ai = env + bas[PTR_EXP+ish*BAS_SLOTS];
    const double *aj = env + bas[PTR_EXP+jsh*BAS_SLOTS];
    const double *ci = env + bas[PTR_COEFF+ish*BAS_SLOTS];
    const double *cj = env + bas[PTR_COEFF+jsh*BAS_SLOTS];
    const double *ri = env + atm[PTR_COORD+bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    const double *rj = env + atm[PTR_COORD+bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];

    for (int iloc = 0; iloc < nslots; iloc++){
        if (ecpbas[ANG_OF+ecploc[iloc]*BAS_SLOTS] != -1 || ecpbas[SO_TYPE_OF+ecploc[iloc]*BAS_SLOTS] == 1) {
            continue;
            }

        int atm_id = ecpbas[ATOM_OF+ecploc[iloc]*BAS_SLOTS];
        rc = env + atm[PTR_COORD+atm_id*ATM_SLOTS];
        ecpshls = ecploc + iloc;

        rca[0] = rc[0] - ri[0];
        rca[1] = rc[1] - ri[1];
        rca[2] = rc[2] - ri[2];
        rcb[0] = rc[0] - rj[0];
        rcb[1] = rc[1] - rj[1];
        rcb[2] = rc[2] - rj[2];
    }

}
*/

extern "C" {
int ECPsph_ine(double *out, int order, double *zs, int n)
{
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS);
    dim3 blocks(ntile);
    _ine_kernel<<<blocks, threads>>>(out, order, zs, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}

int ECPang_nuc_part(double *omega, double *x, int n, const int l){
    int ntile = (n + THREADS - 1) / THREADS;
    dim3 threads(THREADS);
    dim3 blocks(ntile);
    switch (l){
    case 0: _ang_nuc_part<0><<<blocks, threads>>>(omega, x, n); break;
    case 1: _ang_nuc_part<1><<<blocks, threads>>>(omega, x, n); break;
    case 2: _ang_nuc_part<2><<<blocks, threads>>>(omega, x, n); break;
    case 3: _ang_nuc_part<3><<<blocks, threads>>>(omega, x, n); break;
    case 4: _ang_nuc_part<4><<<blocks, threads>>>(omega, x, n); break;
    default:
        break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return 1;
    }
    return 0;
}
}
