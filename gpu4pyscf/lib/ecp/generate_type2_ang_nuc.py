# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code generator for type2_ang_nuc.cu
"""

from pyscf.gto.mole import cart2sph

cart_pow_y = [
        0,
        1, 0,
        2, 1, 0,
        3, 2, 1, 0,
        4, 3, 2, 1, 0,
        5, 4, 3, 2, 1, 0,
        6, 5, 4, 3, 2, 1, 0,
        7, 6, 5, 4, 3, 2, 1, 0,
        8, 7, 6, 5, 4, 3, 2, 1, 0,
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        10,9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

cart_pow_z = [
        0,
        0, 1,
        0, 1, 2,
        0, 1, 2, 3,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5, 6,
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7, 8,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def calculate_c(l):
    c2s = cart2sph(l)
    m, n = c2s.shape

    g = []
    for li in reversed(range(l+1)):
        for lj in reversed(range(l-li+1)):
            lk = l - li - lj
            g.append(f'rx[{li}]*ry[{lj}]*rz[{lk}]')

    c_scripts = []
    for j in range(n):
        for i in range(m):
            if abs(c2s[i,j]) > 1e-16:
                c_scripts.append(f'    c[{j}] += {c2s[i,j]}*({g[i]});')
    return '\n'.join(c_scripts)

def calculate_cart(l):
    c2s = cart2sph(l)
    m, n = c2s.shape
    cart_scripts = []
    for i in range(m):
        nuc = [
            f"""    // l = {l}, i = {i}""",
            """    nuc = 0.0;"""]
        for j in range(n):
            if abs(c2s[i,j]) > 1e-16:
                nuc.append(f"""    nuc += c[{j}]*{c2s[i,j]};""")
        cart_scripts.append('\n'.join(nuc))
    return cart_scripts

def calculate_nuc(l):
    cart_scripts = calculate_cart(l)
    nuc_scripts = []

    for n in range((l+1)*(l+2)//2):
        ps = cart_pow_y[n]
        pt = cart_pow_z[n]
        pr = l - ps - pt
        nuc_scripts.append(cart_scripts[n])

        loop = f"""
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++){{
        const int pv = _cart_pow_y[m];
        const int pw = _cart_pow_z[m];
        const int pu = lc - pv - pw;
        buf[m] += nuc * int_unit_xyz(i+pu+{pr}, j+pv+{ps}, k+pw+{pt});
    }}"""
        nuc_scripts.append(loop)
    return '\n'.join(nuc_scripts)

header = """/*
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
"""

from jinja2 import Template
template_string = """
template <> __device__
void type2_ang_nuc_l<{{ l }}>(double * __restrict__ omega, const int lc,
                    const int i, const int j, const int k,
                    double * __restrict__ unitr){
    constexpr int l = {{ l }};
    double rx[l+1], ry[l+1], rz[l+1];
    rx[0] = ry[0] = rz[0] = 1.0;
    for (int li = 1; li <= l; li++) {
        rx[li] = rx[li - 1] * unitr[0];
        ry[li] = ry[li - 1] * unitr[1];
        rz[li] = rz[li - 1] * unitr[2];
    }

    double c[2*l+1];
    for (int m = 0; m < 2*l+1; m++) c[m] = 0.0;
{{ c_scripts }};

    double buf[(ECP_LMAX+1)*(ECP_LMAX+2)/2];
    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] = 0.0;
    double nuc;

{{ nuc_scripts }}

    for (int m = 0; m < (lc+1)*(lc+2)/2; m++) buf[m] *= 4.0 * M_PI;
    cart2sph(omega, lc, buf);
}
"""

template = Template(template_string)

with open('type2_ang_nuc.cu', 'w') as f:
    f.write(header)
    for l in range(0, 11):
        c_scripts = calculate_c(l)
        nuc_scripts = calculate_nuc(l)
        redered = template.render(
            l=l,
            c_scripts=c_scripts,
            nuc_scripts=nuc_scripts)
        f.write(redered)
