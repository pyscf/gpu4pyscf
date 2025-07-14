/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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

#include "gvhf-rys/gamma_inc.cu"

__device__
static void boys_fn(double *out, double theta, double rr, double omega,
                    double fac, int order, int thread_id, int threads)
{
    double theta_rr = theta * rr;
    if (omega == 0) {
        eval_gamma_inc_fn(out, theta_rr, order, thread_id, threads);
    } else if (omega > 0) {
        double theta_fac = omega * omega / (omega * omega + theta);
        eval_gamma_inc_fn(out, theta_fac*theta_rr, order, thread_id, threads);
        double scale = sqrt(theta_fac);
        for (int n = 0 ; n <= order; ++n) {
            out[thread_id+n*threads] *= scale;
            scale *= theta_fac;
        }
    } else { // omega < 0
        eval_gamma_inc_fn(out, theta_rr, order, thread_id, threads);
        double theta_fac = omega * omega / (omega * omega + theta);
        double *out1 = out + threads * (order+1);
        eval_gamma_inc_fn(out1, theta_fac*theta_rr, order, thread_id, threads);
        double scale = sqrt(theta_fac);
        for (int n = 0 ; n <= order; ++n) {
            out[thread_id+n*threads] -= scale * out1[thread_id+n*threads];
            scale *= theta_fac;
        }
    }
    out[thread_id] *= fac;
    double a2 = -2. * theta;
    for (int n = 1; n <= order; n++) {
        fac *= a2;
        out[thread_id+n*threads] *= fac;
    }
}
