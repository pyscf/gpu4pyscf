/* Copyright 2024 The GPU4PySCF Authors. All Rights Reserved.
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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include "libxc.h"

#define THREADS 128

// Up to order = 2, do_exc = True, do_vxc = True, do_fxc = True, do_kxc = False, do_lxc = False
#define ADD_LDA if(out->zk     != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->zk, out_lda->zk, coef, np, dim->zk); \
                if(out->vrho   != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->vrho, out_lda->vrho, coef, np, dim->vrho); \
                if(out->v2rho2 != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2rho2, out_lda->v2rho2, coef, np, dim->v2rho2);

#define ADD_GGA if(out->zk     != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->zk, out_gga->zk, coef, np, dim->zk); \
                if(out->vrho   != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->vrho, out_gga->vrho, coef, np, dim->vrho); \
                if(out->vrho   != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->vsigma, out_gga->vsigma, coef, np, dim->vsigma); \
                if(out->v2rho2 != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2rho2, out_gga->v2rho2, coef, np, dim->v2rho2); \
                if(out->v2rho2 != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2rhosigma, out_gga->v2rhosigma, coef, np, dim->v2rhosigma); \
                if(out->v2rho2 != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2sigma2, out_gga->v2sigma2, coef, np, dim->v2sigma2);

#define ADD_MGGA if(out->zk     != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->zk, out_mgga->zk, coef, np, dim->zk); \
                 if(out->vrho   != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->vrho, out_mgga->vrho, coef, np, dim->vrho); \
                 if(out->vrho   != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->vsigma, out_mgga->vsigma, coef, np, dim->vsigma); \
                 if(out->vrho   != NULL && out->vlapl != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->vlapl, out_mgga->vlapl, coef, np, dim->vlapl); \
                 if(out->vrho   != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->vtau, out_mgga->vtau, coef, np, dim->vtau); \
                 if(out->v2rho2 != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2rho2, out_mgga->v2rho2, coef, np, dim->v2rho2); \
                 if(out->v2rho2 != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2rhosigma, out_mgga->v2rhosigma, coef, np, dim->v2rhosigma); \
                 if(out->v2rho2 != NULL && out->v2rholapl != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2rholapl, out_mgga->v2rholapl, coef, np, dim->v2rholapl); \
                 if(out->v2rho2 != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2rhotau, out_mgga->v2rhotau, coef, np, dim->v2rhotau); \
                 if(out->v2rho2 != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2sigma2, out_mgga->v2sigma2, coef, np, dim->v2sigma2);\
                 if(out->v2rho2 != NULL && out->v2sigmalapl != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2sigmalapl, out_mgga->v2sigmalapl, coef, np, dim->v2sigmalapl);\
                 if(out->v2rho2 != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2sigmatau, out_mgga->v2sigmatau, coef, np, dim->v2sigmatau);\
                 if(out->v2rho2 != NULL && out->v2lapl2 != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2lapl2, out_mgga->v2lapl2, coef, np, dim->v2lapl2);\
                 if(out->v2rho2 != NULL && out->v2lapltau != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2lapltau, out_mgga->v2lapltau, coef, np, dim->v2lapltau);\
                 if(out->v2rho2 != NULL) _add_out<<<blocks, threads, 0, stream>>>(out->v2tau2, out_mgga->v2tau2, coef, np, dim->v2tau2);

__global__
static void _add_out(double *out, const double *buf, double coef, int np, int dim){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < np) {
        #pragma unroll
        for (int j = 0; j < dim; j++){
            int idx = i + j * np;
            out[idx] += coef * buf[idx];
        }
    }
}

extern "C" {

__host__
void _memset_lda(xc_lda_out_params *out, int order, int np, const xc_dimensions *dim){
    if(order >= 0) cudaMemset(out->zk, 0, sizeof(double)*np*dim->zk);
    if(order >= 1) cudaMemset(out->vrho, 0, sizeof(double)*np*dim->vrho);
    if(order >= 2) cudaMemset(out->v2rho2, 0, sizeof(double)*np*dim->v2rho2);
}

__host__
void _memset_gga(xc_gga_out_params *out, int order, int np, const xc_dimensions *dim){
    if(order >= 0) cudaMemset(out->zk, 0, sizeof(double)*np*dim->zk);
    if(order >= 1) cudaMemset(out->vrho, 0, sizeof(double)*np*dim->vrho);
    if(order >= 1) cudaMemset(out->vsigma, 0, sizeof(double)*np*dim->vsigma); // (sigma, lapl, tau)
    if(order >= 2) cudaMemset(out->v2rho2, 0, sizeof(double)*np*dim->v2rho2);
    if(order >= 2) cudaMemset(out->v2rhosigma, 0, sizeof(double)*np*dim->v2rhosigma);
    if(order >= 2) cudaMemset(out->v2sigma2, 0, sizeof(double)*np*dim->v2sigma2);
}

__host__
void _memset_mgga(xc_mgga_out_params *out, int order, int np, const xc_dimensions *dim){
    if(order >= 0) cudaMemset(out->zk, 0, sizeof(double)*np*dim->zk);

    if(order >= 1) cudaMemset(out->vrho, 0, sizeof(double)*np*dim->vrho);
    if(order >= 1) cudaMemset(out->vsigma, 0, sizeof(double)*np*dim->vsigma);
    if(order >= 1 && out->vlapl != NULL) cudaMemset(out->vlapl, 0, sizeof(double)*np*dim->vlapl); // (sigma, lapl, tau)
    if(order >= 1) cudaMemset(out->vtau, 0, sizeof(double)*np*dim->vtau);

    if(order >= 2) cudaMemset(out->v2rho2, 0, sizeof(double)*np*dim->v2rho2);
    if(order >= 2) cudaMemset(out->v2rhosigma, 0, sizeof(double)*np*dim->v2rhosigma);
    if(order >= 2 && out->v2rholapl != NULL) cudaMemset(out->v2rholapl, 0, sizeof(double)*np*dim->v2rholapl);
    if(order >= 2) cudaMemset(out->v2rhotau, 0, sizeof(double)*np*dim->v2rhotau);
    if(order >= 2) cudaMemset(out->v2sigma2, 0, sizeof(double)*np*dim->v2sigma2);
    if(order >= 2 && out->v2sigmalapl != NULL) cudaMemset(out->v2sigmalapl, 0, sizeof(double)*np*dim->v2sigmalapl);
    if(order >= 2) cudaMemset(out->v2sigmatau, 0, sizeof(double)*np*dim->v2sigmatau);
    if(order >= 2 && out->v2lapl2 != NULL) cudaMemset(out->v2lapl2, 0, sizeof(double)*np*dim->v2lapl2);
    if(order >= 2 && out->v2lapltau != NULL) cudaMemset(out->v2lapltau, 0, sizeof(double)*np*dim->v2lapltau);
    if(order >= 2) cudaMemset(out->v2tau2, 0, sizeof(double)*np*dim->v2tau2);
}

__host__
int _xc_lda(const xc_func_type *func, int np, int order, const double *rho,
            xc_lda_out_params *out){

    if(func->info->lda == NULL){
        fprintf(stderr, "Nested xc functional is not supported\n");
        return 1;
    }
    const xc_dimensions *dim = &(func->dim);

    if(order < 0) return 0;
    _memset_lda(out, order, np, dim);

    if(func->info->lda != NULL){
        if(func->nspin == XC_UNPOLARIZED){
            if(func->info->lda->unpol[order] != NULL)
                func->info->lda->unpol[order](func, np, rho, out);
        }else{
            if(func->info->lda->pol[order] != NULL)
                func->info->lda->pol[order](func, np, rho, out);
        }
    }
    return 0;
}

__host__
int _xc_gga(const xc_func_type *func, int np, int order, const double *rho, const double *sigma,
            xc_gga_out_params *out){

    if(func->info->gga == NULL){
        fprintf(stderr, "Nested xc functional is not supported\n");
        return 1;
    }
    const xc_dimensions *dim = &(func->dim);

    if(order < 0) return 0;
    _memset_gga(out, order, np, dim);

    /* call the GGA routines */
    if(func->info->gga != NULL){
        if(func->nspin == XC_UNPOLARIZED){
            if(func->info->gga->unpol[order] != NULL)
                func->info->gga->unpol[order](func, np, rho, sigma, out);
        }else{
            if(func->info->gga->pol[order] != NULL)
                func->info->gga->pol[order](func, np, rho, sigma, out);
        }
    }
    return 0;
}

__host__
int _xc_mgga(const xc_func_type *func, int np, int order, const double *rho, const double *sigma,
            const double *lapl, const double *tau,
            xc_mgga_out_params *out){

    if(func->info->mgga == NULL){
        fprintf(stderr, "Nested xc functional is not supported\n");
        return 1;
    }
    const xc_dimensions *dim = &(func->dim);

    if(order < 0) return 0;
    _memset_mgga(out, order, np, dim);

    /* call the mGGA routines */
    if(func->info->mgga != NULL){
        if(func->nspin == XC_UNPOLARIZED){
            if(func->info->mgga->unpol[order] != NULL)
                func->info->mgga->unpol[order](func, np, rho, sigma, lapl, tau, out);
        }else{
            if(func->info->mgga->pol[order] != NULL)
                func->info->mgga->pol[order](func, np, rho, sigma, lapl, tau, out);
        }
    }
    return 0;
}

__host__
int xc_lda(cudaStream_t stream,
    const xc_func_type *func, int np, const double *rho,
    xc_lda_out_params *out, xc_lda_out_params *buf)
{
    int ierr = 0;

    int order = -1;
    if(out->zk     != NULL) order = 0;
    if(out->vrho   != NULL) order = 1;
    if(out->v2rho2 != NULL) order = 2;
    if(out->v3rho3 != NULL) order = 3;
    if(out->v4rho4 != NULL) order = 4;

    // If the functional is not a mix
    if(func->info->lda != NULL){
        ierr = _xc_lda(func, np, order, rho, out);
        return ierr;
    }

    // If the functional is a mix of multiple functionals (more common, such as B3LYP)
    if(func->mix_coef == NULL){
        return ierr;
    }

    const xc_dimensions *dim = &(func->dim);
    _memset_lda(out, order, np, dim);
    dim3 threads(THREADS);
    dim3 blocks((np+THREADS-1)/THREADS);
    for (int ii=0; ii< func->n_func_aux; ii++){
        xc_func_type *aux = func->func_aux[ii];
        double coef = func->mix_coef[ii];
        /* Evaluate the functional */
        switch(aux->info->family){
            case XC_FAMILY_LDA:{
                xc_lda_out_params *out_lda = (xc_lda_out_params *)(buf);
                ierr = _xc_lda(aux, np, order, rho, out_lda);
                ADD_LDA;
                break;
            }
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of xc lda: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return ierr;
}

__host__
int xc_gga(cudaStream_t stream,
    const xc_func_type *func, int np, const double *rho, const double *sigma,
    xc_gga_out_params *out, xc_gga_out_params *buf)
{
    int order = -1;
    if(out->zk     != NULL) order = 0;
    if(out->vrho   != NULL) order = 1;
    if(out->v2rho2 != NULL) order = 2;
    if(out->v3rho3 != NULL) order = 3;
    if(out->v4rho4 != NULL) order = 4;

    // If the functional is not a mix
    int ierr = 0;
    if(func->info->gga != NULL){
        ierr = _xc_gga(func, np, order, rho, sigma, out);
        return ierr;
    }

    // If the functional is a mix of multiple functionals (more common, such as B3LYP)
    if(func->mix_coef == NULL){
        return ierr;
    }
    //const xc_dimensions *dim = &(func->dim);
    _memset_gga(out, order, np, &(func->dim));
    //printf("%d %d %d\n", func->dim->rho, func->dim->vrho, func->dim->vsigma);

    dim3 threads(THREADS);
    dim3 blocks((np+THREADS-1)/THREADS);
    for (int ii=0; ii< func->n_func_aux; ii++){
        xc_func_type *aux = func->func_aux[ii];
        double coef = func->mix_coef[ii];
        const xc_dimensions *dim = &(aux->dim);

        /* Evaluate the functional */
        switch(aux->info->family){
            case XC_FAMILY_LDA:{
                xc_lda_out_params *out_lda = (xc_lda_out_params *)(buf);
                ierr = _xc_lda(aux, np, order, rho, out_lda);
                ADD_LDA;
                break;
            }
            case XC_FAMILY_GGA:{
                xc_gga_out_params *out_gga = (xc_gga_out_params *)(buf);
                ierr = _xc_gga(aux, np, order, rho, sigma, out_gga);
                ADD_GGA;
                break;
            }
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of xc_gga: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return ierr;
}

__host__
int xc_mgga(cudaStream_t stream,
        const xc_func_type *func, int np,
        const double *rho, const double *sigma, const double *lapl, const double *tau,
        xc_mgga_out_params *out, xc_mgga_out_params *buf)
{
    int order = -1;

    if(out->zk     != NULL) order = 0;
    if(out->vrho   != NULL) order = 1;
    if(out->v2rho2 != NULL) order = 2;
    if(out->v3rho3 != NULL) order = 3;
    if(out->v4rho4 != NULL) order = 4;

    int ierr = 0;
    // If the functional is not a mix
    if(func->info->mgga != NULL){
        ierr = _xc_mgga(func, np, order, rho, sigma, lapl, tau, out);
        return ierr;
    }

    // If the functional is a mix of multiple functionals (more common, such as B3LYP)
    if(func->mix_coef == NULL){
        return ierr;
    }

    const xc_dimensions *dim = &(func->dim);
    _memset_mgga(out, order, np, dim);
    dim3 threads(THREADS);
    dim3 blocks((np+THREADS-1)/THREADS);
    for (int ii=0; ii< func->n_func_aux; ii++){
        xc_func_type *aux = func->func_aux[ii];
        double coef = func->mix_coef[ii];
        /* Evaluate the functional */
        switch(aux->info->family){
            case XC_FAMILY_LDA:{
                xc_lda_out_params *out_lda = (xc_lda_out_params *)(buf);
                ierr = _xc_lda(aux, np, order, rho, out_lda);
                ADD_LDA;
                break;
            }
            case XC_FAMILY_GGA:{
                xc_gga_out_params *out_gga = (xc_gga_out_params *)(buf);
                ierr = _xc_gga(aux, np, order, rho, sigma, out_gga);
                ADD_GGA;
                break;
            }
            case XC_FAMILY_MGGA:{
                xc_mgga_out_params *out_mgga = (xc_mgga_out_params *)(buf);
                ierr = _xc_mgga(aux, np, order, rho, sigma, lapl, tau, out_mgga);
                ADD_MGGA;
                break;
            }
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of xc mgga: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
