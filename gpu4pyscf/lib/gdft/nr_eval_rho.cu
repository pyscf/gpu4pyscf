
__constant__ BasisProdCache c_bpcache;
#define NG_PER_BLOCK       256
template <typename T1, typename T2>
__global__
static void _eval_rho(BasOffsets offsets, T2 *dm_sparse, int npairs, 
    T2 *exp_sparse, T2 *coef_sparse, T2 *coord_pairs){
    int ngrids = offsets.ngrids;
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;

    double* __restrict__ rho = offsets.data;
    double* __restrict__ rhox = offsets.data + 1 * ngrids;
    double* __restrict__ rhoy = offsets.data + 2 * ngrids;
    double* __restrict__ rhoz = offsets.data + 3 * ngrids;
    
    T2* __restrict__ xi = coord_pairs;
    T2* __restrict__ yi = coord_pairs + npairs;
    T2* __restrict__ zi = coord_pairs + 2*npairs;
    T2* __restrict__ xj = coord_pairs + 3*npairs;
    T2* __restrict__ yj = coord_pairs + 4*npairs;
    T2* __restrict__ zj = coord_pairs + 5*npairs;

    T2* __restrict__ ei = exp_sparse;
    T2* __restrict__ ej = exp_sparse + npairs;
    
    double *gridx = offsets.gridx;
    double *gridy = offsets.gridx + ngrids;
    double *gridz = offsets.gridx + ngrids * 2;
    T2 xg = 0.0;
    T2 yg = 0.0;
    T2 zg = 0.0;

    if (grid_id < ngrids) {
        xg = gridx[grid_id];
        yg = gridy[grid_id];
        zg = gridz[grid_id];
    }

    T2 rho_i  = 0.0;
    T2 rhox_i = 0.0;
    T2 rhoy_i = 0.0;
    T2 rhoz_i = 0.0;
    
    T2 __shared__ dm_shared[NG_PER_BLOCK];
    T2 __shared__ xi_shared[NG_PER_BLOCK];
    T2 __shared__ yi_shared[NG_PER_BLOCK];
    T2 __shared__ zi_shared[NG_PER_BLOCK];

    T2 __shared__ xj_shared[NG_PER_BLOCK];
    T2 __shared__ yj_shared[NG_PER_BLOCK];
    T2 __shared__ zj_shared[NG_PER_BLOCK];

    T2 __shared__ ei_shared[NG_PER_BLOCK];
    T2 __shared__ ej_shared[NG_PER_BLOCK];
    T2 fac_ij = offsets.fac * offsets.fac;
    for (int block_id = 0; block_id < npairs; block_id+=NG_PER_BLOCK){
        int tx = threadIdx.x;
        int bas_ij = block_id + tx;

        if (bas_ij < npairs){
            dm_shared[tx] = dm_sparse[bas_ij] * fac_ij;
            xi_shared[tx] = xi[bas_ij];
            yi_shared[tx] = yi[bas_ij];
            zi_shared[tx] = zi[bas_ij];

            xj_shared[tx] = xj[bas_ij];
            yj_shared[tx] = yj[bas_ij];
            zj_shared[tx] = zj[bas_ij];

            ei_shared[tx] = ei[bas_ij];
            ej_shared[tx] = ej[bas_ij];
        }
        __syncthreads();
        for (int task_id = 0; task_id < NG_PER_BLOCK && task_id + block_id < npairs; ++task_id){
            /* shell i */
            T2 rx = xg - xi_shared[task_id];
            T2 ry = yg - yi_shared[task_id];
            T2 rz = zg - zi_shared[task_id];
            T2 rri = rx * rx + ry * ry + rz * rz;
            T2 ei_loc = ei_shared[task_id];
            T2 erri = ei_loc * rri;
            T2 e_2a = -2.0 * ei_loc;

            T2 gtox = e_2a * rx;
            T2 gtoy = e_2a * ry;
            T2 gtoz = e_2a * rz;

            /* shell j*/
            rx = xg - xj_shared[task_id];
            ry = yg - yj_shared[task_id];
            rz = zg - zj_shared[task_id];
            T2 rrj = rx * rx + ry * ry + rz * rz;
            T2 ej_loc = ej_shared[task_id];
            T2 errj = ej_loc * rrj;
            e_2a = -2.0 * ej_loc;

            gtox += e_2a * rx;
            gtoy += e_2a * ry;
            gtoz += e_2a * rz;

            T2 eij = exp(-erri - errj);
            T2 dme_ij = dm_shared[task_id] * eij;
            
            rho_i  += dme_ij;
            rhox_i += gtox * dme_ij;
            rhoy_i += gtoy * dme_ij;
            rhoz_i += gtoz * dme_ij;
        }
        __syncthreads();
    }
    
    // Due to the symmetry
    if (grid_id < ngrids){
        rho[grid_id] = rho_i * 2.0;
        rhox[grid_id] = rhox_i * 2.0;
        rhoy[grid_id] = rhoy_i * 2.0;
        rhoz[grid_id] = rhoz_i * 2.0;
    }
}
