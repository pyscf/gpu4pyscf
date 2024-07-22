
__constant__ BasisProdCache c_bpcache;
#define NG_PER_BLOCK       256
__global__
static void _eval_rho(BasOffsets offsets, double *dm_sparse, int npairs, 
    double *exp_sparse, double *coef_sparse, double *coord_pairs){
    int ngrids = offsets.ngrids;
    int grid_id = blockIdx.x * blockDim.x + threadIdx.x;

    double* __restrict__ rho = offsets.data;
    double* __restrict__ rhox = offsets.data + 1 * ngrids;
    double* __restrict__ rhoy = offsets.data + 2 * ngrids;
    double* __restrict__ rhoz = offsets.data + 3 * ngrids;
    
    double* __restrict__ xi = coord_pairs;
    double* __restrict__ yi = coord_pairs + npairs;
    double* __restrict__ zi = coord_pairs + 2*npairs;
    double* __restrict__ xj = coord_pairs + 3*npairs;
    double* __restrict__ yj = coord_pairs + 4*npairs;
    double* __restrict__ zj = coord_pairs + 5*npairs;

    double* __restrict__ ei = exp_sparse;
    double* __restrict__ ej = exp_sparse + npairs;
    
    double *gridx = offsets.gridx;
    double *gridy = offsets.gridx + ngrids;
    double *gridz = offsets.gridx + ngrids * 2;
    double xg = 0.0;
    double yg = 0.0;
    double zg = 0.0;

    if (grid_id < ngrids) {
        xg = gridx[grid_id];
        yg = gridy[grid_id];
        zg = gridz[grid_id];
    }

    double rho_i  = 0.0;
    double rhox_i = 0.0;
    double rhoy_i = 0.0;
    double rhoz_i = 0.0;
    
    double __shared__ dm_shared[NG_PER_BLOCK];
    double __shared__ xi_shared[NG_PER_BLOCK];
    double __shared__ yi_shared[NG_PER_BLOCK];
    double __shared__ zi_shared[NG_PER_BLOCK];

    double __shared__ xj_shared[NG_PER_BLOCK];
    double __shared__ yj_shared[NG_PER_BLOCK];
    double __shared__ zj_shared[NG_PER_BLOCK];

    double __shared__ ei_shared[NG_PER_BLOCK];
    double __shared__ ej_shared[NG_PER_BLOCK];
    double fac_ij = offsets.fac * offsets.fac;
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
            double rx = xg - xi_shared[task_id];
            double ry = yg - yi_shared[task_id];
            double rz = zg - zi_shared[task_id];
            double rri = rx * rx + ry * ry + rz * rz;
            double ei_loc = ei_shared[task_id];
            double erri = ei_loc * rri;
            double e_2a = -2.0 * ei_loc;

            double gtox = e_2a * rx;
            double gtoy = e_2a * ry;
            double gtoz = e_2a * rz;

            /* shell j*/
            rx = xg - xj_shared[task_id];
            ry = yg - yj_shared[task_id];
            rz = zg - zj_shared[task_id];
            double rrj = rx * rx + ry * ry + rz * rz;
            double ej_loc = ej_shared[task_id];
            double errj = ej_loc * rrj;
            e_2a = -2.0 * ej_loc;

            gtox += e_2a * rx;
            gtoy += e_2a * ry;
            gtoz += e_2a * rz;

            double eij = exp(-erri - errj);
            double dme_ij = dm_shared[task_id] * eij;
            
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
