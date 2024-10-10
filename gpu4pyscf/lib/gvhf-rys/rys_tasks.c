#include <omp.h>
#include "vhf.cuh"

#define AO_BLOCKSIZE    64
#define SHL_BLOCKSIZE   4

static int _create_tasks_1t(uint16_t *shl_quartets, double *vj, double *vk,
                            double *q_cond, double *dm_cond, double cutoff,
                            int *shls_slice, int *ao_loc, int max_tasks,
                            int *atm, int natm, int *bas, int nbas, double *env)
{
        // the first shell-quartet in the bin
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        int lsh1 = shls_slice[7];
        //int di = ao_loc[ish0+1] - ao_loc[ish0];
        //int dj = ao_loc[jsh0+1] - ao_loc[jsh0];
        //int dk = ao_loc[ksh0+1] - ao_loc[ksh0];
        //int dl = ao_loc[lsh0+1] - ao_loc[lsh0];
        int basblk_i = SHL_BLOCKSIZE; //MAX(AO_BLOCKSIZE/di, 4);
        int basblk_j = SHL_BLOCKSIZE; //MAX(AO_BLOCKSIZE/dj, 4);
        int basblk_k = SHL_BLOCKSIZE; //MAX(AO_BLOCKSIZE/dk, 4);
        int basblk_l = SHL_BLOCKSIZE; //MAX(AO_BLOCKSIZE/dl, 4);
        int _ish0, _jsh0, _ksh0, _lsh0, _ish1, _jsh1, _ksh1, _lsh1;

        int ntasks = 0;
        for (_ish0 = ish0; _ish0 < ish1; _ish0+=basblk_i) {
                _ish1 = MIN(_ish0+basblk_i, ish1);
                jsh1 = MIN(shls_slice[3], _ish1);
                ksh1 = MIN(shls_slice[5], _ish1);
        for (_jsh0 = jsh0; _jsh0 < jsh1; _jsh0+=basblk_j) {
                _jsh1 = MIN(_jsh0+basblk_j, jsh1);
        for (_ksh0 = ksh0; _ksh0 < ksh1; _ksh0+=basblk_k) {
                _ksh1 = MIN(_ksh0+basblk_k, ksh1);
                lsh1 = MIN(shls_slice[7], _ksh1);
        for (_lsh0 = lsh0; _lsh0 < lsh1; _lsh0+=basblk_l) {
                _lsh1 = MIN(_lsh0+basblk_l, lsh1);
                for (int ish = _ish0; ish < _ish1; ish++) {
                for (int jsh = _jsh0; jsh < MIN(ish+1, _jsh1); jsh++) {
                        int bas_ij = ish * nbas + jsh;
                        double q_ij = q_cond[bas_ij];
                        for (int ksh = _ksh0; ksh < MIN(ish+1, _ksh1); ksh++) {
                        for (int lsh = _lsh0; lsh < MIN(ksh+1, _lsh1); lsh++) {
                                int bas_kl = ksh * nbas + lsh;
                                if (bas_ij < bas_kl) continue;

                                double q_kl = q_cond[bas_kl];
                                double q_ijkl = q_ij + q_kl;
                                if (q_ijkl < cutoff) continue;
                                double d_cutoff = cutoff - q_ijkl;
                                if (vk == NULL) {
                                        // J only
                                        if ((dm_cond[jsh*nbas+ish] < d_cutoff) &&
                                            (dm_cond[lsh*nbas+ksh] < d_cutoff)) {
                                                continue;
                                        }
                                } else if (vj == NULL) {
                                        // K only
                                        if ((dm_cond[jsh*nbas+ksh] < d_cutoff) &&
                                            (dm_cond[jsh*nbas+lsh] < d_cutoff) &&
                                            (dm_cond[ish*nbas+ksh] < d_cutoff) &&
                                            (dm_cond[ish*nbas+lsh] < d_cutoff)) {
                                                continue;
                                        }
                                } else if ((dm_cond[jsh*nbas+ish] < d_cutoff) &&
                                           (dm_cond[lsh*nbas+ksh] < d_cutoff) &&
                                           (dm_cond[jsh*nbas+ksh] < d_cutoff) &&
                                           (dm_cond[jsh*nbas+lsh] < d_cutoff) &&
                                           (dm_cond[ish*nbas+ksh] < d_cutoff) &&
                                           (dm_cond[ish*nbas+lsh] < d_cutoff)) {
                                        continue;
                                }

                                if (ntasks >= max_tasks) {
                                        return -1;
                                }
                                shl_quartets[ntasks*4+0] = ish;
                                shl_quartets[ntasks*4+1] = jsh;
                                shl_quartets[ntasks*4+2] = ksh;
                                shl_quartets[ntasks*4+3] = lsh;
                                ntasks++;
                        } }
                } }
        } } } }
        return ntasks;
}

static int _create_tasks_mt(uint16_t *shl_quartets, double *vj, double *vk,
                            double *q_cond, double *dm_cond, double cutoff,
                            int *shls_slice, int *ao_loc, int max_tasks,
                            int *atm, int natm, int *bas, int nbas, double *env)
{
        // the first shell-quartet in the bin
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        int lsh1 = shls_slice[7];
        //int di = ao_loc[ish0+1] - ao_loc[ish0];
        //int dj = ao_loc[jsh0+1] - ao_loc[jsh0];
        //int dk = ao_loc[ksh0+1] - ao_loc[ksh0];
        //int dl = ao_loc[lsh0+1] - ao_loc[lsh0];
        int basblk_i = SHL_BLOCKSIZE; //MAX(AO_BLOCKSIZE/di, 4);
        int basblk_j = SHL_BLOCKSIZE; //MAX(AO_BLOCKSIZE/dj, 4);
        int basblk_k = SHL_BLOCKSIZE; //MAX(AO_BLOCKSIZE/dk, 4);
        int basblk_l = SHL_BLOCKSIZE; //MAX(AO_BLOCKSIZE/dl, 4);
        int nthreads = omp_get_num_threads();
        int tasks_per_thread[256];
        int tasks_offsets[256];
        int tot_tasks = 0;

#pragma omp parallel
{
        int ntasks = 0;
        int thread_id = omp_get_thread_num();
        int _ish0, _jsh0, _ksh0, _lsh0, _ish1, _jsh1, _ksh1, _lsh1;
#pragma omp for schedule(static, 1)
        for (_ish0 = ish0; _ish0 < ish1; _ish0+=basblk_i) {
                _ish1 = MIN(_ish0+basblk_i, ish1);
                jsh1 = MIN(shls_slice[3], _ish1);
                ksh1 = MIN(shls_slice[5], _ish1);
        for (_jsh0 = jsh0; _jsh0 < jsh1; _jsh0+=basblk_j) {
                _jsh1 = MIN(_jsh0+basblk_j, jsh1);
        for (_ksh0 = ksh0; _ksh0 < ksh1; _ksh0+=basblk_k) {
                _ksh1 = MIN(_ksh0+basblk_k, ksh1);
                lsh1 = MIN(shls_slice[7], _ksh1);
        for (_lsh0 = lsh0; _lsh0 < lsh1; _lsh0+=basblk_l) {
                _lsh1 = MIN(_lsh0+basblk_l, lsh1);
                for (int ish = _ish0; ish < _ish1; ish++) {
                for (int jsh = _jsh0; jsh < MIN(ish+1, _jsh1); jsh++) {
                        int bas_ij = ish * nbas + jsh;
                        double q_ij = q_cond[bas_ij];
                        for (int ksh = _ksh0; ksh < MIN(ish+1, _ksh1); ksh++) {
                        for (int lsh = _lsh0; lsh < MIN(ksh+1, _lsh1); lsh++) {
                                int bas_kl = ksh * nbas + lsh;
                                if (bas_ij < bas_kl) continue;

                                double q_kl = q_cond[bas_kl];
                                double q_ijkl = q_ij + q_kl;
                                if (q_ijkl < cutoff) continue;
                                double d_cutoff = cutoff - q_ijkl;
                                if (vk == NULL) {
                                        // J only
                                        if ((dm_cond[jsh*nbas+ish] < d_cutoff) &&
                                            (dm_cond[lsh*nbas+ksh] < d_cutoff)) {
                                                continue;
                                        }
                                } else if (vj == NULL) {
                                        // K only
                                        if ((dm_cond[jsh*nbas+ksh] < d_cutoff) &&
                                            (dm_cond[jsh*nbas+lsh] < d_cutoff) &&
                                            (dm_cond[ish*nbas+ksh] < d_cutoff) &&
                                            (dm_cond[ish*nbas+lsh] < d_cutoff)) {
                                                continue;
                                        }
                                } else if ((dm_cond[jsh*nbas+ish] < d_cutoff) &&
                                           (dm_cond[lsh*nbas+ksh] < d_cutoff) &&
                                           (dm_cond[jsh*nbas+ksh] < d_cutoff) &&
                                           (dm_cond[jsh*nbas+lsh] < d_cutoff) &&
                                           (dm_cond[ish*nbas+ksh] < d_cutoff) &&
                                           (dm_cond[ish*nbas+lsh] < d_cutoff)) {
                                        continue;
                                }
                                ntasks++;
                        } }
                } }
        } } } }
        tasks_per_thread[thread_id] = ntasks;
#pragma omp barrier
#pragma omp master
        for (int n = 0; n < nthreads; n++) {
                tasks_offsets[n] = tot_tasks;
                tot_tasks += tasks_per_thread[n];
        }
}
        if (tot_tasks >= max_tasks) {
                return -1;
        }

#pragma omp parallel
{
        int thread_id = omp_get_thread_num();
        int ntasks = tasks_offsets[thread_id];
        int _ish0, _jsh0, _ksh0, _lsh0, _ish1, _jsh1, _ksh1, _lsh1;
#pragma omp for schedule(static, 1)
        for (_ish0 = ish0; _ish0 < ish1; _ish0+=basblk_i) {
                _ish1 = MIN(_ish0+basblk_i, ish1);
                jsh1 = MIN(shls_slice[3], _ish1);
                ksh1 = MIN(shls_slice[5], _ish1);
        for (_jsh0 = jsh0; _jsh0 < jsh1; _jsh0+=basblk_j) {
                _jsh1 = MIN(_jsh0+basblk_j, jsh1);
        for (_ksh0 = ksh0; _ksh0 < ksh1; _ksh0+=basblk_k) {
                _ksh1 = MIN(_ksh0+basblk_k, ksh1);
                lsh1 = MIN(shls_slice[7], _ksh1);
        for (_lsh0 = lsh0; _lsh0 < lsh1; _lsh0+=basblk_l) {
                _lsh1 = MIN(_lsh0+basblk_l, lsh1);
                for (int ish = _ish0; ish < _ish1; ish++) {
                for (int jsh = _jsh0; jsh < MIN(ish+1, _jsh1); jsh++) {
                        int bas_ij = ish * nbas + jsh;
                        double q_ij = q_cond[bas_ij];
                        for (int ksh = _ksh0; ksh < MIN(ish+1, _ksh1); ksh++) {
                        for (int lsh = _lsh0; lsh < MIN(ksh+1, _lsh1); lsh++) {
                                int bas_kl = ksh * nbas + lsh;
                                if (bas_ij < bas_kl) continue;

                                double q_kl = q_cond[bas_kl];
                                double q_ijkl = q_ij + q_kl;
                                if (q_ijkl < cutoff) continue;
                                double d_cutoff = cutoff - q_ijkl;
                                if (vk == NULL) {
                                        // J only
                                        if ((dm_cond[jsh*nbas+ish] < d_cutoff) &&
                                            (dm_cond[lsh*nbas+ksh] < d_cutoff)) {
                                                continue;
                                        }
                                } else if (vj == NULL) {
                                        // K only
                                        if ((dm_cond[jsh*nbas+ksh] < d_cutoff) &&
                                            (dm_cond[jsh*nbas+lsh] < d_cutoff) &&
                                            (dm_cond[ish*nbas+ksh] < d_cutoff) &&
                                            (dm_cond[ish*nbas+lsh] < d_cutoff)) {
                                                continue;
                                        }
                                } else if ((dm_cond[jsh*nbas+ish] < d_cutoff) &&
                                           (dm_cond[lsh*nbas+ksh] < d_cutoff) &&
                                           (dm_cond[jsh*nbas+ksh] < d_cutoff) &&
                                           (dm_cond[jsh*nbas+lsh] < d_cutoff) &&
                                           (dm_cond[ish*nbas+ksh] < d_cutoff) &&
                                           (dm_cond[ish*nbas+lsh] < d_cutoff)) {
                                        continue;
                                }

                                shl_quartets[ntasks*4+0] = ish;
                                shl_quartets[ntasks*4+1] = jsh;
                                shl_quartets[ntasks*4+2] = ksh;
                                shl_quartets[ntasks*4+3] = lsh;
                                ntasks++;
                        } }
                } }
        } } } }
}
        return tot_tasks;
}

int RYS_create_tasks(uint16_t *shl_quartets, double *vj, double *vk,
                     double *q_cond, double *dm_cond, double cutoff,
                     int *shls_slice, int *ao_loc, int max_tasks,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        int nthreads = omp_get_num_threads();
        int (*f)(uint16_t *shl_quartets, double *vj, double *vk,
                 double *q_cond, double *dm_cond, double cutoff,
                 int *shls_slice, int *ao_loc, int max_tasks,
                 int *atm, int natm, int *bas, int nbas, double *env);
        if (nthreads == 1) {
                f = _create_tasks_1t;
        } else {
                f = _create_tasks_mt;
        }
        return (*f)(shl_quartets, vj, vk, q_cond, dm_cond, cutoff,
                    shls_slice, ao_loc, max_tasks, atm, natm, bas, nbas, env);
}
