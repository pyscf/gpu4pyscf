#include <sycl_device.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <oneapi/mkl/blas.hpp>
#include <iostream>


// Left-side in-place triangular solve, column-major: B <- op(A)^-1 B.
// trans: 0 = N, 1 = T, 2 = C (conjugate transpose), as in cuBLAS / CuPy.
template <typename T>
static void onemkl_trsm_impl(T* a, T* b, int m, int n, int lda, int ldb,
                             int lower, int trans, int unit_diagonal) {
  auto queue = *sycl_get_queue();

  oneapi::mkl::uplo uplo = lower ? oneapi::mkl::uplo::L : oneapi::mkl::uplo::U;
  oneapi::mkl::transpose transA = trans == 2 ? oneapi::mkl::transpose::C
                                : trans == 1 ? oneapi::mkl::transpose::T
                                             : oneapi::mkl::transpose::N;
  oneapi::mkl::diag diag = unit_diagonal ? oneapi::mkl::diag::U : oneapi::mkl::diag::N;
  T alpha = T(1);

  // in-place. Not waited on, like cuBLAS trsm on a stream: later work on
  // the in-order queue is ordered after it, and trsm has no scratchpad or
  // info output to read back.
  oneapi::mkl::blas::column_major::trsm(queue,
                                        oneapi::mkl::side::left,
                                        uplo,
                                        transA,
                                        diag,
                                        m, n, alpha,
                                        a, lda, b, ldb);
}

extern "C" void onemkl_trsm(double* a, double* b,
                            int m, int n, int lda, int ldb,
                            int lower, int trans, int unit_diagonal) {
  onemkl_trsm_impl(a, b, m, n, lda, ldb, lower, trans, unit_diagonal);
}

extern "C" void onemkl_strsm(float* a, float* b,
                             int m, int n, int lda, int ldb,
                             int lower, int trans, int unit_diagonal) {
  onemkl_trsm_impl(a, b, m, n, lda, ldb, lower, trans, unit_diagonal);
}

extern "C" void onemkl_ctrsm(std::complex<float>* a, std::complex<float>* b,
                             int m, int n, int lda, int ldb,
                             int lower, int trans, int unit_diagonal) {
  onemkl_trsm_impl(a, b, m, n, lda, ldb, lower, trans, unit_diagonal);
}

extern "C" void onemkl_ztrsm(std::complex<double>* a, std::complex<double>* b,
                             int m, int n, int lda, int ldb,
                             int lower, int trans, int unit_diagonal) {
  onemkl_trsm_impl(a, b, m, n, lda, ldb, lower, trans, unit_diagonal);
}

// cuBLAS-style column-major dgemm: C <- alpha op(A) op(B) + beta C.
// trans: 0 = N, 1 = T, 2 = C. Not waited on, like cublasDgemm on a stream.
extern "C" void onemkl_dgemm(int transa, int transb, long m, long n, long k,
                             double alpha, const double* a, long lda,
                             const double* b, long ldb,
                             double beta, double* c, long ldc) {
  auto queue = *sycl_get_queue();
  auto op = [](int t) {
    return t == 2 ? oneapi::mkl::transpose::C
         : t == 1 ? oneapi::mkl::transpose::T
                  : oneapi::mkl::transpose::N;
  };
  oneapi::mkl::blas::column_major::gemm(queue, op(transa), op(transb),
                                        m, n, k, alpha, a, lda, b, ldb,
                                        beta, c, ldc);
}

extern "C" void onemkl_dsygvd_scratchpad_size(int itype,
                                              int n,
                                              int lda,
                                              int ldb,
                                              int* scratch_size) {
  try {
    auto queue = *sycl_get_queue();
    *scratch_size = oneapi::mkl::lapack::sygvd_scratchpad_size<double>(queue,
                                                                       itype,
                                                                       oneapi::mkl::job::vec,
                                                                       oneapi::mkl::uplo::lower,
                                                                       n, lda, ldb);
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
  }
}
extern "C" void onemkl_zhegvd_scratchpad_size(int itype,
                                              int n,
                                              int lda,
                                              int ldb,
                                              int* scratch_size) {
  try {
    auto queue = *sycl_get_queue();
    *scratch_size = oneapi::mkl::lapack::hegvd_scratchpad_size<std::complex<double>>(queue,
                                                                                     itype,
                                                                                     oneapi::mkl::job::vec,
                                                                                     oneapi::mkl::uplo::lower,
                                                                                     n, lda, ldb);
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
  }
}

extern "C" void onemkl_dsygvd(int itype,
                              int n,
                              double* A,
                              int lda,
                              double* B,
                              int ldb,
                              double* w,
                              double* scratchpad,
                              int scratchpad_size) {
  try {
    auto queue = *sycl_get_queue();
    auto e = oneapi::mkl::lapack::sygvd(queue,
                                        itype,
                                        oneapi::mkl::job::vec,
                                        oneapi::mkl::uplo::lower,
                                        n,
                                        A, lda,
                                        B, ldb,
                                        w, scratchpad, scratchpad_size);
    e.wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
  }
}
extern "C" void onemkl_zhegvd(int itype,
                              int n,
                              std::complex<double>* A,
                              int lda,
                              std::complex<double>* B,
                              int ldb,
                              double* w,
                              std::complex<double>* scratchpad,
                              int scratchpad_size) {
  try {
    auto queue = *sycl_get_queue();
    auto e = oneapi::mkl::lapack::hegvd(queue,
                                        itype,
                                        oneapi::mkl::job::vec,
                                        oneapi::mkl::uplo::lower,
                                        n,
                                        A, lda,
                                        B, ldb,
                                        w, scratchpad, scratchpad_size);
    e.wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
  }
}



extern "C" int64_t onemkl_dpotrf_scratchpad_size(int n,
                                                 int lda) {
  try {
    auto queue = *sycl_get_queue();
    auto scratch_size = oneapi::mkl::lapack::potrf_scratchpad_size<double>(queue,
                                                                           oneapi::mkl::uplo::upper,
                                                                           n, lda);
    return scratch_size;
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
  }
}
extern "C" int64_t onemkl_zpotrf_scratchpad_size(int n,
                                                 int lda) {
  try {
    auto queue = *sycl_get_queue();
    auto scratch_size = oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<double>>(queue,
                                                                                         oneapi::mkl::uplo::upper,
                                                                                         n, lda);
    return scratch_size;
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
  }
}


// Returns 0 on success, non-zero if the factorization failed. cuSOLVER
// reports this through dev_info and cusolver.py raises LinAlgError on it;
// callers in df.py rely on that to fall back to an eigendecomposition when
// j2c is singular, so the status has to survive the wrapper.
extern "C" int onemkl_dpotrf(int n,
                              double* A,
                              int lda,
                              double* scratchpad,
                              int scratchpad_size) {
  try {
    auto queue = *sycl_get_queue();
    auto e = oneapi::mkl::lapack::potrf(queue,
                                        oneapi::mkl::uplo::upper,
                                        n,
                                        A, lda,
                                        scratchpad, scratchpad_size);
    e.wait();
  } catch (oneapi::mkl::lapack::exception const& e) {
    // info() > 0 means the leading minor of that order is not positive
    // definite -- the expected signal for a singular input, not a bug.
    return e.info() ? static_cast<int>(e.info()) : 1;
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception in dpotrf: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
// Returns 0 on success, non-zero if the factorization failed. cuSOLVER
// reports this through dev_info and cusolver.py raises LinAlgError on it;
// callers in df.py rely on that to fall back to an eigendecomposition when
// j2c is singular, so the status has to survive the wrapper.
extern "C" int onemkl_zpotrf(int n,
                              std::complex<double>* A,
                              int lda,
                              std::complex<double>* scratchpad,
                              int scratchpad_size) {
  try {
    auto queue = *sycl_get_queue();
    auto e = oneapi::mkl::lapack::potrf(queue,
                                        oneapi::mkl::uplo::upper,
                                        n,
                                        A, lda,
                                        scratchpad, scratchpad_size);
    e.wait();
  } catch (oneapi::mkl::lapack::exception const& e) {
    // info() > 0 means the leading minor of that order is not positive
    // definite -- the expected signal for a singular input, not a bug.
    return e.info() ? static_cast<int>(e.info()) : 1;
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception in zpotrf: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
