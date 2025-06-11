#include "gint/sycl_device.hpp"
#include <oneapi/mkl/lapack.hpp>
#include <oneapi/mkl/blas.hpp>
#include <iostream>


extern "C" void onemkl_trsm(double* a, double* b,
                            int m, int n, int lda, int ldb,
                            int lower, int trans, int unit_diagonal) {
  auto queue = *sycl_get_queue();

  oneapi::mkl::uplo uplo = lower ? oneapi::mkl::uplo::L : oneapi::mkl::uplo::U;
  oneapi::mkl::transpose transA = trans ? oneapi::mkl::transpose::T : oneapi::mkl::transpose::N;
  oneapi::mkl::diag diag = unit_diagonal ? oneapi::mkl::diag::U : oneapi::mkl::diag::N;
  double alpha = 1.0;

  // in-place
  auto e = oneapi::mkl::blas::column_major::trsm(queue,
                                                 oneapi::mkl::side::left,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 m, n, alpha,
                                                 a, lda, b, ldb);
  e.wait();
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



extern "C" void onemkl_dpotrf_scratchpad_size(int n,
                                              int lda,
                                              int* scratch_size) {
  try {
    auto queue = *sycl_get_queue();
    *scratch_size = oneapi::mkl::lapack::potrf_scratchpad_size<double>(queue,
                                                                       oneapi::mkl::uplo::upper,
                                                                       n, lda);
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
  }
}
extern "C" void onemkl_zpotrf_scratchpad_size(int n,
                                              int lda,
                                              int* scratch_size) {
  try {
    auto queue = *sycl_get_queue();
    *scratch_size = oneapi::mkl::lapack::potrf_scratchpad_size<std::complex<double>>(queue,
                                                                                     oneapi::mkl::uplo::upper,
                                                                                     n, lda);
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
  }
}


extern "C" void onemkl_dpotrf(int n,
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
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
  }
}
extern "C" void onemkl_zpotrf(int n,
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
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
  }
}
