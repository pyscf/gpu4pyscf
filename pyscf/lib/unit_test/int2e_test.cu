#include <catch.hpp>

#include "gint/nr_fill_ao_ints.cuh"
#include <armadillo>

TEST_CASE("Let's try replicating the numbers!") {
  BasisProdCache * bpcache_host;

  double diagonal_factor = 0.5;
//  BasisProdCache * bpcache_dev_pointer = &c_bpcache;

  std::vector<int> basis_pair_to_shells = arma::conv_to<std::vector<int>>::from(
      arma::vectorise(arma::Mat<int>{{1, 0, 1},
                                     {1, 0, 0}}.t()));

  std::vector<int> basis_pairs_locations{0, 3};
  int ncptype = 1;
  int n_atoms = 2;
  int n_basis = 2;// I guess it's H_1s x 2
  std::vector<int> ao_loc{0, 1, 2};
  std::vector<int> atm = arma::conv_to<std::vector<int>>::from(
      arma::vectorise(arma::Mat<int>{{1, 20, 1, 23, 0, 0},
                                     {1, 24, 1, 27, 0, 0}}.t()));
  std::vector<int> basis = arma::conv_to<std::vector<int>>::from(
      arma::vectorise(arma::Mat<int>({{0, 0, 3, 1, 0, 28, 31, 0},
                                      {1, 0, 3, 1, 0, 28, 31, 0}}).t()));

  std::vector<double> environment{0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0,
                                  1.88972612, 0, 0, 0, 3.42525091, 0.62391373,
                                  0.1688554, 0.98170675, 0.94946401,
                                  0.29590645};


  GINTinit_basis_prod(&bpcache_host, diagonal_factor, ao_loc.data(),
                      basis_pair_to_shells.data(),
                      basis_pairs_locations.data(),
                      ncptype, atm.data(), n_atoms, basis.data(), n_basis,
                      environment.data());

}
