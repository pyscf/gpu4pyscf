#pragma once
namespace gpu4pyscf::gpbc::multi_grid {

extern __constant__ double lattice_vectors[9];
extern __constant__ double reciprocal_lattice_vectors[9];
extern __constant__ double dxyz_dabc[9];
extern __constant__ double reciprocal_norm[3];

} // namespace gpu4pyscf::gpbc::multi_grid
