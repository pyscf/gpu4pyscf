#pragma once
namespace gpu4pyscf::gpbc::multi_grid {

__constant__ double lattice_vectors[9];
__constant__ double reciprocal_lattice_vectors[9];
__constant__ double dxyz_dabc[9];
__constant__ double reciprocal_norm[3];

} // namespace gpu4pyscf::gpbc::multi_grid