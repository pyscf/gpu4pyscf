#pragma once

namespace gpu4pyscf::gpbc::multi_grid {

template <typename T, int ANG>
__forceinline__ __device__ void gto_cartesian(T values[], const T fx,
                                              const T fy, const T fz) {
  if constexpr (ANG == 0) {
    values[0] = 1;
  } else if constexpr (ANG == 1) {
    values[0] = fx;
    values[1] = fy;
    values[2] = fz;
  } else if constexpr (ANG == 2) {
    values[0] = fx * fx;
    values[1] = fx * fy;
    values[2] = fx * fz;
    values[3] = fy * fy;
    values[4] = fy * fz;
    values[5] = fz * fz;
  } else if constexpr (ANG == 3) {
    values[0] = fx * fx * fx;
    values[1] = fx * fx * fy;
    values[2] = fx * fx * fz;
    values[3] = fx * fy * fy;
    values[4] = fx * fy * fz;
    values[5] = fx * fz * fz;
    values[6] = fy * fy * fy;
    values[7] = fy * fy * fz;
    values[8] = fy * fz * fz;
    values[9] = fz * fz * fz;
  } else if constexpr (ANG == 4) {
    values[0] = fx * fx * fx * fx;
    values[1] = fx * fx * fx * fy;
    values[2] = fx * fx * fx * fz;
    values[3] = fx * fx * fy * fy;
    values[4] = fx * fx * fy * fz;
    values[5] = fx * fx * fz * fz;
    values[6] = fx * fy * fy * fy;
    values[7] = fx * fy * fy * fz;
    values[8] = fx * fy * fz * fz;
    values[9] = fx * fz * fz * fz;
    values[10] = fy * fy * fy * fy;
    values[11] = fy * fy * fy * fz;
    values[12] = fy * fy * fz * fz;
    values[13] = fy * fz * fz * fz;
    values[14] = fz * fz * fz * fz;
  }
}

namespace gradient {
template <typename T, int ANG>
__forceinline__ __device__ void
gto_cartesian(T gradient_values[], const T original_values[], const T fx,
              const T fy, const T fz, const T exponent) {
  const T minus_2afx = -2 * exponent * fx;
  const T minus_2afy = -2 * exponent * fy;
  const T minus_2afz = -2 * exponent * fz;
  if constexpr (ANG == 0) {
    // For s orbital (ANG=0), f(x,y,z) = 1
    // f'_x = 0, so g_x = -2 * exponent * fx
    gradient_values[0] = minus_2afx; // x gradient
    gradient_values[1] = minus_2afy; // y gradient
    gradient_values[2] = minus_2afz; // z gradient
  } else if constexpr (ANG == 1) {
    // For p orbitals (ANG=1), f(x,y,z) = {x, y, z}
    // First row: x gradient
    gradient_values[0] = 1 + minus_2afx * fx; // d/dx(x) - 2*exponent*x*fx
    gradient_values[1] = minus_2afx * fy;     // d/dx(y) - 2*exponent*y*fx
    gradient_values[2] = minus_2afx * fz;     // d/dx(z) - 2*exponent*z*fx
    // Second row: y gradient
    gradient_values[3] = minus_2afy * fx;     // d/dy(x) - 2*exponent*x*fy
    gradient_values[4] = 1 + minus_2afy * fy; // d/dy(y) - 2*exponent*y*fy
    gradient_values[5] = minus_2afy * fz;     // d/dy(z) - 2*exponent*z*fy
    // Third row: z gradient
    gradient_values[6] = minus_2afz * fx;     // d/dz(x) - 2*exponent*x*fz
    gradient_values[7] = minus_2afz * fy;     // d/dz(y) - 2*exponent*y*fz
    gradient_values[8] = 1 + minus_2afz * fz; // d/dz(z) - 2*exponent*z*fz
  } else if constexpr (ANG == 2) {
    // For d orbitals (ANG=2), f(x,y,z) = {xx, xy, xz, yy, yz, zz}
    // First row: x gradient
    gradient_values[0] =
        2 * fx + minus_2afx * original_values[0]; // d/dx(xx) - 2*exponent*xx*fx
    gradient_values[1] =
        fy + minus_2afx * original_values[1]; // d/dx(xy) - 2*exponent*xy*fx
    gradient_values[2] =
        fz + minus_2afx * original_values[2]; // d/dx(xz) - 2*exponent*xz*fx
    gradient_values[3] =
        minus_2afx * original_values[3]; // d/dx(yy) - 2*exponent*yy*fx
    gradient_values[4] =
        minus_2afx * original_values[4]; // d/dx(yz) - 2*exponent*yz*fx
    gradient_values[5] =
        minus_2afx * original_values[5]; // d/dx(zz) - 2*exponent*zz*fx
    // Second row: y gradient
    gradient_values[6] =
        minus_2afy * original_values[0]; // d/dy(xx) - 2*exponent*xx*fy
    gradient_values[7] =
        fx + minus_2afy * original_values[1]; // d/dy(xy) - 2*exponent*xy*fy
    gradient_values[8] =
        minus_2afy * original_values[2]; // d/dy(xz) - 2*exponent*xz*fy
    gradient_values[9] =
        2 * fy + minus_2afy * original_values[3]; // d/dy(yy) - 2*exponent*yy*fy
    gradient_values[10] =
        fz + minus_2afy * original_values[4]; // d/dy(yz) - 2*exponent*yz*fy
    gradient_values[11] =
        minus_2afy * original_values[5]; // d/dy(zz) - 2*exponent*zz*fy
    // Third row: z gradient
    gradient_values[12] =
        minus_2afz * original_values[0]; // d/dz(xx) - 2*exponent*xx*fz
    gradient_values[13] =
        minus_2afz * original_values[1]; // d/dz(xy) - 2*exponent*xy*fz
    gradient_values[14] =
        fx + minus_2afz * original_values[2]; // d/dz(xz) - 2*exponent*xz*fz
    gradient_values[15] =
        minus_2afz * original_values[3]; // d/dz(yy) - 2*exponent*yy*fz
    gradient_values[16] =
        fy + minus_2afz * original_values[4]; // d/dz(yz) - 2*exponent*yz*fz
    gradient_values[17] =
        2 * fz + minus_2afz * original_values[5]; // d/dz(zz) - 2*exponent*zz*fz
  } else if constexpr (ANG == 3) {
    // For f orbitals (ANG=3), f(x,y,z) = {xxx, xxy, xxz, xyy, xyz, xzz, yyy,
    // yyz, yzz, zzz} First row: x gradient
    gradient_values[0] =
        3 * fx * fx +
        minus_2afx * original_values[0]; // d/dx(xxx) - 2*exponent*xxx*fx
    gradient_values[1] =
        2 * fx * fy +
        minus_2afx * original_values[1]; // d/dx(xxy) - 2*exponent*xxy*fx
    gradient_values[2] =
        2 * fx * fz +
        minus_2afx * original_values[2]; // d/dx(xxz) - 2*exponent*xxz*fx
    gradient_values[3] =
        fy * fy +
        minus_2afx * original_values[3]; // d/dx(xyy) - 2*exponent*xyy*fx
    gradient_values[4] =
        fy * fz +
        minus_2afx * original_values[4]; // d/dx(xyz) - 2*exponent*xyz*fx
    gradient_values[5] =
        fz * fz +
        minus_2afx * original_values[5]; // d/dx(xzz) - 2*exponent*xzz*fx
    gradient_values[6] =
        minus_2afx * original_values[6]; // d/dx(yyy) - 2*exponent*yyy*fx
    gradient_values[7] =
        minus_2afx * original_values[7]; // d/dx(yyz) - 2*exponent*yyz*fx
    gradient_values[8] =
        minus_2afx * original_values[8]; // d/dx(yzz) - 2*exponent*yzz*fx
    gradient_values[9] =
        minus_2afx * original_values[9]; // d/dx(zzz) - 2*exponent*zzz*fx
    // Second row: y gradient
    gradient_values[10] =
        minus_2afy * original_values[0]; // d/dy(xxx) - 2*exponent*xxx*fy
    gradient_values[11] =
        fx * fx +
        minus_2afy * original_values[1]; // d/dy(xxy) - 2*exponent*xxy*fy
    gradient_values[12] =
        minus_2afy * original_values[2]; // d/dy(xxz) - 2*exponent*xxz*fy
    gradient_values[13] =
        2 * fx * fy +
        minus_2afy * original_values[3]; // d/dy(xyy) - 2*exponent*xyy*fy
    gradient_values[14] =
        fx * fz +
        minus_2afy * original_values[4]; // d/dy(xyz) - 2*exponent*xyz*fy
    gradient_values[15] =
        minus_2afy * original_values[5]; // d/dy(xzz) - 2*exponent*xzz*fy
    gradient_values[16] =
        3 * fy * fy +
        minus_2afy * original_values[6]; // d/dy(yyy) - 2*exponent*yyy*fy
    gradient_values[17] =
        2 * fy * fz +
        minus_2afy * original_values[7]; // d/dy(yyz) - 2*exponent*yyz*fy
    gradient_values[18] =
        fz * fz +
        minus_2afy * original_values[8]; // d/dy(yzz) - 2*exponent*yzz*fy
    gradient_values[19] =
        minus_2afy * original_values[9]; // d/dy(zzz) - 2*exponent*zzz*fy
    // Third row: z gradient
    gradient_values[20] =
        minus_2afz * original_values[0]; // d/dz(xxx) - 2*exponent*xxx*fz
    gradient_values[21] =
        minus_2afz * original_values[1]; // d/dz(xxy) - 2*exponent*xxy*fz
    gradient_values[22] =
        fx * fx +
        minus_2afz * original_values[2]; // d/dz(xxz) - 2*exponent*xxz*fz
    gradient_values[23] =
        minus_2afz * original_values[3]; // d/dz(xyy) - 2*exponent*xyy*fz
    gradient_values[24] =
        fx * fy +
        minus_2afz * original_values[4]; // d/dz(xyz) - 2*exponent*xyz*fz
    gradient_values[25] =
        2 * fx * fz +
        minus_2afz * original_values[5]; // d/dz(xzz) - 2*exponent*xzz*fz
    gradient_values[26] =
        minus_2afz * original_values[6]; // d/dz(yyy) - 2*exponent*yyy*fz
    gradient_values[27] =
        fy * fy +
        minus_2afz * original_values[7]; // d/dz(yyz) - 2*exponent*yyz*fz
    gradient_values[28] =
        2 * fy * fz +
        minus_2afz * original_values[8]; // d/dz(yzz) - 2*exponent*yzz*fz
    gradient_values[29] =
        3 * fz * fz +
        minus_2afz * original_values[9]; // d/dz(zzz) - 2*exponent*zzz*fz
  } else if constexpr (ANG == 4) {
    // For g orbitals (ANG=4), f(x,y,z) = {xxxx, xxxy, xxxz, xxyy, xxyz, xxzz,
    // xyyy, xyyz, xyzz, xzzz, yyyy, yyyz, yyzz, yzzz, zzzz} First row: x
    // gradient
    gradient_values[0] =
        4 * fx * fx * fx +
        minus_2afx * original_values[0]; // d/dx(xxxx) - 2*exponent*xxxx*fx
    gradient_values[1] =
        3 * fx * fx * fy +
        minus_2afx * original_values[1]; // d/dx(xxxy) - 2*exponent*xxxy*fx
    gradient_values[2] =
        3 * fx * fx * fz +
        minus_2afx * original_values[2]; // d/dx(xxxz) - 2*exponent*xxxz*fx
    gradient_values[3] =
        2 * fx * fy * fy +
        minus_2afx * original_values[3]; // d/dx(xxyy) - 2*exponent*xxyy*fx
    gradient_values[4] =
        2 * fx * fy * fz +
        minus_2afx * original_values[4]; // d/dx(xxyz) - 2*exponent*xxyz*fx
    gradient_values[5] =
        2 * fx * fz * fz +
        minus_2afx * original_values[5]; // d/dx(xxzz) - 2*exponent*xxzz*fx
    gradient_values[6] =
        fy * fy * fy +
        minus_2afx * original_values[6]; // d/dx(xyyy) - 2*exponent*xyyy*fx
    gradient_values[7] =
        fy * fy * fz +
        minus_2afx * original_values[7]; // d/dx(xyyz) - 2*exponent*xyyz*fx
    gradient_values[8] =
        fy * fz * fz +
        minus_2afx * original_values[8]; // d/dx(xyzz) - 2*exponent*xyzz*fx
    gradient_values[9] =
        fz * fz * fz +
        minus_2afx * original_values[9]; // d/dx(xzzz) - 2*exponent*xzzz*fx
    gradient_values[10] =
        minus_2afx * original_values[10]; // d/dx(yyyy) - 2*exponent*yyyy*fx
    gradient_values[11] =
        minus_2afx * original_values[11]; // d/dx(yyyz) - 2*exponent*yyyz*fx
    gradient_values[12] =
        minus_2afx * original_values[12]; // d/dx(yyzz) - 2*exponent*yyzz*fx
    gradient_values[13] =
        minus_2afx * original_values[13]; // d/dx(yzzz) - 2*exponent*yzzz*fx
    gradient_values[14] =
        minus_2afx * original_values[14]; // d/dx(zzzz) - 2*exponent*zzzz*fx
    // Second row: y gradient
    gradient_values[15] =
        minus_2afy * original_values[0]; // d/dy(xxxx) - 2*exponent*xxxx*fy
    gradient_values[16] =
        fx * fx * fx +
        minus_2afy * original_values[1]; // d/dy(xxxy) - 2*exponent*xxxy*fy
    gradient_values[17] =
        minus_2afy * original_values[2]; // d/dy(xxxz) - 2*exponent*xxxz*fy
    gradient_values[18] =
        2 * fx * fx * fy +
        minus_2afy * original_values[3]; // d/dy(xxyy) - 2*exponent*xxyy*fy
    gradient_values[19] =
        fx * fx * fz +
        minus_2afy * original_values[4]; // d/dy(xxyz) - 2*exponent*xxyz*fy
    gradient_values[20] =
        minus_2afy * original_values[5]; // d/dy(xxzz) - 2*exponent*xxzz*fy
    gradient_values[21] =
        3 * fx * fy * fy +
        minus_2afy * original_values[6]; // d/dy(xyyy) - 2*exponent*xyyy*fy
    gradient_values[22] =
        2 * fx * fy * fz +
        minus_2afy * original_values[7]; // d/dy(xyyz) - 2*exponent*xyyz*fy
    gradient_values[23] =
        fx * fz * fz +
        minus_2afy * original_values[8]; // d/dy(xyzz) - 2*exponent*xyzz*fy
    gradient_values[24] =
        minus_2afy * original_values[9]; // d/dy(xzzz) - 2*exponent*xzzz*fy
    gradient_values[25] =
        4 * fy * fy * fy +
        minus_2afy * original_values[10]; // d/dy(yyyy) - 2*exponent*yyyy*fy
    gradient_values[26] =
        3 * fy * fy * fz +
        minus_2afy * original_values[11]; // d/dy(yyyz) - 2*exponent*yyyz*fy
    gradient_values[27] =
        2 * fy * fz * fz +
        minus_2afy * original_values[12]; // d/dy(yyzz) - 2*exponent*yyzz*fy
    gradient_values[28] =
        fz * fz * fz +
        minus_2afy * original_values[13]; // d/dy(yzzz) - 2*exponent*yzzz*fy
    gradient_values[29] =
        minus_2afy * original_values[14]; // d/dy(zzzz) - 2*exponent*zzzz*fy
    // Third row: z gradient
    gradient_values[30] =
        minus_2afz * original_values[0]; // d/dz(xxxx) - 2*exponent*xxxx*fz
    gradient_values[31] =
        minus_2afz * original_values[1]; // d/dz(xxxy) - 2*exponent*xxxy*fz
    gradient_values[32] =
        fx * fx * fx +
        minus_2afz * original_values[2]; // d/dz(xxxz) - 2*exponent*xxxz*fz
    gradient_values[33] =
        minus_2afz * original_values[3]; // d/dz(xxyy) - 2*exponent*xxyy*fz
    gradient_values[34] =
        fx * fx * fy +
        minus_2afz * original_values[4]; // d/dz(xxyz) - 2*exponent*xxyz*fz
    gradient_values[35] =
        2 * fx * fx * fz +
        minus_2afz * original_values[5]; // d/dz(xxzz) - 2*exponent*xxzz*fz
    gradient_values[36] =
        minus_2afz * original_values[6]; // d/dz(xyyy) - 2*exponent*xyyy*fz
    gradient_values[37] =
        fx * fy * fy +
        minus_2afz * original_values[7]; // d/dz(xyyz) - 2*exponent*xyyz*fz
    gradient_values[38] =
        2 * fx * fy * fz +
        minus_2afz * original_values[8]; // d/dz(xyzz) - 2*exponent*xyzz*fz
    gradient_values[39] =
        3 * fx * fz * fz +
        minus_2afz * original_values[9]; // d/dz(xzzz) - 2*exponent*xzzz*fz
    gradient_values[40] =
        minus_2afz * original_values[10]; // d/dz(yyyy) - 2*exponent*yyyy*fz
    gradient_values[41] =
        fy * fy * fy +
        minus_2afz * original_values[11]; // d/dz(yyyz) - 2*exponent*yyyz*fz
    gradient_values[42] =
        2 * fy * fy * fz +
        minus_2afz * original_values[12]; // d/dz(yyzz) - 2*exponent*yyzz*fz
    gradient_values[43] =
        3 * fy * fz * fz +
        minus_2afz * original_values[13]; // d/dz(yzzz) - 2*exponent*yzzz*fz
    gradient_values[44] =
        4 * fz * fz * fz +
        minus_2afz * original_values[14]; // d/dz(zzzz) - 2*exponent*zzzz*fz
  }
}
} // namespace gradient
} // namespace gpu4pyscf::gpbc
