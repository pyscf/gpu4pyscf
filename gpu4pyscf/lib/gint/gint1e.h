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

#define MAX_NROOTS_INT3C_1E 5

// 1 roots upto (p|s)  6    = 3*1*(2*1)
// 2 roots upto (d|p)  36   = 3*2*(3*2)
// 3 roots upto (f|d)  108  = 3*3*(4*3)
// 4 roots upto (g|f)  240  = 3*4*(5*4)
// 5 roots upto (h|g)  450  = 3*5*(6*5)
// 6 roots upto (i|h)  450  = 3*6*(7*6)

#define GSIZE1_INT3C_1E 6
#define GSIZE2_INT3C_1E 36
#define GSIZE3_INT3C_1E 108
#define GSIZE4_INT3C_1E 240
#define GSIZE5_INT3C_1E 450
#define GSIZE6_INT3C_1E 756
