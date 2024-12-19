/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
