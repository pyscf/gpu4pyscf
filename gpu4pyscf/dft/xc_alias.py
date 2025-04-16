# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

map = {
    'LDA'           : 'LDA_X,LDA_C_VWN',
    'PBE'           : 'GGA_X_PBE,GGA_C_PBE',
    'M06'           : 'HYB_MGGA_X_M06,MGGA_C_M06',
    'B3LYP'         : 'HYB_GGA_XC_B3LYP',
    'CAMB3LYP'      : 'HYB_GGA_XC_CAM_B3LYP',
    'CAMYBLYP'      : 'HYB_GGA_XC_CAMY_BLYP',
    'CAMYB3LYP'     : 'HYB_GGA_XC_CAMY_B3LYP',
    'PBE0'          : 'HYB_GGA_XC_PBEH',
    'WB97'          : 'HYB_GGA_XC_WB97',
    'WB97X'         : 'HYB_GGA_XC_WB97X',
}
