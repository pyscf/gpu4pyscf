# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
