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

import unittest
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf.dft.rks import RKS
from gpu4pyscf.dft.uks import UKS


def numerical_gradient_range_separation_in_kernel(mol, xc, auxbasis, atom_grid, unrestricted=False):
    def get_energy(mol):
        if unrestricted:
            mf = UKS(mol, xc=xc)
        else:
            mf = RKS(mol, xc=xc)

        assert auxbasis is not None
        mf = mf.density_fit(auxbasis=auxbasis)
        mf.range_separated_mode = 'mix_inside_kernel'

        mf.grids.atom_grid = atom_grid
        if unrestricted:
            mf.conv_tol = 1e-10
        else:
            mf.conv_tol = 1e-12

        mf.nlc = None
        mf.disp = False

        e = mf.kernel()
        assert mf.converged
        return e

    numerical_gradient = np.zeros((mol.natm, 3))
    dx = 1e-4
    mol_copy = mol.copy()
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            mol_copy.build()
            E_p = get_energy(mol_copy)

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            E_m = get_energy(mol_copy)

            numerical_gradient[i_atom, i_xyz] = (E_p - E_m) / (2 * dx)

    # np.set_printoptions(precision=16, suppress=True, linewidth=np.inf)
    # print(repr(numerical_gradient))
    return numerical_gradient

def numerical_hessian_range_separation_in_kernel(mol, xc, auxbasis, atom_grid, unrestricted=False):

    def get_gradient(mol):
        if unrestricted:
            mf = UKS(mol, xc=xc)
        else:
            mf = RKS(mol, xc=xc)

        assert auxbasis is not None
        mf = mf.density_fit(auxbasis=auxbasis)
        mf.range_separated_mode = 'mix_inside_kernel'

        mf.grids.atom_grid = atom_grid
        mf.conv_tol = 1e-12

        mf.kernel()
        assert mf.converged

        gobj = mf.Gradients()
        gobj.grid_response = True
        g = gobj.kernel()
        return g

    numerical_hessian = np.zeros((mol.natm, mol.natm, 3, 3))
    dx = 1e-4
    mol_copy = mol.copy()
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            mol_copy.build()
            g_p = get_gradient(mol_copy)

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            g_m = get_gradient(mol_copy)

            numerical_hessian[i_atom, :, i_xyz, :] = (g_p - g_m) / (2 * dx)
    numerical_hessian = numerical_hessian.transpose(0,2,1,3).reshape(mol.natm * 3, mol.natm * 3)

    return numerical_hessian

def setUpModule():
    global mol, mol_unrestricted
    mol = pyscf.M(
        atom = """
            Ti 0.0 0.0 0.0
            F 0.0 0.0 2.0
            F 0.0 2.0 -1.0
            F 1.73 -1.0 -1.1
            F -1.73 -1.0 -1.0
        """,
        basis = 'def2-svp',
        verbose = 0,
    )
    
    mol_unrestricted = pyscf.M(
        atom = """
            Ti 0.0 0.0 0.0
            F 0.0 0.0 2.0
            F 0.0 2.0 -1.0
            F 1.73 -1.0 -1.1
        """,
        basis = 'def2-svp',
        spin = 1,
        verbose = 0,
    )

# def tearDownModule():
#     global mol
#     mol.stdout.close()
#     del mol

class KnownValues(unittest.TestCase):
    def test_range_separation_in_kernel_lr_rks_integrated(self):
        auxbasis = "def2-universal-jkfit"
        xc = "HYB_GGA_XC_LC_WPBE_WHS"

        mf = RKS(mol, xc = xc).density_fit(auxbasis = auxbasis)
        mf.grids.atom_grid = (99, 590)

        mf.range_separated_mode = 'mix_inside_kernel'
        test_energy = mf.kernel()
        assert mf.converged

        assert "omega_0.400000_lr_factor_1.000000_sr_factor_0.000000" in mf.with_df._rsh_df

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        # Reference from mf.range_separated_mode = 'mix_outside_kernel'
        ref_energy = -1247.9255346708449
        ref_gradient = np.array([
            [ 0.0042380836836976, -0.0022423672842287,  0.0538063704048741],
            [-0.0003757872335427,  0.0002212043104924,  0.1135585721838446],
            [ 0.0001604236687629,  0.0984083091471124, -0.0547826700785592],
            [ 0.0811101959883072, -0.0469190666379937, -0.0577157862677815],
            [-0.0851329161072218, -0.0494680795354014, -0.0548664862424371],
        ])

        assert np.max(np.abs(test_energy - ref_energy)) < 1e-6
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    def test_range_separation_in_kernel_sr_rks_integrated(self):
        auxbasis = "def2-universal-jkfit"
        xc = "HSE06"

        mf = RKS(mol, xc = xc).density_fit(auxbasis = auxbasis)
        mf.grids.atom_grid = (50, 194)

        mf.range_separated_mode = 'mix_inside_kernel'
        test_energy = mf.kernel()
        assert mf.converged

        assert "omega_-0.110000_lr_factor_0.000000_sr_factor_0.250000" in mf.with_df._rsh_df

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        # Reference from mf.range_separated_mode = 'mix_outside_kernel'
        ref_energy = -1247.8968925712015
        ref_gradient = np.array([
            [ 0.0043869280652665, -0.0018964494883719,  0.0514691006503987],
            [-0.0003910347987463,  0.0002288583200017,  0.1042818371189327],
            [ 0.0001425905382645,  0.0903931265244982, -0.0510960716149569],
            [ 0.0742957162153495, -0.0430914191462186, -0.0538900599143091],
            [-0.078434200020018 , -0.0456341162098184, -0.0507648062400019],
        ])

        assert np.max(np.abs(test_energy - ref_energy)) < 1e-6
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    def test_range_separation_in_kernel_mr_rks_integrated(self):
        auxbasis = "def2-universal-jkfit"
        xc = "wB97X"

        mf = RKS(mol, xc = xc).density_fit(auxbasis = auxbasis)
        mf.grids.atom_grid = (50, 194)

        mf.range_separated_mode = 'mix_inside_kernel'
        test_energy = mf.kernel()
        assert mf.converged

        assert "omega_-0.300000_lr_factor_1.000000_sr_factor_0.157706" in mf.with_df._rsh_df

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        # Reference from mf.range_separated_mode = 'mix_outside_kernel'
        ref_energy = -1248.3081913421827
        ref_gradient = np.array([
            [ 0.0043093130371917, -0.0019864531112988,  0.0556703302533692],
            [-0.0003666848390124,  0.0002126565898848,  0.1102147141892083],
            [ 0.000168213339775 ,  0.0966682080772774, -0.0543758546877173],
            [ 0.079659894715375 , -0.0461905873867536, -0.057424220541936 ],
            [-0.0837707362533315, -0.0487038241690811, -0.0540849692130809],
        ])

        assert np.max(np.abs(test_energy - ref_energy)) < 1e-6
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    def test_range_separation_in_kernel_mr_rks_numerical_gradient(self):
        auxbasis = "cc-pvtz" # Intentionally using a small auxbasis, so mix_inside_kernel / mix_outside_kernel make a big difference
        xc = "wB97MV"

        mf = RKS(mol, xc = xc).density_fit(auxbasis = auxbasis)
        mf.grids.atom_grid = (50, 194)
        mf.conv_tol = 1e-12

        mf.nlc = None
        mf.disp = False

        mf.range_separated_mode = 'mix_inside_kernel'
        test_energy = mf.kernel()
        assert mf.converged

        assert "omega_-0.300000_lr_factor_1.000000_sr_factor_0.150000" in mf.with_df._rsh_df

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        # Consistency test
        ref_energy = -1251.4674059097918

        # ref_gradient = numerical_gradient_range_separation_in_kernel(mol, xc, auxbasis, (50, 194))
        ref_gradient = np.array([
            [ 0.0049463506002212, -0.0022990855086391,  0.0599723318828183],
            [-0.0011067004379584,  0.0006481718628493,  0.090159866203976 ],
            [-0.000012513510228 ,  0.0791717252468516, -0.0490252000417968],
            [ 0.0649910327865655, -0.0377062087864033, -0.0521606182246614],
            [-0.0688181717123371, -0.0398146232782892, -0.0489463673147839],
        ])

        assert np.max(np.abs(test_energy - ref_energy)) < 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 3e-7

    def test_range_separation_in_kernel_mr_uks_integrated(self):
        auxbasis = "def2-universal-jkfit"
        xc = "wB97X"

        mf = UKS(mol_unrestricted, xc = xc).density_fit(auxbasis = auxbasis)
        mf.grids.atom_grid = (50, 194)
        mf.conv_tol = 1e-10

        mf.range_separated_mode = 'mix_inside_kernel'
        test_energy = mf.kernel()
        assert mf.converged

        assert "omega_-0.300000_lr_factor_1.000000_sr_factor_0.157706" in mf.with_df._rsh_df

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        # Reference from mf.range_separated_mode = 'mix_outside_kernel'
        ref_energy = -1148.6007450576428
        ref_gradient = np.array([
            [-0.0866323962119679, -0.0533703429717916, -0.0033074510159818],
            [ 0.0043754451114529,  0.0030063815508724,  0.0953003352105384],
            [ 0.0060482823841714,  0.0875626574105706, -0.0443885397391881],
            [ 0.0762086687162604, -0.0371986959897157, -0.0476043444552516],
        ])

        assert np.max(np.abs(test_energy - ref_energy)) < 1e-6
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-6

    def test_range_separation_in_kernel_mr_uks_numerical_gradient(self):
        auxbasis = "cc-pvtz" # Intentionally using a small auxbasis, so mix_inside_kernel / mix_outside_kernel make a big difference
        xc = "CAM-B3LYP"

        mf = UKS(mol_unrestricted, xc = xc).density_fit(auxbasis = auxbasis)
        mf.grids.atom_grid = (50, 194)
        mf.conv_tol = 1e-10

        mf.range_separated_mode = 'mix_inside_kernel'
        test_energy = mf.kernel()
        assert mf.converged

        assert "omega_-0.330000_lr_factor_0.650000_sr_factor_0.190000" in mf.with_df._rsh_df

        gobj = mf.Gradients()
        gobj.grid_response = True
        test_gradient = gobj.kernel()

        # Consistency test
        ref_energy = -1150.9832383558733

        # ref_gradient = numerical_gradient_range_separation_in_kernel(mol_unrestricted, xc, auxbasis, (50, 194), unrestricted=True)
        ref_gradient = np.array([
            [-0.0823117920845107, -0.0513560019044235,  0.0005991182661091],
            [ 0.0060785816913267,  0.0048450351641804,  0.0791324328019982],
            [ 0.009713846793602 ,  0.0739035112928832, -0.0381434199425712],
            [ 0.0665193704207923, -0.0273925479632453, -0.0415881322624045],
        ])

        assert np.max(np.abs(test_energy - ref_energy)) < 1e-9
        assert np.max(np.abs(test_gradient - ref_gradient)) < 1e-5

    def test_range_separation_in_kernel_mr_rks_hessian(self):
        mol = pyscf.M(
            atom = """
            O  0.0000  0.7375 -0.0528
            O  0.0000 -0.7375 -0.1528
            H  0.8190  0.8170  0.4220
            H -0.8190 -0.8170  1.4220
            """,
            basis = "def2-svp",
            verbose = 0,
        )
        auxbasis = "cc-pvtz"
        xc = "wB97X"

        mf = RKS(mol, xc = xc).density_fit(auxbasis = auxbasis)
        mf.grids.atom_grid = (50, 194)
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-10
        mf.cphf_grids = mf.grids

        mf.range_separated_mode = "mix_inside_kernel"

        mf.kernel()
        assert mf.converged

        assert "omega_-0.300000_lr_factor_1.000000_sr_factor_0.157706" in mf.with_df._rsh_df

        hobj = mf.Hessian()
        hobj.grid_response = True
        test_hessian = hobj.kernel()
        test_hessian = test_hessian.transpose(0,2,1,3).reshape(mol.natm * 3, mol.natm * 3)

        # ref_hessian = numerical_hessian_range_separation_in_kernel(mol, xc, auxbasis, (50, 194))
        ref_hessian = np.array([
            [ 0.5509458132757494,  0.0474730297561621,  0.3071219088218724, -0.0513918312167183, -0.03480762879704  , -0.0085196768662144, -0.5017177842048248,  0.0014255789787532, -0.301843164290938 ,  0.0021638020708536, -0.0140909799081768,  0.003240932200943 ],
            [ 0.04747309052755  ,  0.2251008043607783, -0.0046046865387339,  0.0195357270066365, -0.164130513846672 ,  0.0105274700312741, -0.0494808552331527, -0.0314290892750035, -0.0341865120323526, -0.0175279623193525, -0.0295412014481022,  0.0282637285392573],
            [ 0.3071219346206799, -0.0046046880441963,  0.1897652104299041, -0.008970725681734 , -0.0038416831227295, -0.0455671698063043, -0.2994124054667147, -0.0034123108283435, -0.1549418173474315,  0.0012611965177767,  0.0118586817834943,  0.0107437767243868],
            [-0.0513917547961817,  0.0195357922994077, -0.0089709294082141,  0.0577366001644486,  0.0106396830190647,  0.036637485163471 ,  0.0067469694320366, -0.0394693205474361,  0.0030095891356829, -0.0130918148627535,  0.0092938453341573, -0.0306761449497817],
            [-0.0348077172995787, -0.1641304989608017, -0.0038417131142943,  0.0106397430457705,  0.2000416303360453,  0.0291022823295517,  0.0069067214791829, -0.0266204029175654,  0.0042467310390482,  0.0172612527737925, -0.0092907286747268, -0.0295073003075963],
            [-0.0085195504384572,  0.0105273969452924, -0.0455672867771817,  0.0366374800017666,  0.029102387690827 ,  0.0149245904645934,  0.0016321661688856, -0.0244915302294313,  0.0037984466161323, -0.0297500957058272, -0.0151382541910272,  0.0268442496009769],
            [-0.5017178134658629, -0.0494808155870885, -0.299412385152964 ,  0.0067470780207302,  0.006906681964125 ,  0.0016322975526784,  0.4906811244453912,  0.0387762623454968,  0.2989806866837696,  0.0042896110480362,  0.0037978709657716, -0.0012005990590591],
            [ 0.0014256003844082, -0.0314290959391172, -0.0034122837000439, -0.0394693084010411, -0.0266203973531276, -0.0244915017200142,  0.0387762678899506,  0.0588968231035603,  0.026855495317557 , -0.0007325598294639, -0.0008473299253908,  0.0010482900092423],
            [-0.3018432236134849, -0.0341865406472408, -0.1549417762780614,  0.0030094167891015,  0.0042467394578694,  0.0037983626921534,  0.2989807551267987,  0.0268555193538855,  0.1537793276129573, -0.0001469484400829,  0.0030842819581656, -0.0026359140420373],
            [ 0.0021637467950697, -0.0175280063352545,  0.0012614020400425, -0.0130918468607688,  0.0172612589910415, -0.0297501082069385,  0.0042896981788942, -0.0007325197559638, -0.0001471059540847,  0.0066384019370425,  0.0009992667832082,  0.0286358120593633],
            [-0.0140909841639392, -0.0295412117612415,  0.0118586798814047,  0.0092938382642571, -0.00929072323963  , -0.0151382528490451,  0.003797875893774 , -0.0008473288148902,  0.0030842904452655,  0.0009992699850914,  0.0396792636933596,  0.000195282497395 ],
            [ 0.0032408423555896,  0.0282638320125983,  0.0107438532304105, -0.0306761708071535, -0.0295074431910791,  0.0268442161222016, -0.0012005193705811,  0.0010483214701873, -0.0026359579585744,  0.0286358478240878,  0.0001952898770474, -0.0349521112957829],
        ])

        assert np.max(np.abs(test_hessian - ref_hessian)) < 3e-7

    def test_range_separation_in_kernel_mr_uks_hessian(self):
        mol = pyscf.M(
            atom = """
            O  0.0000  0.7375 -0.0528
            O  0.0000 -0.7375 -0.1528
            H  0.8190  0.8170  0.4220
            H -0.8190 -0.8170  1.4220
            """,
            basis = "def2-svp",
            verbose = 0,
        )
        auxbasis = "cc-pvtz"
        xc = "wB97X"

        mf = UKS(mol, xc = xc).density_fit(auxbasis = auxbasis)
        mf.grids.atom_grid = (50, 194)
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-10
        mf.cphf_grids = mf.grids

        mf.range_separated_mode = "mix_inside_kernel"

        mf.kernel()
        assert mf.converged

        assert "omega_-0.300000_lr_factor_1.000000_sr_factor_0.157706" in mf.with_df._rsh_df

        hobj = mf.Hessian()
        hobj.grid_response = True
        test_hessian = hobj.kernel()
        test_hessian = test_hessian.transpose(0,2,1,3).reshape(mol.natm * 3, mol.natm * 3)

        # ref_hessian = numerical_hessian_range_separation_in_kernel(mol, xc, auxbasis, (50, 194))
        ref_hessian = np.array([
            [ 0.5509458132757494,  0.0474730297561621,  0.3071219088218724, -0.0513918312167183, -0.03480762879704  , -0.0085196768662144, -0.5017177842048248,  0.0014255789787532, -0.301843164290938 ,  0.0021638020708536, -0.0140909799081768,  0.003240932200943 ],
            [ 0.04747309052755  ,  0.2251008043607783, -0.0046046865387339,  0.0195357270066365, -0.164130513846672 ,  0.0105274700312741, -0.0494808552331527, -0.0314290892750035, -0.0341865120323526, -0.0175279623193525, -0.0295412014481022,  0.0282637285392573],
            [ 0.3071219346206799, -0.0046046880441963,  0.1897652104299041, -0.008970725681734 , -0.0038416831227295, -0.0455671698063043, -0.2994124054667147, -0.0034123108283435, -0.1549418173474315,  0.0012611965177767,  0.0118586817834943,  0.0107437767243868],
            [-0.0513917547961817,  0.0195357922994077, -0.0089709294082141,  0.0577366001644486,  0.0106396830190647,  0.036637485163471 ,  0.0067469694320366, -0.0394693205474361,  0.0030095891356829, -0.0130918148627535,  0.0092938453341573, -0.0306761449497817],
            [-0.0348077172995787, -0.1641304989608017, -0.0038417131142943,  0.0106397430457705,  0.2000416303360453,  0.0291022823295517,  0.0069067214791829, -0.0266204029175654,  0.0042467310390482,  0.0172612527737925, -0.0092907286747268, -0.0295073003075963],
            [-0.0085195504384572,  0.0105273969452924, -0.0455672867771817,  0.0366374800017666,  0.029102387690827 ,  0.0149245904645934,  0.0016321661688856, -0.0244915302294313,  0.0037984466161323, -0.0297500957058272, -0.0151382541910272,  0.0268442496009769],
            [-0.5017178134658629, -0.0494808155870885, -0.299412385152964 ,  0.0067470780207302,  0.006906681964125 ,  0.0016322975526784,  0.4906811244453912,  0.0387762623454968,  0.2989806866837696,  0.0042896110480362,  0.0037978709657716, -0.0012005990590591],
            [ 0.0014256003844082, -0.0314290959391172, -0.0034122837000439, -0.0394693084010411, -0.0266203973531276, -0.0244915017200142,  0.0387762678899506,  0.0588968231035603,  0.026855495317557 , -0.0007325598294639, -0.0008473299253908,  0.0010482900092423],
            [-0.3018432236134849, -0.0341865406472408, -0.1549417762780614,  0.0030094167891015,  0.0042467394578694,  0.0037983626921534,  0.2989807551267987,  0.0268555193538855,  0.1537793276129573, -0.0001469484400829,  0.0030842819581656, -0.0026359140420373],
            [ 0.0021637467950697, -0.0175280063352545,  0.0012614020400425, -0.0130918468607688,  0.0172612589910415, -0.0297501082069385,  0.0042896981788942, -0.0007325197559638, -0.0001471059540847,  0.0066384019370425,  0.0009992667832082,  0.0286358120593633],
            [-0.0140909841639392, -0.0295412117612415,  0.0118586798814047,  0.0092938382642571, -0.00929072323963  , -0.0151382528490451,  0.003797875893774 , -0.0008473288148902,  0.0030842904452655,  0.0009992699850914,  0.0396792636933596,  0.000195282497395 ],
            [ 0.0032408423555896,  0.0282638320125983,  0.0107438532304105, -0.0306761708071535, -0.0295074431910791,  0.0268442161222016, -0.0012005193705811,  0.0010483214701873, -0.0026359579585744,  0.0286358478240878,  0.0001952898770474, -0.0349521112957829],
        ])

        assert np.max(np.abs(test_hessian - ref_hessian)) < 3e-7


if __name__ == "__main__":
    print("Full Tests for DF JK")
    unittest.main()
