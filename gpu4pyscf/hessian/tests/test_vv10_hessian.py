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
import pyscf
from gpu4pyscf.dft import rks
from gpu4pyscf.hessian.rks import _get_vnlc_deriv1, _get_vnlc_deriv1_numerical, \
                                  _get_enlc_deriv2, _get_enlc_deriv2_numerical

def setUpModule():
    global mol

    atom = '''
    O  0.0000  0.7375 -0.0528
    O  0.0000 -0.7375 -0.1528
    H  0.8190  0.8170  0.4220
    H -0.8190 -0.8170  1.4220
    '''
    basis = 'def2-svp'

    mol = pyscf.M(atom=atom, basis=basis, max_memory=32000,
                  output='/dev/null', verbose=1)

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def make_mf(mol, nlcgrid = (75,302), vv10_only = False, density_fitting = False):
    # Note: (75, 302) nlc grid is required to reduce error in de2 below 1e-5
    if not vv10_only:
        mf = rks.RKS(mol, xc = "wb97x-v")
        mf.grids.level = 5
    else:
        mf = rks.RKS(mol, xc = "0*PBE,0*PBE")
        mf.nlc = "vv10"
        mf.grids.atom_grid = (3,6)
    mf.conv_tol = 1e-15
    mf.direct_scf_tol = 1e-16
    mf.nlcgrids.atom_grid = nlcgrid
    mf.conv_tol_cpscf = 1e-10
    if density_fitting:
        mf = mf.density_fit(auxbasis = "def2-universal-jkfit")
    mf.kernel()
    assert mf.converged
    return mf

def numerical_d2enlc(mf):
    mol = mf.mol

    numerical_hessian = np.zeros([mol.natm, mol.natm, 3, 3])

    dx = 1e-3
    mol_copy = mol.copy()
    mf_copy = mf.copy()
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            mol_copy.build()
            mf_copy.reset(mol_copy)
            mf_copy.kernel()
            assert mf_copy.converged
            grad_obj = mf_copy.Gradients()
            grad_obj.grid_response = True
            gradient_p = grad_obj.kernel()

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            mf_copy.reset(mol_copy)
            mf_copy.kernel()
            assert mf_copy.converged
            grad_obj = mf_copy.Gradients()
            grad_obj.grid_response = True
            gradient_m = grad_obj.kernel()

            numerical_hessian[i_atom, :, i_xyz, :] = (gradient_p - gradient_m) / (2 * dx)

    np.set_printoptions(linewidth = np.iinfo(np.int32).max, threshold = np.iinfo(np.int32).max, precision = 16, suppress = True)
    print(repr(numerical_hessian))
    return numerical_hessian

def analytical_d2enlc(mf):
    hess_obj = mf.Hessian()
    hess_obj.auxbasis_response = 2
    analytical_hessian = hess_obj.kernel()
    return analytical_hessian

class KnownValues(unittest.TestCase):
    def test_vv10_only_hessian_direct(self):
        mf = make_mf(mol, vv10_only = True)

        # reference_hessian = numerical_d2enlc(mf)
        reference_hessian = np.array([[[[ 0.5416385094555443,  0.0608587976822506,  0.4059780361467813],
         [ 0.0608583153386411,  0.2400708605971857,  0.0171074122129466],
         [ 0.4059767934618819,  0.0171075175856572,  0.0715012127059378]],

        [[ 0.0138411261745644, -0.0307587099088735, -0.0114409966949225],
         [-0.0046670364924131, -0.3662177810292988, -0.0328075691992114],
         [-0.0077329455397644, -0.0418507928570122,  0.0212319435189956]],

        [[-0.5632752360931192, -0.0127530395704345, -0.4031497682549512],
         [-0.0421637920360318,  0.1460901948911464, -0.0167164189056601],
         [-0.4027394645520488, -0.0044779744087786, -0.0910390554401674]],

        [[ 0.0077956004770896, -0.0173470482155158,  0.008612728790991 ],
         [-0.0140274868233869, -0.0199432745474626,  0.0324165758873729],
         [ 0.004495616644451 ,  0.02922124965829  , -0.0016941007819904]]],


       [[[ 0.0138412926078413, -0.0046674378371137, -0.007733128939813 ],
         [-0.0307585783373421, -0.3662178497929602, -0.0418510083978196],
         [-0.0114406839240022, -0.0328073191777634,  0.0212322387688757]],

        [[-0.0265688329405926,  0.0280390994733537,  0.0028061508691168],
         [ 0.0280390210395422,  0.4299653153587712,  0.0711125396311019],
         [ 0.0028060484083409,  0.0711122337611059, -0.0189041069124096]],

        [[ 0.0215131863576801, -0.0311007717287426,  0.0034433470949002],
         [-0.0042374598432371, -0.0574409077624405, -0.0113070426581707],
         [ 0.0060660526035594, -0.0238224669175113,  0.0092249375318598]],

        [[-0.0087856460534996,  0.0077291100670229,  0.0014836309728539],
         [ 0.0069570171381539, -0.0063065579256061, -0.017954488580163 ],
         [ 0.0025685829361799, -0.0144824477082139, -0.0115530693778898]]],


       [[[-0.5632753518693967, -0.0421644346566552, -0.402739948897668 ],
         [-0.0127529019099404,  0.1460905464591988, -0.0044775873883074],
         [-0.4031484872100144, -0.0167171608183025, -0.091038376287822 ]],

        [[ 0.0215132045501172, -0.0042371650881279,  0.0060659394314211],
         [-0.0311010481525466, -0.0574412135865288, -0.0238223321953335],
         [ 0.0034430790948267, -0.0113066360620806,  0.0092245405951541]],

        [[ 0.5449169029909662,  0.0467555396718167,  0.3963758752382196],
         [ 0.046755349856431 , -0.0877756999153601,  0.0269887824478898],
         [ 0.396375146332506 ,  0.0269891017112278,  0.0821096295715584]],

        [[-0.0031547556844091, -0.000353939952541 ,  0.0002981342411279],
         [-0.0029013998091298, -0.0008736328979408,  0.0013111371459651],
         [ 0.0033302617750142,  0.0010346951694329, -0.0002957938823878]]],


       [[[ 0.0077954733069818, -0.0140276105691228,  0.0044955368216359],
         [-0.0173472062745539, -0.0199430947489532,  0.0292210259493775],
         [ 0.0086125911367141,  0.0324162169498265, -0.0016944729875901]],

        [[-0.0087853560465749,  0.0069573606920059,  0.0025681631434793],
         [ 0.0077291657290882, -0.0063068020379475, -0.0144823954941753],
         [ 0.0014838511660648, -0.017954068078474 , -0.011553030224154 ]],

        [[-0.0031549510246531, -0.0029016744511612,  0.0033300169713923],
         [-0.0003539166857358, -0.0008736721611724,  0.0010348932112381],
         [ 0.0002979025925942,  0.0013112596001785, -0.0002961460349171]],

        [[ 0.0041448337520511,  0.009971924351676 , -0.0103937169317336],
         [ 0.0099719572323465,  0.027123568895393 , -0.0157735236657186],
         [-0.0103943449092925, -0.015773408461317 ,  0.013543649255765 ]]]])

        test_hessian = analytical_d2enlc(mf)

        assert np.linalg.norm(test_hessian - reference_hessian) < 1e-5

    def test_vv10_only_hessian_density_fitting(self):
        mf = make_mf(mol, vv10_only = True, density_fitting = True)

        # reference_hessian = numerical_d2enlc(mf)
        reference_hessian = np.array([[[[ 0.5415690822132557,  0.0608562722286266,  0.4059126705860394],
         [ 0.0608570487260485,  0.2400788616032656,  0.0171129679309989],
         [ 0.4059109970324659,  0.0171147380978454,  0.0714620692536805]],

        [[ 0.0138241297472988, -0.0307645192521022, -0.011444424837026 ],
         [-0.0046682567246444, -0.3662416225749254, -0.0328128942426176],
         [-0.0077385738328807, -0.0418744339949484,  0.0212219265202096]],

        [[-0.5631978607103516, -0.0127426783599893, -0.4030859833951128],
         [-0.0421540206807514,  0.146110122013654 , -0.0167203915124592],
         [-0.4026860393344656, -0.0044790556974483, -0.090979895049581 ]],

        [[ 0.0078046487382299, -0.0173490746282756,  0.008617737641714 ],
         [-0.0140347713284972, -0.0199473610911771,  0.0324203178236893],
         [ 0.0045136161368475,  0.0292387516011017, -0.0017041007286944]]],


       [[[ 0.0138236935279812, -0.0046692417816629, -0.0077378711652587],
         [-0.0307656303105697, -0.3662417637855242, -0.0418725459100933],
         [-0.0114433360939303, -0.032812823865136 ,  0.0212207692167898]],

        [[-0.0265537883114599,  0.0280364159861435,  0.0027999313607641],
         [ 0.0280356838286977,  0.4300057094841492,  0.0711167561483483],
         [ 0.0028008877173101,  0.0711167590345951, -0.0188834771153168]],

        [[ 0.0215198898387836, -0.031100138694895 ,  0.0034520875638044],
         [-0.0042300795324302, -0.0574485161345395, -0.011293806742918 ],
         [ 0.006071742150171 , -0.0238195073950509,  0.0092259066575284]],

        [[-0.0087897950546978,  0.0077329644918023,  0.0014858522405237],
         [ 0.0069600260220737, -0.0063154296129908, -0.017950403495004 ],
         [ 0.0025707062291658, -0.0144844277661926, -0.0115631987543385]]],


       [[[-0.5631981291014387, -0.0421528096943291, -0.4026862782104956],
         [-0.0127431449786775,  0.1461110761713513, -0.0044783628956324],
         [-0.4030846823723788, -0.0167210497092896, -0.0909792552441502]],

        [[ 0.0215192912686873, -0.0042311047909749,  0.0060719560596167],
         [-0.0311007162406632, -0.0574491767011409, -0.0238194060092622],
         [ 0.0034516618868906, -0.0112928265325607,  0.0092247984134763]],

        [[ 0.5448428814074369,  0.0467446440063357,  0.3963146766036152],
         [ 0.0467451058514534, -0.0877858009699084,  0.0269845353028098],
         [ 0.3963148097061442,  0.0269848332785649,  0.0820484189942849]],

        [[-0.0031640435635971, -0.0003607295108454,  0.0002996455498172],
         [-0.0029012446319254, -0.0008760985011347,  0.0013132336018629],
         [ 0.0033182107773144,  0.0010290429619253, -0.0002939621596143]]],


       [[[ 0.0078054889197654, -0.0140356641091799,  0.0045126400246009],
         [-0.0173492152711896, -0.0199454559561829,  0.0292383102136751],
         [ 0.0086169191971797,  0.0324203680515112, -0.0017032086265245]],

        [[-0.0087886606801521,  0.0069605205768042,  0.0025708432059846],
         [ 0.0077325109240495, -0.0063174013167355, -0.014483500062612 ],
         [ 0.0014854317597936, -0.0179506621575953, -0.0115629838429721]],

        [[-0.0031656710521855, -0.0029017588178415,  0.0033178427655267],
         [-0.0003608430507729, -0.0008764673800621,  0.0010296243039276],
         [ 0.0002997056496312,  0.0013136119419999, -0.0002945709423052]],

        [[ 0.0041488428184633,  0.009976902353992 , -0.0104013259940028],
         [ 0.0099775473980102,  0.0271393246558116, -0.0157844344571556],
         [-0.0104020566066565, -0.015783317837248 ,  0.0135607634125789]]]])

        test_hessian = analytical_d2enlc(mf)

        assert np.linalg.norm(test_hessian - reference_hessian) < 2e-5

    def test_vv10_energy_second_derivative(self):
        mf = make_mf(mol, vv10_only = True, density_fitting = True)
        hess_obj = mf.Hessian()

        reference_de2 = _get_enlc_deriv2_numerical(hess_obj, mf.mo_coeff, mf.mo_occ, max_memory = None)
        test_de2 = _get_enlc_deriv2(hess_obj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert np.linalg.norm(test_de2 - reference_de2) < 1e-5

    def test_vv10_fock_first_derivative(self):
        mf = make_mf(mol, vv10_only = True, density_fitting = True, nlcgrid = (10,14))
        hess_obj = mf.Hessian()

        reference_dF = _get_vnlc_deriv1_numerical(hess_obj, mf.mo_coeff, mf.mo_occ, max_memory = None)
        test_dF = _get_vnlc_deriv1(hess_obj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert np.linalg.norm(test_dF - reference_dF) < 1e-8


    # # TODO: Supress the diff between analytical and numerical hessian below 1e-3
    # def test_wb97xv_hessian_loose_grid(self):
    #     mf = make_mf(mol, nlc_atom_grid_loose, vv10_only = False)

    #     reference_hessian = numerical_d2enlc(mf)

    #     test_hessian = analytical_d2enlc(mf)

    #     assert np.linalg.norm(test_hessian - reference_hessian) < 1e-15

if __name__ == "__main__":
    print("Full Tests for RKS Hessian with VV10")
    unittest.main()
