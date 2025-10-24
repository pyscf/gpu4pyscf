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

import pyscf
import numpy as np
import cupy as cp
import unittest
import pytest
from gpu4pyscf.dft import RKS
from gpu4pyscf.hessian.rks import _get_exc_deriv2, _get_vxc_deriv1
from gpu4pyscf.hessian.tests.test_vv10_hessian import numerical_d2e_dft

def setUpModule():
    global mol

    mol = pyscf.M(
        atom = '''
            O  0.0000  0.7375 -0.0528
            O  0.0000 -0.7375 -0.1528
            H  0.8190  0.8170  0.4220
            H -0.8190 -0.8170  1.4220
        ''',
        basis = 'def2-svp',
        charge = 0,
        spin = 0,
        output='/dev/null',
        verbose = 0,
    )

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _get_exc_deriv2_numerical(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Attention: Numerical xc energy 2nd derivative includes grid response.
    """
    mol = hessobj.mol
    mf = hessobj.base
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = np.dot(mocc, mocc.T) * 2

    de2 = cp.empty([mol.natm, mol.natm, 3, 3])

    def get_xc_de(grad_obj, dm):
        assert grad_obj.grid_response
        from gpu4pyscf.grad.rks import get_exc_full_response
        mol = grad_obj.mol
        ni = mf._numint
        mf.grids.build()
        exc_grid, exc_orbital = get_exc_full_response(ni, mol, mf.grids, mf.xc, dm)

        aoslices = mol.aoslice_by_atom()
        exc_orbital = [exc_orbital[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]]
        exc_orbital = cp.asarray(exc_orbital)
        de = 2 * exc_orbital + exc_grid
        return de

    dx = 1e-5
    mol_copy = mol.copy()
    grad_obj = mf.Gradients()
    grad_obj.grid_response = True

    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            grad_obj.reset(mol_copy)
            de_p = get_xc_de(grad_obj, dm0)

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            grad_obj.reset(mol_copy)
            de_m = get_xc_de(grad_obj, dm0)

            de2[i_atom, :, i_xyz, :] = (de_p - de_m) / (2 * dx)
    grad_obj.reset(mol)

    return de2

def _get_vxc_deriv1_numerical(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Attention: Numerical xc Fock matrix 1st derivative includes grid response.
    """
    mol = hessobj.mol
    mf = hessobj.base
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = np.dot(mocc, mocc.T) * 2

    nao = mol.nao
    vmat = cp.empty([mol.natm, 3, nao, nao])

    def get_vxc_vmat(mol, mf, dm):
        ni = mf._numint
        mf.grids.build()
        n, exc, vxc = ni.nr_rks(mol, mf.grids, mf.xc, dm)
        return vxc

    dx = 1e-5
    mol_copy = mol.copy()
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            vmat_p = get_vxc_vmat(mol_copy, mf, dm0)

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            vmat_m = get_vxc_vmat(mol_copy, mf, dm0)

            vmat[i_atom, i_xyz, :, :] = (vmat_p - vmat_m) / (2 * dx)
    mf.reset(mol)

    vmat = cp.einsum('Adij,jq->Adiq', vmat, mocc)
    vmat = cp.einsum('Adiq,ip->Adpq', vmat, mo_coeff)
    return vmat

class KnownValues(unittest.TestCase):
    # All reference results from the same calculation with mf.level_shift = 0

    def test_hessian_grid_response_d2edAdB_lda(self):
        mf = RKS(mol, xc = 'LDA')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_de2 = _get_exc_deriv2(hobj, mf.mo_coeff, mf.mo_occ, mf.make_rdm1(), max_memory = None)
        reference_de2 = _get_exc_deriv2_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert cp.max(cp.abs(test_de2 - reference_de2)) < 1e-8

    def test_hessian_grid_response_d2edAdB_gga(self):
        mf = RKS(mol, xc = 'PBE0')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        # hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_de2 = _get_exc_deriv2(hobj, mf.mo_coeff, mf.mo_occ, mf.make_rdm1(), max_memory = None)
        reference_de2 = _get_exc_deriv2_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert cp.max(cp.abs(test_de2 - reference_de2)) < 1e-8

    def test_hessian_grid_response_d2edAdB_mgga(self):
        mf = RKS(mol, xc = 'wB97M-d3bj')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        # hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_de2 = _get_exc_deriv2(hobj, mf.mo_coeff, mf.mo_occ, mf.make_rdm1(), max_memory = None)
        reference_de2 = _get_exc_deriv2_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert cp.max(cp.abs(test_de2 - reference_de2)) < 1e-8

    def test_hessian_grid_response_dFdA_lda(self):
        mf = RKS(mol, xc = 'LDA')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        # hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_dF = _get_vxc_deriv1(hobj, mf.mo_coeff, mf.mo_occ, max_memory = 16000)
        reference_dF = _get_vxc_deriv1_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert cp.max(cp.abs(test_dF - reference_dF)) < 1e-8

    def test_hessian_grid_response_dFdA_gga(self):
        mf = RKS(mol, xc = 'wB97X-V')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_dF = _get_vxc_deriv1(hobj, mf.mo_coeff, mf.mo_occ, max_memory = 16000)
        reference_dF = _get_vxc_deriv1_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert cp.max(cp.abs(test_dF - reference_dF)) < 1e-8

    def test_hessian_grid_response_dFdA_mgga(self):
        mf = RKS(mol, xc = 'r2SCAN')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-8
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        # hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_dF = _get_vxc_deriv1(hobj, mf.mo_coeff, mf.mo_occ, max_memory = 16000)
        reference_dF = _get_vxc_deriv1_numerical(hobj, mf.mo_coeff, mf.mo_occ, max_memory = None)

        assert cp.max(cp.abs(test_dF - reference_dF)) < 1e-8

    def test_hessian_grid_response_lda(self):
        mf = RKS(mol, xc = 'LDA')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-10
        mf.cphf_grids.atom_grid = mf.grids.atom_grid
        mf.cphf_grids.prune = mf.grids.prune
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_hessian = hobj.kernel()

        # reference_hessian = numerical_d2e_dft(mf, dx = 1e-3)
        reference_hessian = np.array([[[[ 0.5300145120906707,  0.0777596822709725,  0.3665658962896945],
         [ 0.0777563685916416,  0.2920108787387576,  0.0830531836397697],
         [ 0.3665654818175712,  0.0830529240705147,  0.0967252805517682]],

        [[ 0.0668516887880333, -0.0298342288900244, -0.0043356571765019],
         [ 0.0358785655379737, -0.2049812145044072, -0.0670380166744033],
         [ 0.0016511523578122, -0.0886795037020605,  0.0376162179960282]],

        [[-0.6023943276798338, -0.0313793527272344, -0.3702636214316657],
         [-0.0995928308185068, -0.0583886149599921, -0.0428921237065616],
         [-0.365959444827757 , -0.0154012654027813, -0.1393131249710633]],

        [[ 0.0055281267831719, -0.016546100650966 ,  0.008033382313144 ],
         [-0.0140421033134608, -0.0286410493317291,  0.0268769567418614],
         [-0.0022571893532608,  0.0210278450393231,  0.0049716264134414]]],


       [[[ 0.0668518257587181,  0.0358783542449004,  0.0016508520321867],
         [-0.0298298475999159, -0.2049835584561066, -0.0886799470170008],
         [-0.0043366237366671, -0.0670371891233756,  0.0376165283992314]],

        [[-0.0504421692882201, -0.0032797850062494,  0.0364714100389296],
         [-0.0032797183494027,  0.2433034644360177,  0.1140643008193942],
         [ 0.0364714595771226,  0.114063862174163 , -0.0796519803036855]],

        [[-0.0036986698201957, -0.0458507826046062, -0.0000395054270408],
         [ 0.0188158227185653, -0.0298178521229708,  0.0063294358211285],
         [ 0.0022551772826951, -0.0210526596703398,  0.002194011898049 ]],

        [[-0.0127109866494557,  0.0132522133688417, -0.0380827566472397],
         [ 0.0142937432346391, -0.0085020539162262, -0.0317137896215236],
         [-0.0343900131212216, -0.0259740133834452,  0.0398414400167302]]],


       [[[-0.6023954582076163, -0.0995920738118272, -0.36595974336745  ],
         [-0.0313789150582222, -0.058388824444755 , -0.0154011387394348],
         [-0.370262666572474 , -0.0428927146893798, -0.139312836719363 ]],

        [[-0.0036988489912418,  0.0188155782074872,  0.002255305804999 ],
         [-0.0458506344111209, -0.0298172068138314, -0.0210528054814807],
         [-0.0000395144680101,  0.0063308978814902,  0.0021938074749039]],

        [[ 0.6073372588613069,  0.0805659508398837,  0.3652696124576416],
         [ 0.0805654371127673,  0.089992314623033 ,  0.0333669072410947],
         [ 0.3652683489745456,  0.0333670365380567,  0.1404157077209423]],

        [[-0.0012429516495494,  0.0002105447649281, -0.0015651748868084],
         [-0.0033358876397327, -0.0017862833597004,  0.0030870369805425],
         [ 0.0050338320656818,  0.0031947802703325, -0.0032966784717647]]],


       [[[ 0.0055278852058027, -0.0140437958853212, -0.0022572300038548],
         [-0.0165453108372793, -0.0286419147688122,  0.0210282274152873],
         [ 0.0080338595591645,  0.0268773435569969,  0.0049711930832919]],

        [[-0.0127104395919383,  0.0142963100673299, -0.0343903096999831],
         [ 0.0132527521485154, -0.0085003390175586, -0.0259747051150061],
         [-0.0380831138059035, -0.0317154640967487,  0.0398423013810989]],

        [[-0.0012435772682728, -0.0033359406892419,  0.0050338191734944],
         [ 0.0002095636775223, -0.001786272822657 ,  0.0031943357824415],
         [-0.0015641742898698,  0.0030867957651615, -0.0032968165593061]],

        [[ 0.0084261316563583,  0.0030834265106472,  0.0316137205341738],
         [ 0.0030829950150579,  0.0389285266058914,  0.0017521419131694],
         [ 0.0316134285370251,  0.0017513247679568, -0.0415166779059728]]]])

        assert np.max(np.abs(test_hessian - reference_hessian)) < 1e-5
        # Translation invariance
        assert np.max(np.abs(np.sum(test_hessian, axis = 0))) < 1e-8

    def test_hessian_grid_response_gga(self):
        mf = RKS(mol, xc = 'revPBE')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-10
        mf.cphf_grids.atom_grid = mf.grids.atom_grid
        mf.cphf_grids.prune = mf.grids.prune
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_hessian = hobj.kernel()

        # reference_hessian = numerical_d2e_dft(mf, dx = 1e-3)
        reference_hessian = np.array([[[[ 0.5211014689474602,  0.0818298772289339,  0.3621828758923473],
         [ 0.0818311071932865,  0.2759328201253908,  0.0908171756636711],
         [ 0.3621832871225639,  0.090809563985772 ,  0.0705164815708592]],

        [[ 0.0898246311866885, -0.0296462281372811, -0.0054553505044463],
         [ 0.0405698308465277, -0.1856630985441754, -0.0784293992189822],
         [-0.0000654860216262, -0.1041209538472643,  0.0550392821468959]],

        [[-0.6166737276873668, -0.0340542513423969, -0.3656905139206579],
         [-0.1082741376279284, -0.0613860274405464, -0.0402416258413751],
         [-0.3604594321260368, -0.0112974074741867, -0.1281067368018585]],

        [[ 0.0057476275753809, -0.018129397787725 ,  0.0089629885290377],
         [-0.0141268003484019, -0.0288836942885229,  0.0278538493955205],
         [-0.0016583690003458,  0.0246087973089226,  0.0025509730809947]]],


       [[[ 0.0898264880393462,  0.0405649275583286, -0.0000685941623502],
         [-0.0296488440867382, -0.1856636088168884, -0.1041307104721612],
         [-0.0054457202771641, -0.0784381216751839,  0.0550441272768598]],

        [[-0.0772733862538094, -0.0071227092863779,  0.0427548843626724],
         [-0.0071301942856303,  0.2202772874149161,  0.1294509513147801],
         [ 0.0427577878222632,  0.1294612553870422, -0.1062307799845064]],

        [[-0.0033858607684234, -0.0484852636225552,  0.0015125636227964],
         [ 0.0219683583189134, -0.0254848048363288,  0.006864815660923 ],
         [ 0.0027771437787205, -0.0219385867513133,  0.003282521427117 ]],

        [[-0.0091672410530985,  0.0150430453040862, -0.0441988537959181],
         [ 0.0148106800885728, -0.0091288738047757, -0.0321850564984905],
         [-0.0400892113225915, -0.0290845469637646,  0.0479041312813622]]],


       [[[-0.6166717548808931, -0.1082764684294801, -0.3604581354346714],
         [-0.0340526768404237, -0.0613889838296799, -0.0112968603364072],
         [-0.36569306083023  , -0.040234869585376 , -0.1281070430000919]],

        [[-0.0033836163325293,  0.0219689395795086,  0.0027836794078695],
         [-0.048483680105238 , -0.0254816540827818, -0.0219386176047998],
         [ 0.0015110900341356,  0.0068554070962179,  0.0032895326101956]],

        [[ 0.6215155015985019,  0.0861001048482746,  0.3587131812340383],
         [ 0.0860981713861619,  0.0887184980972178,  0.0299558810533673],
         [ 0.3587164896459161,  0.0299571003006815,  0.1284579999493163]],

        [[-0.001460130378772 ,  0.0002074242383965, -0.0010387252236121],
         [-0.0035618144355598, -0.001847860139681 ,  0.0032795969322486],
         [ 0.0054654811387223,  0.0034223621695195, -0.0036404895737974]]],


       [[[ 0.0057604990114646, -0.0141236661237443, -0.0016510380347334],
         [-0.0181293476040345, -0.0288743403338287,  0.0246114115399765],
         [ 0.0089432854565308,  0.0278602949803641,  0.0025454978591077]],

        [[-0.0091650926198686,  0.0148087181726098, -0.040089004358701 ],
         [ 0.015042775963349 , -0.0091398518842212, -0.0290818846693242],
         [-0.044206120694737 , -0.0321961173570529,  0.0479082065766301]],

        [[-0.0014689455949402, -0.0035609064124587,  0.0054572821501786],
         [ 0.0002079902221563, -0.0018468557416162,  0.0034214693209744],
         [-0.0010231749383305,  0.0032788639754422, -0.003628871325545 ]],

        [[ 0.0048735391709465,  0.0028758544779739,  0.0362827602617966],
         [ 0.0028785814333854,  0.0398610479943051,  0.0010490038057087],
         [ 0.0362860101758566,  0.0010569583545339, -0.0468248330827148]]]])

        assert np.max(np.abs(test_hessian - reference_hessian)) < 1e-4
        # Translation invariance
        assert np.max(np.abs(np.sum(test_hessian, axis = 0))) < 1e-10

    def test_hessian_grid_response_mgga(self):
        mf = RKS(mol, xc = 'r2SCAN')
        mf.grids.atom_grid = (10,14)
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-10
        mf.cphf_grids.atom_grid = mf.grids.atom_grid
        mf.cphf_grids.prune = mf.grids.prune
        # mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_hessian = hobj.kernel()

        # reference_hessian = numerical_d2e_dft(mf, dx = 1e-3)
        reference_hessian = np.array([[[[ 0.523734706295631 ,  0.0753789823830786,  0.3793649978935942],
         [ 0.0753762770011601,  0.3761594584155148,  0.1084622374811772],
         [ 0.3793634383625344,  0.1084631234125055,  0.0961344047761825]],

        [[ 0.0907561758982448, -0.0288877211866989, -0.0062947318258955],
         [ 0.0466553225219046, -0.2363220833307977, -0.0841542128825479],
         [ 0.0009854169533349, -0.108869971646719 ,  0.0530936135297866]],

        [[-0.6196611118669537, -0.0284265817212526, -0.3817086460535801],
         [-0.1077274888869884, -0.1099936813466207, -0.0514502540196471],
         [-0.3779280116542605, -0.0213611112894752, -0.1544327194060724]],

        [[ 0.0051702296344214, -0.0180646794542827,  0.0086383799746126],
         [-0.0143041106742681, -0.0298436937953006,  0.0271422294368939],
         [-0.0024208437155449,  0.02176795963163  ,  0.0052047011241396]]],


       [[[ 0.0907596009603884,  0.046655958800379 ,  0.0009853439373875],
         [-0.028889606052962 , -0.2363171114190266, -0.108870066500677 ],
         [-0.0062881925024794, -0.0841633060373326,  0.0530954092212155]],

        [[-0.0703104504681251, -0.0104292962914698,  0.0443349031923335],
         [-0.010431751466207 ,  0.2691507548782113,  0.1339028472002735],
         [ 0.0443363957293441,  0.1339107285804886, -0.101393912790293 ]],

        [[-0.0074468736492861, -0.0513211383551537, -0.0000511330627839],
         [ 0.0239894623812464, -0.0235383320353888,  0.0085742039762637],
         [ 0.0027850558008691, -0.0213902204554417,  0.0004956117096722]],

        [[-0.0130022768034532,  0.0150944759143568, -0.0452691140513939],
         [ 0.0153318951261472, -0.0092953115116146, -0.033606984719603 ],
         [-0.0408332590360883, -0.0283572020258749,  0.0478028918643458]]],


       [[[-0.6196570434079396, -0.1077335861987549, -0.377929896557494 ],
         [-0.0284249016471172, -0.1099977109202399, -0.0213614520587213],
         [-0.3817077669772129, -0.0514508339826136, -0.154433883237437 ]],

        [[-0.0074432366331256,  0.0239927749650093,  0.002789353671373 ],
         [-0.0513205945155276, -0.0235340749110691, -0.0213901879971834],
         [-0.0000516546919752,  0.0085720210956097,  0.0005011542708599]],

        [[ 0.6284243417404856,  0.083453333562189 ,  0.3765973741294282],
         [ 0.0834517332783946,  0.1354568434271397,  0.0394447353755378],
         [ 0.3765980162564464,  0.0394454823487433,  0.1575935306153964]],

        [[-0.0013240616982824,  0.0002874776059147, -0.0014568312558527],
         [-0.003706237127088 , -0.0019250575582497,  0.003306904708511 ],
         [ 0.0051614054484284,  0.0034333304633205, -0.003660801644545 ]]],


       [[[ 0.0051896436292775, -0.0143123051934424, -0.0024156231417938],
         [-0.0180676261100077, -0.029837508065178 ,  0.0217677950209438],
         [ 0.0086225595549161,  0.0271503545405949,  0.0052017963785844]],

        [[-0.0129978416092136,  0.0153429380551628, -0.0408363529487143],
         [ 0.015094819897872 , -0.0093026903753568, -0.0283517038857095],
         [-0.0452734060986601, -0.0336150204942598,  0.0478064593325556]],

        [[-0.0013389392359286, -0.0037064449386337,  0.0051505783645878],
         [ 0.0002911680723194, -0.0019250909708557,  0.0034342771330076],
         [-0.0014460076953604,  0.0033059148360493, -0.0036517947810122]],

        [[ 0.0091471372143936,  0.0026758120430237,  0.0381013977220346],
         [ 0.0026816381184513,  0.0410652893926278,  0.0031496317139945],
         [ 0.0380968542328941,  0.0031587513025511, -0.0493564609190811]]]])

        assert np.max(np.abs(test_hessian - reference_hessian)) < 1e-4
        # Translation invariance
        assert np.max(np.abs(np.sum(test_hessian, axis = 0))) < 1e-10

    def test_hessian_grid_response_vv10(self):
        mf = RKS(mol, xc = 'wB97M-V')
        mf.grids.atom_grid = (10,14)
        mf.nlcgrids.atom_grid = (15,14)
        mf.conv_tol = 1e-12
        mf.conv_tol_cpscf = 1e-10
        mf.cphf_grids.atom_grid = mf.grids.atom_grid
        mf.cphf_grids.prune = mf.grids.prune
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.kernel()
        assert mf.converged

        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hobj.grid_response = True

        test_hessian = hobj.kernel()

        # reference_hessian = numerical_d2e_dft(mf, dx = 1e-3)
        reference_hessian = np.array([[[[ 0.5119242880652353,  0.0600552100236129,  0.3310596056884663],
         [ 0.0600392618100853,  0.3522280861885108,  0.0693688618416122],
         [ 0.3310443052255696,  0.0693448856354806,  0.1492223138748194]],

        [[ 0.0308251280138477, -0.0333680253703506, -0.005599868620032 ],
         [ 0.0294542457289979, -0.252317468694585 , -0.0537590194421567],
         [-0.0000980633357919, -0.0759912042447297,  0.0111281742900537]],

        [[-0.5466985222231102, -0.0130722299946928, -0.3309991091859921],
         [-0.0754200138237682, -0.0709503882117546, -0.0408666584863493],
         [-0.3303974530681764, -0.010527259263815 , -0.164147626015243 ]],

        [[ 0.0039491061278873, -0.0136149546572928,  0.0055393721119512],
         [-0.0140734937215115, -0.0289602293423452,  0.0252568160871158],
         [-0.0005487888224964,  0.0171735778803639,  0.0037971378455404]]],


       [[[ 0.0308214718031108,  0.029449212807009 , -0.0000881169148204],
         [-0.0333487353674222, -0.2523200205919451, -0.076028116420046 ],
         [-0.0056012422029461, -0.0537444828001554,  0.0111428811970793]],

        [[-0.013535855779831 ,  0.0044398683085589,  0.0391973633909748],
         [ 0.0044644487885367,  0.2946616204013708,  0.0991369437092215],
         [ 0.0391914902598198,  0.0991106174268452, -0.0482020126073568]],

        [[-0.0005149240169811, -0.0451818390371228, -0.0014823988582213],
         [ 0.0149344297191156, -0.0309069154972863,  0.0058540084142411],
         [ 0.0025530751499581, -0.0237451529775945,  0.0019341008570262]],

        [[-0.01677069200684  ,  0.0112927579201949, -0.0376268476180441],
         [ 0.0139498568615393, -0.0114346843808066, -0.0289628356984206],
         [-0.0361433232080044, -0.0216209816420454,  0.035125030558969 ]]],


       [[[-0.5466591996347026, -0.0754623991694459, -0.3304145171768025],
         [-0.0130503282580463, -0.0709486737129339, -0.0104926674910355],
         [-0.3310033845190796, -0.040869880543859 , -0.1641580640592832]],

        [[-0.0005353207638004,  0.0149760989085479,  0.0025490046078325],
         [-0.0452319171716647, -0.0309128825808358, -0.0237768699469232],
         [-0.001458751309788 ,  0.0058480319840015,  0.0019407190016141]],

        [[ 0.548445034233902 ,  0.0607068782965681,  0.3286525416672514],
         [ 0.0606905330995389,  0.1031843955936473,  0.0323626448164305],
         [ 0.3286423568700236,  0.0323747868162805,  0.1634653472850633]],

        [[-0.0012505138210495, -0.0002205780329501, -0.0007870290899548],
         [-0.0024082876681208, -0.001322839295187 ,  0.0019068926244703],
         [ 0.0038197789620775,  0.0026470617451313, -0.0012480022199557]]],


       [[[ 0.0039625207426397, -0.0140906595982315, -0.0005921162040678],
         [-0.0136156526573394, -0.0289617980460122,  0.017185508928097 ],
         [ 0.0055446708626672,  0.0252692762554574,  0.0037741474692154]],

        [[-0.016747727537833 ,  0.0140229270337855, -0.0361259722384544],
         [ 0.0112854386013633, -0.0114345172059771, -0.0216287835588247],
         [-0.0376177222767213, -0.0289570957576757,  0.0351351629871433]],

        [[-0.0012339905870462, -0.0024587224934924,  0.0038372855798441],
         [-0.0002151739453282, -0.0013231699385052,  0.0026430112562981],
         [-0.0007893728557473,  0.0018908784633065, -0.0012431896900811]],

        [[ 0.0140191973790893,  0.0025264550559956,  0.032880802861901 ],
         [ 0.0025453880014292,  0.0417194851861924,  0.0018002633722647],
         [ 0.0328624242675324,  0.0017969410377461, -0.0376661207661111]]]])

        assert np.max(np.abs(test_hessian - reference_hessian)) < 2e-4
        # Translation invariance
        assert np.max(np.abs(np.sum(test_hessian, axis = 0))) < 1e-8

if __name__ == "__main__":
    print("Tests for KS hessian with grid response")
    unittest.main()
