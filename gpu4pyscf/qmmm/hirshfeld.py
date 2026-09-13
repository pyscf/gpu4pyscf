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

import cupy as cp
import numpy as np

import pyscf
from pyscf.data.elements import charge as charge_of_element
from pyscf.gto.mole import ANG_OF
from pyscf.dft.LebedevGrid import LEBEDEV_NGRID, MakeAngularGrid
from gpu4pyscf.lib import logger
from gpu4pyscf.dft.numint import NumInt
from gpu4pyscf.solvent.pcm import natm_without_ghost
from gpu4pyscf.dft.gen_grid import Grids
from gpu4pyscf.dft.uks import UKS
from gpu4pyscf.scf.uhf import UHF as UHF_GPU
from pyscf.scf.uhf import UHF as UHF_CPU

HIRSHFELD_REMOVE_ZERO_RHO_GRID_THRESHOLD = 1e-10

_neutral_atom_spin = [
    #    Z    El  Name           Ground level  2S  S    Mult.  Source
    #    ---  --  -------------  ------------  --  ---  -----  -------------------------------
    1, # 1    H   Hydrogen       2S<1/2>       1   1/2  2      NIST term
    0, # 2    He  Helium         1S0           0   0    1      NIST term
    1, # 3    Li  Lithium        2S<1/2>       1   1/2  2      NIST term
    0, # 4    Be  Beryllium      1S0           0   0    1      NIST term
    1, # 5    B   Boron          2P*<1/2>      1   1/2  2      NIST term
    2, # 6    C   Carbon         3P0           2   1    3      NIST term
    3, # 7    N   Nitrogen       4S*<3/2>      3   3/2  4      NIST term
    2, # 8    O   Oxygen         3P<2>         2   1    3      NIST term
    1, # 9    F   Fluorine       2P*<3/2>      1   1/2  2      NIST term
    0, # 10   Ne  Neon           1S0           0   0    1      NIST term
    1, # 11   Na  Sodium         2S<1/2>       1   1/2  2      NIST term
    0, # 12   Mg  Magnesium      1S0           0   0    1      NIST term
    1, # 13   Al  Aluminum       2P*<1/2>      1   1/2  2      NIST term
    2, # 14   Si  Silicon        3P0           2   1    3      NIST term
    3, # 15   P   Phosphorus     4S*<3/2>      3   3/2  4      NIST term
    2, # 16   S   Sulfur         3P<2>         2   1    3      NIST term
    1, # 17   Cl  Chlorine       2P*<3/2>      1   1/2  2      NIST term
    0, # 18   Ar  Argon          1S0           0   0    1      NIST term
    1, # 19   K   Potassium      2S<1/2>       1   1/2  2      NIST term
    0, # 20   Ca  Calcium        1S0           0   0    1      NIST term
    1, # 21   Sc  Scandium       2D<3/2>       1   1/2  2      NIST term
    2, # 22   Ti  Titanium       3F<2>         2   1    3      NIST term
    3, # 23   V   Vanadium       4F<3/2>       3   3/2  4      NIST term
    6, # 24   Cr  Chromium       7S<3>         6   3    7      NIST term
    5, # 25   Mn  Manganese      6S<5/2>       5   5/2  6      NIST term
    4, # 26   Fe  Iron           5D<4>         4   2    5      NIST term
    3, # 27   Co  Cobalt         4F<9/2>       3   3/2  4      NIST term
    2, # 28   Ni  Nickel         3F<4>         2   1    3      NIST term
    1, # 29   Cu  Copper         2S<1/2>       1   1/2  2      NIST term
    0, # 30   Zn  Zinc           1S0           0   0    1      NIST term
    1, # 31   Ga  Gallium        2P*<1/2>      1   1/2  2      NIST term
    2, # 32   Ge  Germanium      3P0           2   1    3      NIST term
    3, # 33   As  Arsenic        4S*<3/2>      3   3/2  4      NIST term
    2, # 34   Se  Selenium       3P<2>         2   1    3      NIST term
    1, # 35   Br  Bromine        2P*<3/2>      1   1/2  2      NIST term
    0, # 36   Kr  Krypton        1S0           0   0    1      NIST term
    1, # 37   Rb  Rubidium       2S<1/2>       1   1/2  2      NIST term
    0, # 38   Sr  Strontium      1S0           0   0    1      NIST term
    1, # 39   Y   Yttrium        2D<3/2>       1   1/2  2      NIST term
    2, # 40   Zr  Zirconium      3F<2>         2   1    3      NIST term
    5, # 41   Nb  Niobium        6D<1/2>       5   5/2  6      NIST term
    6, # 42   Mo  Molybdenum     7S<3>         6   3    7      NIST term
    5, # 43   Tc  Technetium     6S<5/2>       5   5/2  6      NIST term
    4, # 44   Ru  Ruthenium      5F<5>         4   2    5      NIST term
    3, # 45   Rh  Rhodium        4F<9/2>       3   3/2  4      NIST term
    0, # 46   Pd  Palladium      1S0           0   0    1      NIST term
    1, # 47   Ag  Silver         2S<1/2>       1   1/2  2      NIST term
    0, # 48   Cd  Cadmium        1S0           0   0    1      NIST term
    1, # 49   In  Indium         2P*<1/2>      1   1/2  2      NIST term
    2, # 50   Sn  Tin            3P0           2   1    3      NIST term
    3, # 51   Sb  Antimony       4S*<3/2>      3   3/2  4      NIST term
    2, # 52   Te  Tellurium      3P<2>         2   1    3      NIST term
    1, # 53   I   Iodine         2P*<3/2>      1   1/2  2      NIST term
    0, # 54   Xe  Xenon          1S0           0   0    1      NIST term
    1, # 55   Cs  Cesium         2S<1/2>       1   1/2  2      NIST term
    0, # 56   Ba  Barium         1S0           0   0    1      NIST term
    1, # 57   La  Lanthanum      2D<3/2>       1   1/2  2      NIST term
    0, # 58   Ce  Cerium         1G*<4>        0   0    1      NIST term
    3, # 59   Pr  Praseodymium   4I*<9/2>      3   3/2  4      NIST term
    4, # 60   Nd  Neodymium      5I<4>         4   2    5      NIST term
    5, # 61   Pm  Promethium     6H*<5/2>      5   5/2  6      NIST term
    6, # 62   Sm  Samarium       7F0           6   3    7      NIST term
    7, # 63   Eu  Europium       8S*<7/2>      7   7/2  8      NIST term
    8, # 64   Gd  Gadolinium     9D*<2>        8   4    9      NIST term
    5, # 65   Tb  Terbium        6H*<15/2>     5   5/2  6      NIST term
    4, # 66   Dy  Dysprosium     5I<8>         4   2    5      NIST term
    3, # 67   Ho  Holmium        4I*<15/2>     3   3/2  4      NIST term
    2, # 68   Er  Erbium         3H<6>         2   1    3      NIST term
    1, # 69   Tm  Thulium        2F*<7/2>      1   1/2  2      NIST term
    0, # 70   Yb  Ytterbium      1S0           0   0    1      NIST term
    1, # 71   Lu  Lutetium       2D<3/2>       1   1/2  2      NIST term
    2, # 72   Hf  Hafnium        3F<2>         2   1    3      NIST term
    3, # 73   Ta  Tantalum       4F<3/2>       3   3/2  4      NIST term
    4, # 74   W   Tungsten       5D0           4   2    5      NIST term
    5, # 75   Re  Rhenium        6S<5/2>       5   5/2  6      NIST term
    4, # 76   Os  Osmium         5D<4>         4   2    5      NIST term
    3, # 77   Ir  Iridium        4F<9/2>       3   3/2  4      NIST term
    2, # 78   Pt  Platinum       3D<3>         2   1    3      NIST term
    1, # 79   Au  Gold           2S<1/2>       1   1/2  2      NIST term
    0, # 80   Hg  Mercury        1S0           0   0    1      NIST term
    1, # 81   Tl  Thallium       2P*<1/2>      1   1/2  2      NIST term
    2, # 82   Pb  Lead           (1/2,1/2)0    2   1    3      Hund/config (NIST)
    3, # 83   Bi  Bismuth        4S*<3/2>      3   3/2  4      NIST term
    2, # 84   Po  Polonium       3P<2>         2   1    3      NIST term
    1, # 85   At  Astatine       2P*<3/2>      1   1/2  2      NIST term
    0, # 86   Rn  Radon          1S0           0   0    1      NIST term
    1, # 87   Fr  Francium       2S<1/2>       1   1/2  2      NIST term
    0, # 88   Ra  Radium         1S0           0   0    1      NIST term
    1, # 89   Ac  Actinium       2D<3/2>       1   1/2  2      NIST term
    2, # 90   Th  Thorium        3F<2>         2   1    3      NIST term
    3, # 91   Pa  Protactinium   4K<11/2>      3   3/2  4      NIST term
    4, # 92   U   Uranium        5L*<6>        4   2    5      NIST term
    5, # 93   Np  Neptunium      6L<11/2>      5   5/2  6      NIST term
    6, # 94   Pu  Plutonium      7F0           6   3    7      NIST term
    7, # 95   Am  Americium      8S*<7/2>      7   7/2  8      NIST term
    8, # 96   Cm  Curium         9D*<2>        8   4    9      NIST term
    5, # 97   Bk  Berkelium      6H*<15/2>     5   5/2  6      NIST term
    4, # 98   Cf  Californium    5I<8>         4   2    5      NIST term
    3, # 99   Es  Einsteinium    4I*<15/2>     3   3/2  4      NIST term
    2, # 100  Fm  Fermium        3H<6>         2   1    3      NIST term
    1, # 101  Md  Mendelevium    2F*<7/2>      1   1/2  2      NIST term
    0, # 102  No  Nobelium       1S0           0   0    1      NIST term
    1, # 103  Lr  Lawrencium     2P*<1/2>      1   1/2  2      NIST term
    2, # 104  Rf  Rutherfordium  3F<2>         2   1    3      NIST term
    3, # 105  Db  Dubnium        4F<3/2>       3   3/2  4      NIST term
    4, # 106  Sg  Seaborgium     0             4   2    5      Hund/config (NIST)
    5, # 107  Bh  Bohrium        <5/2>         5   5/2  6      Hund/config (NIST)
    4, # 108  Hs  Hassium        <4>           4   2    5      Hund/config (NIST)
]

def hirshfeld_kernel(mol, grids, dm, make_mf):
    log = logger.new_logger(mol)

    dm = cp.asarray(dm)
    if dm.ndim == 3:
        assert dm.shape[0] == 2
        dm = dm[0] + dm[1]
    assert dm.shape == (mol.nao, mol.nao)

    natm = natm_without_ghost(mol)
    if natm != mol.natm:
        raise NotImplementedError("Ghost atoms are not supported in Hirshfeld yet")

    assert callable(make_mf)

    elements = mol.elements
    unique_elements = list(set(elements))

    mf_per_element = {}
    for element in unique_elements:
        Z = charge_of_element(element)
        assert 0 < Z and Z < len(_neutral_atom_spin)
        spin = _neutral_atom_spin[Z - 1]
        charge = 0 # Only support neutral atoms for now

        mol_atom = pyscf.M(
            atom = f"{element} 0 0 0",
            basis = mol.basis,
            ecp = mol.ecp,
            charge = charge,
            spin = spin,
            verbose = mol.verbose,
        )

        mf_atom = make_mf(mol_atom)

        if not (isinstance(mf_atom, UHF_GPU) or isinstance(mf_atom, UHF_CPU)):
            log.warn("We highly recommend using unrestricted method to perform atomic calculation.")

        mf_atom.kernel()
        assert mf_atom.converged

        mf_per_element[element] = mf_atom

    if grids.coords is None:
        grids.build()

    grid_coords = cp.asarray(grids.coords)
    grid_weights = cp.asarray(grids.weights)
    ni = NumInt()
    grid_rho = ni.get_rho(mol, dm, grids)

    rho_nonzero_mask = cp.logical_and(
        grid_rho >= HIRSHFELD_REMOVE_ZERO_RHO_GRID_THRESHOLD,
        cp.abs(grid_weights) > 1e-14,
    )

    grid_rho = grid_rho[rho_nonzero_mask]
    grid_coords = cp.ascontiguousarray(grid_coords[rho_nonzero_mask, :])
    grid_weights = grid_weights[rho_nonzero_mask]
    grid_w_rho = grid_weights * grid_rho
    ngrids = grid_coords.shape[0]

    nelec = float(cp.sum(grid_w_rho))
    log.info(f"Total number of electrons integrated by Hirshfeld grid = {nelec}")

    atom_coords = cp.asarray(mol.atom_coords())

    atomic_rho = []
    for i_atom in range(natm):
        element = elements[i_atom]
        mf_atom = mf_per_element[element]

        atom_grid_rij = cp.linalg.norm(grid_coords - atom_coords[i_atom], axis = 1)

        mol_atom = mf_atom.mol
        L_max = np.max(mol_atom._bas[:, ANG_OF])
        n_angular_grid = LEBEDEV_NGRID[L_max]
        if n_angular_grid > 1:
            angular_grid = cp.asarray(MakeAngularGrid(n_angular_grid))
        else:
            angular_grid = cp.array([[0, 0, 1.0, 1.0]])
        spherical_average_grid_coords = angular_grid[None, :, :3] * atom_grid_rij[:, None, None] # Centered at the atom
        spherical_average_grid_coords = spherical_average_grid_coords.reshape(ngrids * n_angular_grid, 3)
        del atom_grid_rij

        fake_grids = Grids(pyscf.M())
        fake_grids.coords = spherical_average_grid_coords
        fake_grids.weights = spherical_average_grid_coords[:, 0] # Just need a correct shape
        def just_raise(*args, **kwargs):
            raise RuntimeError("Should never be called")
        fake_grids.build = just_raise

        dm_atom = mf_atom.make_rdm1()
        if dm_atom.ndim == 3:
            assert dm_atom.shape[0] == 2
            dm_atom = dm_atom[0] + dm_atom[1]
        assert dm_atom.shape == (mol_atom.nao, mol_atom.nao)

        ni = NumInt()
        rho_atom = ni.get_rho(mol_atom, dm_atom, fake_grids)
        rho_atom = rho_atom.reshape(ngrids, n_angular_grid)

        del spherical_average_grid_coords, dm_atom

        rho_atom = cp.sum(rho_atom * angular_grid[:, 3], axis = 1) # Spherical averaged
        del angular_grid
        nelec_atom = float(cp.sum(rho_atom * grid_weights))
        log.info(f"Hirshfeld grid intergation of atom {element} = {nelec_atom}")
        atomic_rho.append(rho_atom)

    atomic_rho = cp.stack(atomic_rho, axis = 0)
    sum_atomic_rho = cp.sum(atomic_rho, axis = 0)

    sum_atomic_rho_nonzero_mask = sum_atomic_rho >= HIRSHFELD_REMOVE_ZERO_RHO_GRID_THRESHOLD
    atom_partition = cp.zeros_like(atomic_rho)
    atom_partition[:, sum_atomic_rho_nonzero_mask] = atomic_rho[:, sum_atomic_rho_nonzero_mask] / sum_atomic_rho[sum_atomic_rho_nonzero_mask]
    del sum_atomic_rho, sum_atomic_rho_nonzero_mask, atomic_rho

    # Hirshfeld Multipoles

    partitioned_w_rho = atom_partition * grid_w_rho[None, :]
    partitioned_nelec = cp.sum(partitioned_w_rho, axis = 1)
    atom_charges = cp.asarray(mol.atom_charges(), dtype = cp.float64)
    charges = atom_charges - partitioned_nelec

    log.info("Hirshfeld Charge (a.u.)")
    for i_atom in range(mol.natm):
        log.info(f"{mol.elements[i_atom]:2s}  {charges[i_atom]:13.8f}")

    atom_grid_vecrij = grid_coords[None, :, :] - atom_coords[:, None, :]

    dipoles = -cp.einsum("Ag,Agx->Ax", partitioned_w_rho, atom_grid_vecrij)

    log.info("Hirshfeld Dipole x y z (a.u.)")
    for i_atom in range(mol.natm):
        log.info(f"{mol.elements[i_atom]:2s}  {dipoles[i_atom, 0]:13.8f}  {dipoles[i_atom, 1]:13.8f}  {dipoles[i_atom, 2]:13.8f}")

    # ORCA did not remove trace, and we follow them
    quadrupoles = -cp.einsum("Ag,Agx,Agy->Axy", partitioned_w_rho, atom_grid_vecrij, atom_grid_vecrij)

    log.info("Hirshfeld Quadrupole xx yy zz xy xz yz (a.u.)")
    for i_atom in range(mol.natm):
        log.info(f"{mol.elements[i_atom]:2s}  {quadrupoles[i_atom, 0, 0]:13.8f}  {quadrupoles[i_atom, 1, 1]:13.8f}  {quadrupoles[i_atom, 2, 2]:13.8f}  "
                 f"{0.5 * (quadrupoles[i_atom, 0, 1] + quadrupoles[i_atom, 1, 0]):13.8f}  "
                 f"{0.5 * (quadrupoles[i_atom, 0, 2] + quadrupoles[i_atom, 2, 0]):13.8f}  "
                 f"{0.5 * (quadrupoles[i_atom, 1, 2] + quadrupoles[i_atom, 2, 1]):13.8f}")

    # ORCA did not remove trace, and we follow them
    octupoles = -cp.einsum("Ag,Agx,Agy,Agz->Axyz", partitioned_w_rho, atom_grid_vecrij, atom_grid_vecrij, atom_grid_vecrij)

    log.info("Hirshfeld Octupole xxx yyy zzz xxy xxz xyy xyz xzz yyz yzz (a.u.)")
    octupole_xyz_term = octupoles[i_atom, 0, 1, 2] + octupoles[i_atom, 0, 2, 1] + octupoles[i_atom, 1, 0, 2] \
                        + octupoles[i_atom, 1, 2, 0] + octupoles[i_atom, 2, 0, 1] + octupoles[i_atom, 2, 1, 0] # Just to make linter happy
    for i_atom in range(mol.natm):
        log.info(f"{mol.elements[i_atom]:2s}  {octupoles[i_atom, 0, 0, 0]:13.8f}  {octupoles[i_atom, 1, 1, 1]:13.8f}  {octupoles[i_atom, 2, 2, 2]:13.8f}  "
                 f"{1.0/3.0 * (octupoles[i_atom, 0, 0, 1] + octupoles[i_atom, 0, 1, 0] + octupoles[i_atom, 1, 0, 0]):13.8f}  "
                 f"{1.0/3.0 * (octupoles[i_atom, 0, 0, 2] + octupoles[i_atom, 0, 2, 0] + octupoles[i_atom, 2, 0, 0]):13.8f}  "
                 f"{1.0/3.0 * (octupoles[i_atom, 0, 1, 1] + octupoles[i_atom, 1, 0, 1] + octupoles[i_atom, 1, 1, 0]):13.8f}  "
                 f"{1.0/6.0 * octupole_xyz_term:13.8f}  "
                 f"{1.0/3.0 * (octupoles[i_atom, 0, 2, 2] + octupoles[i_atom, 2, 0, 2] + octupoles[i_atom, 2, 2, 0]):13.8f}  "
                 f"{1.0/3.0 * (octupoles[i_atom, 1, 1, 2] + octupoles[i_atom, 1, 2, 1] + octupoles[i_atom, 2, 1, 1]):13.8f}  "
                 f"{1.0/3.0 * (octupoles[i_atom, 1, 2, 2] + octupoles[i_atom, 2, 1, 2] + octupoles[i_atom, 2, 2, 1]):13.8f}  "
                 f"")

    return charges.get(), dipoles.get(), quadrupoles.get(), octupoles.get()

def hirshfeld(mol, grids, dm, xc = "wB97X-V", xc_grid = (99,590), nlc_grid = (50,194), auxbasis = None,
              conv_tol = 1e-10, max_cycle = 100, nlc = None, disp = None):
    """
        Original Hirshfeld algorithm following:
        Hirshfeld, F. L. (1977). Bonded-atom fragments for describing molecular charge densities. Theoretica chimica acta, 44(2), 129-138.

        i.e. Use atomic density of neutral atoms regardless of total molecular charge, and perform spherical averaging for atomic density.
    """

    def _make_mf(mol):
        if xc is None or xc.upper() == "HF":
            mf = UHF_GPU(mol)
        else:
            mf = UKS(mol, xc = xc)
            mf.grids.atom_grid = xc_grid
            mf.nlcgrids.atom_grid = nlc_grid
        if nlc is not None:
            mf.nlc = nlc
        if disp is not None:
            mf.disp = disp
        mf.conv_tol = conv_tol
        mf.max_cycle = max_cycle
        if auxbasis is not None:
            mf = mf.density_fit(auxbasis = auxbasis)
        return mf

    return hirshfeld_kernel(mol, grids, dm, make_mf = _make_mf)
