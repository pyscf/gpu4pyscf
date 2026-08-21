# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

"""
EXPERIMENTAL FEATURE
Numerical Gamma-point force constants and phonon modes.

Only the analytic (TO) contribution is included. Non-analytic corrections
(NAC) are not supported.
"""

__all__ = [
    "HAS_PHONOPY",
    "GammaHessian",
]

import copy
import numpy as np
from pyscf import lib
from pyscf.data.nist import BOHR as BOHR_TO_ANGSTROM
from pyscf.data.nist import HARTREE2EV, AMU2AU, HARTREE2WAVENUMBER, HARTREE2J, PLANCK
try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    HAS_PHONOPY = True
except ImportError:
    HAS_PHONOPY = False
from gpu4pyscf.lib import logger

GRAD_TO_FORCE = -(HARTREE2EV / BOHR_TO_ANGSTROM)
EV_A2_TO_HA_BOHR2 = BOHR_TO_ANGSTROM**2 / HARTREE2EV
HARTREE_TO_THZ = HARTREE2J / (PLANCK * 1e12)


# TODO: in the future, the interfaces should be the same with mol.
class GammaHessian(lib.StreamObject):
    """Numerical Gamma-point TO Hessian without NAC, seriously q=0, NOT q->0.

    ``primitive_matrix`` maps the input cell to the primitive cell and must
    be supplied explicitly. ``kernel`` stores the corresponding phonopy
    object in ``phonon``.
    """

    def __init__(
        self,
        mf,
        primitive_matrix,       # transformation matrix to primitive cell from unit cell
        displacement=0.01,      # angstroms
        force_central_diff=False,
        symmetrize=True,
    ):
        primitive_matrix = np.asarray(primitive_matrix, dtype=np.float64)
        if primitive_matrix.shape != (3, 3):
            raise ValueError("primitive_matrix must have shape (3, 3)")

        self.mf = mf
        self.primitive_matrix = primitive_matrix.copy()
        self.displacement = displacement
        self.force_central_diff = force_central_diff
        self.symmetrize = symmetrize

        self.phonon = None
        self.force_sets = None
        self.force_constants = None

    def kernel(self):
        """Compute the primitive-cell Gamma Hessian in Hartree/Bohr**2.

        The returned shape is ``(3N, 3N)``, where N is the number of atoms
        in the primitive cell defined by ``primitive_matrix``.
        """
        # TODO: if there is no phonopy, should change the routine to use manual central difference
        if not HAS_PHONOPY:
            raise ImportError(
                "phonopy is required for GammaHessian; install it with "
                "`pip install phonopy`."
            )
        if not np.isfinite(self.displacement) or self.displacement <= 0:
            raise ValueError("displacement must be a positive finite number")

        self.force_sets = None
        self.force_constants = None
        mf = self.mf
        cell = mf.cell
        natm = cell.natm
        original_coords = cell.atom_coords().copy() # Bohr
        original_atom = copy.deepcopy(cell.atom)
        original_lattice = copy.deepcopy(cell.a)    # Bohr
        original_unit = cell.unit

        # default calculator is vasp, default unit in phonopy for vasp:
        #           | Distance   Atomic mass   Force         Force constants
        # -----------------------------------------------------------------
        # VASP      | Angstrom   AMU           eV/Angstrom   eV/Angstrom^2
        unitcell = PhonopyAtoms(
            symbols=[cell.atom_symbol(i) for i in range(natm)],
            cell=np.asarray(cell.lattice_vectors()) * BOHR_TO_ANGSTROM,
            positions=original_coords * BOHR_TO_ANGSTROM,
            masses=np.asarray(cell.atom_mass_list()),
        )
        self.phonon = Phonopy(
            unitcell,
            supercell_matrix=np.eye(3, dtype=int),
            primitive_matrix=self.primitive_matrix,
            log_level=0,
        )
        is_plusminus = True if bool(self.force_central_diff) else "auto"
        self.phonon.generate_displacements(
            distance=self.displacement,
            is_plusminus=is_plusminus,
        )
        displaced_cells = [
            displaced
            for displaced in self.phonon.supercells_with_displacements
            if displaced is not None
        ]
        logger.info(
            mf,
            "Gamma-point finite-difference Hessian: %d displaced structures, "
            "distance = %.6g Angstrom",
            len(displaced_cells),
            self.displacement,
        )

        force_sets = []
        dm0 = mf.make_rdm1()
        try:
            for index, displaced in enumerate(displaced_cells, 1):
                cell.set_geom_(displaced.positions, unit="Angstrom")
                mf_disp = mf.copy().reset(cell)
                logger.info(
                    mf,
                    "Running displaced SCF and gradient %d/%d",
                    index,
                    len(displaced_cells),
                )
                mf_disp.kernel(dm0=dm0)
                if not mf_disp.converged:
                    logger.warn(
                        mf,
                        "SCF did not converge for displaced structure %d/%d",
                        index,
                        len(displaced_cells),
                    )
                    raise RuntimeError("SCF did not converge")

                dm0 = mf_disp.make_rdm1()
                grad = mf_disp.nuc_grad_method()
                gradient = grad.kernel()
                if hasattr(gradient, "get"):
                    gradient = gradient.get()
                force_sets.append(np.asarray(gradient) * GRAD_TO_FORCE)

                del gradient, grad, mf_disp

            self.force_sets = np.asarray(force_sets)
            self.phonon.forces = self.force_sets
            self.phonon.produce_force_constants()
            if self.symmetrize:
                self.phonon.symmetrize_force_constants()

            qpoints = self.phonon.run_qpoints(
                [[0.0, 0.0, 0.0]],
                with_dynamical_matrices=True,
            )
            dyn_mat = np.asarray(qpoints.dynamical_matrices[0])
            masses = np.repeat(self.phonon.primitive.masses, 3)
            force_constants = (
                dyn_mat * np.sqrt(masses[:, None] * masses[None, :])
            )
            self.force_constants = np.asarray(
                np.real_if_close(force_constants) * EV_A2_TO_HA_BOHR2,
                dtype=np.float64,
            )
            return self.force_constants
        finally:
            cell.set_geom_(
                original_atom,
                a=original_lattice,
                unit=original_unit,
            )

    def phonon_modes(self, unit="cm-1"):
        """Diagonalize the Gamma dynamical matrix manually.

        Returns frequencies, mass-weighted eigenvectors, and the dynamical
        matrix in atomic units.
        """
        if self.force_constants is None:
            raise RuntimeError("kernel() must be called before phonon_modes()")

        fc = np.asarray(self.force_constants, dtype=np.float64)
        masses_amu = np.asarray(
            self.phonon.primitive.masses,
            dtype=np.float64,
        )
        size = 3 * len(masses_amu)
        if fc.shape != (size, size):
            raise ValueError(
                f"force_constants must have shape ({size}, {size}); "
                f"received {fc.shape}"
            )

        unit_key = self._validate_frequency_unit(unit)
        masses_me = masses_amu * AMU2AU
        mass_inv_sqrt = 1.0 / np.sqrt(np.repeat(masses_me, 3))
        dyn_mat = fc * mass_inv_sqrt[:, None] * mass_inv_sqrt[None, :]
        eigenvalues, eigenvectors = np.linalg.eigh(dyn_mat)

        factor = (
            HARTREE2WAVENUMBER
            if unit_key == "cm-1"
            else HARTREE_TO_THZ
        )
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
        frequencies *= factor
        self._log_frequencies(frequencies, unit_key, "manually")
        return frequencies, eigenvectors, dyn_mat

    def phonopy_modes(self, unit="cm-1"):
        """Calculate Gamma-point modes through the phonopy q-point API.

        Returns frequencies, mass-weighted eigenvectors, and the dynamical
        matrix in atomic units.
        """
        if self.phonon is None or self.phonon.force_constants is None:
            raise RuntimeError("kernel() must be called before phonopy_modes()")

        unit_key = self._validate_frequency_unit(unit)
        qpoints = self.phonon.run_qpoints(
            [[0.0, 0.0, 0.0]],
            with_eigenvectors=True,
            with_dynamical_matrices=True,
        )
        frequencies = np.asarray(qpoints.frequencies[0]).copy()
        if unit_key == "cm-1":
            frequencies *= HARTREE2WAVENUMBER / HARTREE_TO_THZ

        eigenvectors = np.asarray(qpoints.eigenvectors[0]).copy()
        dyn_mat = np.asarray(qpoints.dynamical_matrices[0]).copy()
        dyn_mat *= EV_A2_TO_HA_BOHR2 / AMU2AU
        dyn_mat = np.real_if_close(dyn_mat)

        self._log_frequencies(frequencies, unit_key, "phonopy")
        return frequencies, eigenvectors, dyn_mat

    @staticmethod
    def _validate_frequency_unit(unit):
        unit_key = unit.lower()
        if unit_key not in ("cm-1", "thz"):
            raise ValueError("unit must be 'cm-1' or 'thz'")
        return unit_key

    def _log_frequencies(self, frequencies, unit, source):
        logger.info(
            self.mf,
            "Lowest Gamma-point frequencies from %s (%s): %s",
            source,
            unit,
            np.array2string(
                frequencies[: min(6, frequencies.size)],
                precision=4,
            ),
        )
