import numpy as np
import time
from typing import Union, Tuple

FS2AUTIME = 41.34137
A2BOHR = 1.889726
AMU2AU = 1822.8884858012984

class FSSH:
    """
    A class to encapsulate the Fewest Switches Surface Hopping (FSSH) simulation.
    Ref:
        Molecular dynamics with electronic transitions. 
        John C. Tully
        The Journal of Chemical Physics 1990 93 (2), 1061-1071.
        DOI: 10.1063/1.459170
    """
    def __init__(self, tddft, states:list[int], **kwargs):
        """
        Initializes the FSSH simulation with given parameters.
        """

        # set the potential, force, nac function
        self.tddft = tddft
        self.tdgrad = self.tddft.nuc_grad_method()
        self.tdnac = self.tddft.NAC()

        # set total states
        self.states = states
        self.Nstates = len(states)
        self.cur_state = 0
        self.mass = self.tddft.mol.atom_mass_list(True).reshape(-1,1) * AMU2AU  # (Na,1)  Unit: a.u.
        self.nac_idx = [(self.states[i],self.states[j]) for i in range(self.Nstates-1) for j in range(i+1,self.Nstates)]

        self.dt = 0.5 * FS2AUTIME   # a.u.time
        self.nsteps = 1
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    def calc_electronic(self, position):
        """
        Calculate electronic energy, force, and NACV.
        """
        # calculate energy
        self.tddft.mol.set_geom_(position)
        self.tddft.reset(self.tddft.mol)
        self.tddft._scf.kernel()
        self.tddft.kernel()
        energy = np.concatenate([[self.tddft._scf.e_tot], self.tddft.e_tot])[self.states] # (Ns,)  Unit: Ha

        # calculate force
        self.tdgrad.state = self.cur_state
        grad = self.tdgrad.kernel()
        force = -grad  # (Na,D)  Unit: Ha/bohr

        # calculate nacv
        Nacv = np.zeros((self.Nstates, self.Nstates, force.shape[0], force.shape[1]))  # (Ns, Ns, Na, D)  Unit: 1/bohr
        for state in self.nac_idx:
            self.tdnac.states = state
            nk = self.tdnac.kernel()[3]
            Nacv[state[0],state[1]] = nk
            Nacv[state[1],state[0]] = -nk 
        return energy, force, Nacv
    
    def velocity_verlet(self, position, velocity, force):
        """
        Velocity Verlet algorithm.
        v(t+dt/2) = v(t) + a(t) * dt/2
        x(t+dt) = x(t) + v(t+dt/2) * dt
        v(t+dt) = v(t+dt/2) + a(t+dt) * dt/2
        """
        velocity += 0.5 * self.dt * force / self.mass
        position += self.dt * velocity

        return position, velocity

    def exp_propagator(self, c, Veff, dt) -> np.ndarray:
        """
        Exp propagator.
        dc/dt = -i V_eff(R,P) c
        """
        diags, coeff = np.linalg.eigh(Veff)
        U = coeff @ np.diag(np.exp(-1j * diags * dt)) @ coeff.T.conj()
        c_new = np.dot(U, c)
        return c_new

    def update_coefficient(self, coeffs, energy, nact):
        """
        Update quantum coefficients.
        V_eff(R,P) = E(R) - i * d(R) * P/m
        """
        E = np.diag(energy)
        Veff = E - 1j * nact
        c_new = self.exp_propagator(coeffs, Veff, self.dt)
        return c_new

    def compute_hopping_probability(self, coeffs, nact):
        """
        Compute hopping probability.
        g_ij = (2 Re(T_ij * c_i.conj * c_j) - 2/ħ Im(V_ij * c_i.conj * c_j))dt / |c_i|^2
        p_ij = max(0, g_ij)
        """
        g_ij = 2 * (nact[self.cur_state] * coeffs[self.cur_state].conj() * coeffs).real * self.dt / (np.abs(coeffs[self.cur_state])**2)
        p_ij = np.where(g_ij < 0, 0, g_ij)
        return p_ij

    def check_hop(self, g_ij) -> int:
        """
        Check if hop occurs.
        hop if sum_i(0,k-1) < r < sum_i(0,k)
        """
        r = np.random.rand()
        cumu_g_ij = np.cumsum(g_ij)
        for k, u_bound in enumerate(cumu_g_ij):
            l_bound = 0 if k == 0 else cumu_g_ij[k-1]
            if l_bound < r < u_bound:
                return k
        return -1

    def rescale_velocity(self, hop_index, energy, velocity, nacv):
        """
        Rescale velocity.

        v' = v - gamma * d_vec / mass
        if delta > 0:
            gamma = (b +- sqrt(b^2 - 4ac)) / 2a
            a = sum_i (d_i^2 / 2m_i)
            b = sum_i (v_i * d_i)
            c = E_new - E_old
        else:
            gamma = b / a
        """

        # To conserve energy, the new velocity v' = v - gamma * d_vec / mass must satisfy 
        # the energy conservation equation, which leads to a quadratic equation for the 
        # scaling factor gamma:
        #     a*gamma^2 - b*gamma + c = 0
        # where:
        #     a = sum_i (d_i^2 / 2m_i)
        #     b = sum_i (v_i * d_i)
        #     c = E_new - E_old

        d_vec = nacv[self.cur_state, hop_index]
        a = np.sum(d_vec**2 / (2 * self.mass))
        b = np.sum(velocity * d_vec)
        c = energy[hop_index] - energy[self.cur_state]
        delta = b**2 - 4 * a * c

        if delta >= 0:
            gamma = (b + np.sqrt(delta)) / (2 * a) if b < 0 else (b - np.sqrt(delta)) / (2 * a)
            velocity -= gamma * d_vec / self.mass
            return True, velocity
        else:
            gamma = b / a
            velocity -= gamma * d_vec / self.mass
            return False, velocity

    # NOT TESTED YET!!!!
    # def decoherence(self,
    #                 coeffs: np.ndarray,
    #                 velocity: np.ndarray,
    #                 energy: np.ndarray) -> np.ndarray:
    #     """
    #     Decoherence.

    #     c_j = c_j * exp(-dt / tau_ji)
    #     c_i = c_i * sqrt((1 - sum_j(j!=i) |c_j|**2) / |c_i|**2)
    #     tau_ji = ħ / |E_jj - E_ii| * (1 + a / E_kin)
    #     """

    #     E_kin = 0.5 * self.mass * np.sum(velocity ** 2)
    #     cumu_sum = 0
        
    #     for i in range(len(coeffs)):
    #         if i != self.cur_state:
    #             tau_ji = 1 / np.abs(energy[i] - energy[self.cur_state]) * (1 + self.alpha / E_kin)
    #             coeffs[i] = coeffs[i] * np.exp(-self.dt / tau_ji)
    #             cumu_sum += np.abs(coeffs[i]) ** 2
        
    #     coeffs[self.cur_state] = np.sqrt((1 - cumu_sum) / np.abs(coeffs[self.cur_state]) ** 2) * coeffs[self.cur_state]
    #     return coeffs

    # TODO: add screen print
    def screen_print(self):
        pass

    # TODO: add file output
    def write_output(self):
        pass

    def kernel(self, position, velocity, coefficient):
        """
        Runs a single FSSH trajectory.
        Integration Frame Ref:
            Nonadiabatic Field on Quantum Phase Space: A Century after Ehrenfest
            Baihua Wu, Xin He, and Jian Liu
            The Journal of Physical Chemistry Letters 2024 15 (2), 644-658
            DOI: 10.1021/acs.jpclett.3c03385
        """

        position = position * A2BOHR   # (Na,D) a.u.
        velocity = velocity * A2BOHR / (FS2AUTIME * 1e3) # (Na,D) Bohr/a.u.Time
        norm = np.linalg.norm(coefficient)
        coefficient /= norm
        energy, force, nacv = self.calc_electronic(position)
        total_time = 0
        
        for i in range(self.nsteps):
            # 1. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * force / self.mass
            # 2. update the nuclear coordinate within a full-time step
            position = position + self.dt * velocity
            # 3. calculte new energy, force, and nacv
            energy, force, nacv = self.calc_electronic(position) # TODO check for the direction of NACV
            # 4. update the electronic amplitude within a full-time step
            nact = np.einsum('ijnd,nd->ij', nacv, velocity)
            coefficient = self.update_coefficient(coefficient, energy, nact)
            # 5. evaluate the switching probability
            p_ij = self.compute_hopping_probability(coefficient, nact)
            hop_index = self.check_hop(p_ij)
            # 6. adjust nuclear velocity
            if hop_index != -1 and hop_index != self.cur_state:
                hop_allowed, velocity = self.rescale_velocity(hop_index, energy, velocity, nacv)
                if hop_allowed:
                    self.cur_state = hop_index
            # 7. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * force / self.mass
            # 8. update total time
            total_time += self.dt / FS2AUTIME
            # 9. decoherence
            # coefficient = self.decoherence(coefficient, velocity, energy)

            print(f"Step {i+1}: Time {total_time:.2f} fs, State {self.cur_state}, Energy {energy[self.cur_state]:.6f} Ha, "
                  f"density: {np.abs(coefficient)**2}")
            
        return position