#!/usr/bin/env python3
"""Implementation of the Born-Huggins-Mayer-Tosi-Fumi potential for NaCl in OpenMM.

Some relevant papers for this model are listed below.
The model parameters are taken from the second paper.

- F.G. Fumi, M.P. Tosi;
  Ionic sizes and born repulsive parameters in the NaCl-type alkali halides—I:
  The Huggins-Mayer and Pauling forms
  J. Phys. Chem. Solids 1964; 25, 31--43
  https://doi.org/10.1016/0022-3697(64)90159-3

- John W. E. Lewis, Konrad Singer,  Leslie V. Woodcock;
  Thermodynamic and structural properties of liquid ionic salts obtained by Monte Carlo computation.
  Part 2. Eight alkali metal halides
  J. Chem. Soc., Faraday Trans. 1975; 71, 301--312
  https://doi.org/10.1039/F29757100301

- G. Ciccotti, G. Jacucci, I. R. McDonald;
  Transport properties of molten alkali halides
  Phys. Rev. A 1976; 13, 426--436
  https://link.aps.org/doi/10.1103/PhysRevA.13.426

- Jia Wang, Ze Sun, Guimin Lu, Jianguo Yu;
  Molecular Dynamics Simulations of the Local Structures
  and Transport Coeﬃcients of Molten Alkali Chlorides
  J. Phys. Chem. B 2009; 118, 10196--10206
  https://doi.org/10.1021/jp5050332

- Tanooj Shah, Kamron Fazel, Jie Lian, Liping Huang, Yunfeng Shi, Ravishankar Sundararaman;
  First-principles molten salt phase diagrams through thermodynamic integration.
  J. Chem. Phys. 28 September 2023; 159 (12): 124502.
  https://doi.org/10.1063/5.0164824
  https://arxiv.org/abs/2306.02406

"""

import numpy as np
import pytest
from numpy.typing import NDArray
from openmm import CMMotionRemover, CustomNonbondedForce, NonbondedForce, System, unit
from openmm.app import Element, PDBFile, Simulation, Topology

__all__ = ("add_nacl_forces", "build_nacl", "build_nacl_lattice", "load_nacl")

# Useful physical constants (mostly for testing), not related to this model specifically.
COULCONST = 8.9875517862e9 * unit.joule * unit.meter / unit.coulomb**2
COULCONST_PER_MOLE = COULCONST * unit.AVOGADRO_CONSTANT_NA

# Unit conversion constants
C6_UNIT = 1e-79 * unit.joule * unit.AVOGADRO_CONSTANT_NA * unit.meter**6
# C6_UNIT = 1e-60 * unit.erg * unit.AVOGADRO_CONSTANT_NA * unit.centimeter**6
C8_UNIT = 1e-99 * unit.joule * unit.AVOGADRO_CONSTANT_NA * unit.meter**8
# C8_UNIT = 1e-76 * unit.erg * unit.AVOGADRO_CONSTANT_NA * unit.centimeter**8

# Define all parameters for the NaCl Born-Huggins-Mayer-Tosi-Fumi simulation.
PARS = {
    # Physical properties
    "na_charge": 1 * unit.elementary_charge,
    "cl_charge": -1 * unit.elementary_charge,
    # Born-Huggins-Mayer-Tosi-Fumi potential from the Lewis paper for NaCl.
    # Note that OpenMM expects energy units per mole, not per atom or atom pair.
    "sigma_na": 1.17 * unit.angstrom,
    "sigma_cl": 1.585 * unit.angstrom,
    "b": 0.338e-19 * unit.joule * unit.AVOGADRO_CONSTANT_NA,
    "c++": 1.25,
    "c+-": 1.0,
    "c--": 0.75,
    "C_na_na": 1.68 * C6_UNIT,
    "C_na_cl": 11.2 * C6_UNIT,
    "C_cl_cl": 116.0 * C6_UNIT,
    "D_na_na": 0.8 * C8_UNIT,
    "D_na_cl": 13.9 * C8_UNIT,
    "D_cl_cl": 233.0 * C8_UNIT,
    # Note that there is in error in the Lewis paper (wrong units).
    # The following correction is consistent with the papers of Ciccotti, Wang or Shah.
    # In the paper of Lewis, the parameter is referred to as B in the table,
    # but is A in the equation. We're using A here which seems to be the most consistent.
    "A": 3.15e10 / unit.meter,
    # Damping of the long-range interactions for r going to zero, to avoid singularities.
    "r0": 0.1 * unit.nanometer,
}


def build_nacl_lattice(
    ncell: int, density
) -> tuple[System, Topology, NDArray[int], NDArray[float]]:
    """Build a cubic cell with a sodium chloride lattice.

    Parameters
    ----------
    ncell
        Number if unit cells along each axis.
    density
        Density of the NaCl lattice, must have OpenMM units.

    Returns
    -------
    system
        The OpenMM system object, only contains the cell vectors.
    topology
        The topology defining the atoms in the system.
    atnums
        Numpy array with shape (natom,) containing the atomic numbers of the atoms.
    atcoords
        Numpy array with shape (natom, 3) containing the positions of the atoms.
    """
    natom_axis = 2 * ncell
    natom = natom_axis**3
    npair = natom // 2

    # Compute the cell size.
    na_element = Element.getByAtomicNumber(11)
    cl_element = Element.getByAtomicNumber(17)
    total_mass = (
        npair
        * (na_element.mass + cl_element.mass)
        / unit.dalton
        * unit.gram
        / unit.mole
        / unit.AVOGADRO_CONSTANT_NA
    )
    volume = total_mass / density
    cell_length = (volume ** (1 / 3) / unit.nanometer) * unit.nanometer
    cell_vecs = cell_length * np.identity(3)

    # The following loops are not vectorized for the sake of readability.
    # This is not a performance-critical part of the code.
    atnums = np.zeros(natom, dtype=int)
    atcoords = np.zeros((natom, 3))
    i = 0
    delta = cell_length / natom_axis / unit.nanometer
    for i0 in range(natom_axis):
        for i1 in range(natom_axis):
            for i2 in range(natom_axis):
                atnums[i] = 11 if (i0 + i1 + i2) % 2 == 0 else 17
                atcoords[i, 0] = (i0 + 0.5) * delta
                atcoords[i, 1] = (i1 + 0.5) * delta
                atcoords[i, 2] = (i2 + 0.5) * delta
                i += 1
    atcoords *= unit.nanometer

    # Create the OpenMM objects.
    system, topology = build_nacl(atnums, cell_vecs)

    return system, topology, atnums, atcoords


def build_nacl(atnums: NDArray[int], cell_vecs: NDArray) -> tuple[System, Topology]:
    """Low-level function to construct an NaCl system from atomic numbers and positions."""
    # Create OpenMM objects.
    system = System()
    system.setDefaultPeriodicBoxVectors(*cell_vecs)
    topology = Topology()
    topology.setPeriodicBoxVectors(cell_vecs)
    chain = topology.addChain()

    # Fill the system
    na_element = Element.getByAtomicNumber(11)
    cl_element = Element.getByAtomicNumber(17)
    for atnum in atnums:
        residue = topology.addResidue("ion", chain)
        if atnum == 11:
            topology.addAtom("Na", na_element, residue)
            system.addParticle(na_element.mass)
        elif atnum == 17:
            topology.addAtom("Cl", cl_element, residue)
            system.addParticle(cl_element.mass)
        else:
            raise ValueError(f"Unsupported atomic number {atnum}")
    return system, topology


def load_nacl(path_pdb: str):
    """Load a PDB file with NaCl coordinates and return the system and topology."""
    pdb = PDBFile(path_pdb)
    topology = pdb.topology
    system = System()
    system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    atnums = []
    for chain in topology.chains():
        for residue in chain.residues():
            for atom in residue.atoms():
                system.addParticle(atom.element.mass)
                if atom.element.symbol == "Na":
                    atnums.append(11)
                elif atom.element.symbol == "Cl":
                    atnums.append(17)
                else:
                    raise ValueError(f"Unsupported element {atom.element.symbol}")
    atnums = np.array(atnums)
    atcoords = pdb.getPositions(asNumpy=True)
    return system, topology, atnums, atcoords


def add_nacl_forces(
    system: System,
    topology: Topology,
    do_charge: bool = True,
    do_periodic: bool = True,
    cutoff: unit.Quantity | None = None,
):
    """Add the Tosi-Fumi potential to the system.

    Parameters
    ----------
    system
        The OpenMM system object to which the forces are added.
    topology
        The OpenMM topology object defining the atoms.
    do_charge
        If True, add a simple Coulomb interaction between the ions.
    cutoff
        The cutoff distance for the nonbonded interactions.
        If None, a cutoff is computed based on the cell vectors.

    Returns
    -------
    cutoff
        The cutoff distance used for the nonbonded interactions.
    """
    cell_vectors = np.array([vec / unit.nanometer for vec in topology.getPeriodicBoxVectors()])
    print(cell_vectors)
    if cell_vectors is None:
        raise ValueError("Topology does not contain periodic box vectors")
    for i, j in (1, 2), (2, 0), (0, 1):
        print(cell_vectors[i, j], cell_vectors[j, i])
        if cell_vectors[i, j] != 0:
            raise ValueError("Cell vectors are not orthogonal")
        if cell_vectors[j, i] != 0:
            raise ValueError("Cell vectors are not orthogonal")
    cell_spacing = np.diag(cell_vectors).min() * unit.nanometer
    cell_cutoff = cell_spacing * 0.45
    if cutoff is None:
        cutoff = cell_cutoff
    elif cutoff > cell_cutoff:
        raise ValueError("Cutoff is larger than 45% of the cell spacing")

    if do_charge:
        force_q = NonbondedForce()
        for atom in topology.atoms():
            force_q.addParticle(PARS[f"{atom.element.symbol.lower()}_charge"], 0.0, 0.0)
        force_q.setCutoffDistance(cutoff)
        force_q.setNonbondedMethod(NonbondedForce.PME if do_periodic else NonbondedForce.NoCutoff)
        force_q.setUseDispersionCorrection(False)
        system.addForce(force_q)

    # This custom force uses a trick employ parameters for which there are no mixing rules.
    # This is not trivially generalized to more than two elements,
    # nor is it intended to be transferrable in any way.
    force_r = CustomNonbondedForce(
        "b * c * exp(A*(sigma - r)) - C/(r06 + r^6) - D/(r08 + r^8); "
        "sigma = sigma1 + sigma2; "
        "c = c1*delta(i1-i2) + cm1*(1-delta(i1-i2)); "
        "C = C1*delta(i1-i2) + Cm1*(1-delta(i1-i2)); "
        "D = D1*delta(i1-i2) + Dm1*(1-delta(i1-i2)); "
        "r06 = r0^6; "
        "r08 = r0^8; "
    )
    force_r.addGlobalParameter("b", PARS["b"])
    force_r.addGlobalParameter("A", PARS["A"])
    force_r.addGlobalParameter("r0", PARS["r0"])
    force_r.addPerParticleParameter("sigma")
    force_r.addPerParticleParameter("i")
    force_r.addPerParticleParameter("c")
    force_r.addPerParticleParameter("cm")
    force_r.addPerParticleParameter("C")
    force_r.addPerParticleParameter("Cm")
    force_r.addPerParticleParameter("D")
    force_r.addPerParticleParameter("Dm")

    force_r.setCutoffDistance(cutoff)
    force_r.setUseSwitchingFunction(True)
    force_r.setSwitchingDistance(0.8 * cutoff)
    force_r.setNonbondedMethod(
        CustomNonbondedForce.CutoffPeriodic if do_periodic else CustomNonbondedForce.NoCutoff
    )
    force_r.setUseLongRangeCorrection(True)
    for atom in topology.atoms():
        if atom.element.symbol.lower() == "na":
            force_r.addParticle(
                [
                    PARS["sigma_na"],
                    # The following are part of the mixing rule workaround.
                    0.0,
                    PARS["c++"],
                    PARS["c+-"],
                    PARS["C_na_na"],
                    PARS["C_na_cl"],
                    PARS["D_na_na"],
                    PARS["D_na_cl"],
                ]
            )
        elif atom.element.symbol.lower() == "cl":
            force_r.addParticle(
                [
                    PARS["sigma_cl"],
                    # The following are part of the mixing rule workaround.
                    1.0,
                    PARS["c--"],
                    PARS["c+-"],
                    PARS["C_cl_cl"],
                    PARS["C_na_cl"],
                    PARS["D_cl_cl"],
                    PARS["D_na_cl"],
                ]
            )
        else:
            raise ValueError(f"Unsupported element {atom.element}")
    system.addForce(force_r)

    # Not really a force, but important enough anyway...
    system.addForce(CMMotionRemover())

    return cutoff


def _compute_energy_pair(
    r, z1: int, z2: int, do_coulomb: bool = True, do_damp: bool = True
) -> unit.Quantity:
    """Compute the BHMTF potential for a pair of ions, without using OpenMM."""
    z1, z2 = sorted([z1, z2])
    sign1 = "+" if z1 == 11 else "-"
    sign2 = "+" if z2 == 11 else "-"
    sym1 = "na" if z1 == 11 else "cl"
    sym2 = "na" if z2 == 11 else "cl"
    sigma = PARS[f"sigma_{sym1}"] + PARS[f"sigma_{sym2}"]
    energy = PARS["b"] * PARS["c" + sign1 + sign2] * np.exp(PARS["A"] * (sigma - r))
    r0 = PARS["r0"] if do_damp else (0.0 * unit.nanometer)
    energy -= PARS[f"C_{sym1}_{sym2}"] / (r0**6 + r**6)
    energy -= PARS[f"D_{sym1}_{sym2}"] / (r0**8 + r**8)
    if do_coulomb:
        energy += COULCONST_PER_MOLE * PARS[f"{sym1}_charge"] * PARS[f"{sym2}_charge"] / r
    return energy


@pytest.mark.parametrize("do_charge", [True, False])
@pytest.mark.parametrize("z1", [11, 17])
@pytest.mark.parametrize("z2", [11, 17])
def test_pair(do_charge: bool, z1: int, z2: int):
    """Test the Tosi-Fumi potential with a simple NaCl pair, surrounded by vacuum."""
    # Computing the energy of a NaCl pair in vacuum is not trivial,
    # essentially because OpenMM is not designed for this purpose,
    # but is possible.
    from openmm import VerletIntegrator

    cell_vecs = np.array([[50, 0, 0], [0, 50, 0], [0, 0, 50]]) * unit.nanometer
    system, topology = build_nacl([z1, z2], cell_vecs)
    atcoords = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 1]]) * unit.nanometer
    add_nacl_forces(system, topology, do_charge, do_periodic=(z1 != z2))
    integrator = VerletIntegrator(1 * unit.femtosecond)
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(atcoords)
    state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
    energy_openmm = state.getPotentialEnergy()

    # Compute the same energy in pure Python, neglecting periodic boundary conditions
    r = np.linalg.norm(atcoords[0] - atcoords[1]) * unit.nanometer
    energy_python = _compute_energy_pair(r, z1, z2, do_charge)

    # The energies should be close, but not exactly equal.
    print("Energy OpenMM", energy_openmm)
    print("Energy Python", energy_python)
    assert abs(energy_openmm - energy_python) < 0.01 * unit.kilojoule_per_mole


@pytest.mark.parametrize("do_charge", [True, False])
def test_cube(do_charge: bool):
    """Test the Tosi-Fumi potential with a simple NaCl cube, surrounded by vacuum."""
    # Computing the energy of a NaCl cube in vacuum is not trivial,
    # essentially because OpenMM is not designed for this purpose,
    # but is possible.
    from openmm import VerletIntegrator

    system, topology, atnums, atcoords = build_nacl_lattice(1, 1 * unit.gram / unit.centimeter**3)
    cell_vecs = np.array([[50, 0, 0], [0, 50, 0], [0, 0, 50]]) * unit.nanometer
    system.setDefaultPeriodicBoxVectors(*cell_vecs)
    topology.setPeriodicBoxVectors(cell_vecs)
    add_nacl_forces(system, topology, do_charge, do_periodic=True)
    integrator = VerletIntegrator(1 * unit.femtosecond)
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(atcoords)
    state = simulation.context.getState(getEnergy=True)
    energy_openmm = state.getPotentialEnergy()

    # Compute the same energy in pure Python, neglecting periodic boundary conditions
    energy_python = 0.0 * unit.kilojoule_per_mole
    for i in range(len(atcoords)):
        for j in range(i + 1, len(atcoords)):
            r = np.linalg.norm(atcoords[i] - atcoords[j]) * unit.nanometer
            energy_python += _compute_energy_pair(r, atnums[i], atnums[j], do_charge)

    # The energies should be close, but not exactly equal.
    print("Energy OpenMM", energy_openmm)
    print("Energy Python", energy_python)
    assert abs(energy_openmm - energy_python) < 0.01 * unit.kilojoule_per_mole


def plot_pair_potentials():
    """Plot pair potentials for the three possible combinations of Na and Cl."""
    import matplotlib.pyplot as plt

    rs = np.linspace(0.1, 1, 300) * unit.nanometer
    fig, ax = plt.subplots()
    for color, z1, z2 in ("C0", 11, 11), ("C1", 11, 17), ("C2", 17, 17):
        energies = [_compute_energy_pair(r, z1, z2) / unit.kilojoule_per_mole for r in rs]
        ax.plot(rs / unit.nanometer, energies, label=f"$Z_1={z1}$ $Z_2={z2}$", color=color)
        energies = [
            _compute_energy_pair(r, z1, z2, do_damp=False) / unit.kilojoule_per_mole for r in rs
        ]
        ax.plot(rs / unit.nanometer, energies, label="__no_legend__", ls=":", color=color)
    ax.legend()
    ax.set_xlabel("r [nm]")
    ax.set_ylabel("Energy [kJ/mol]")
    ax.set_ylim(-800, 3700)
    fig.savefig("pair_potentials.png")


if __name__ == "__main__":
    plot_pair_potentials()
