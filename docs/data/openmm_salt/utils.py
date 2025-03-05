"""Utility functions for the two simulation notebooks: exploration.ipynb and production.ipynb."""

import sys

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from openmm import LangevinIntegrator, MonteCarloBarostat, System, VerletIntegrator, unit
from openmm.app import DCDReporter, PDBFile, Simulation, StateDataReporter, Topology

__all__ = ("make_plots", "runmd")


def runmd(
    prefix: str,
    system: System,
    topology: Topology,
    atcoords: NDArray,
    nstep: int,
    timestep: unit.Quantity,
    stride: int,
    *,
    temperature: unit.Quantity | None = None,
    pressure: unit.Quantity | None = None,
    atvels: NDArray | None = None,
    tau_thermostat: unit.Quantity = 1 * unit.picosecond,
    seed: int = 42,
    do_opt: bool = False,
) -> tuple[NDArray, NDArray]:
    """Simulate the system for a fixed number of steps.

    Parameters
    ----------
    prefix
        The prefix of the output files.
    system
        The OpenMM system object.
    topology
        The OpenMM topology object.
    atcoords
        The initial coordinates of the atoms.
    nstep
        The number of steps to simulate.
    timestep
        The time step of the simulation.
    stride
        The number of steps between writing frames to the trajectory file.
    temperature
        The temperature of the simulation in Kelvin.
        If None, the simulation will be performed in the NVE ensemble.
    pressure
        The pressure of the simulation in bar. If None, The volume is kept constant.
        (Note that the initial volume is a property of the topology.)
    atvels
        If given, these initial velocities will be used.
    tau_thermostat
        The relaxation time of the thermostat.
    seed
        The seed for the random number generator.
    do_opt
        If True, the energy will be minimized before the simulation.
        This can be useful to remove any bad contacts in the initial structure.

    Returns
    -------
    atcoords
        The final coordinates of the atoms.
    atvels
        The final velocities of the atoms.
    """
    # Check if the system has old "forces" and remove the ones we don't want.
    for ifrc in range(system.getNumForces() - 1, -1, -1):
        force = system.getForce(ifrc)
        if isinstance(force, MonteCarloBarostat):
            system.removeForce(ifrc)

    # Define the ensemble to be simulated in.
    if temperature is None:
        integrator = VerletIntegrator(timestep)
    else:
        integrator = LangevinIntegrator(temperature, 1 / tau_thermostat, timestep)
    if pressure is not None:
        mcb = MonteCarloBarostat(pressure, temperature)
        mcb.setRandomNumberSeed(seed + 746)
        system.addForce(mcb)

    # Define a simulation object.
    sim = Simulation(topology, system, integrator)
    if pressure is None:
        sim.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())

    # Write a frame to the DCD trajectory every 1 steps.
    sim.reporters.append(DCDReporter(f"output/{prefix}_traj.dcd", stride, enforcePeriodicBox=False))

    # Write scalar properties to a CSV file every 10 steps.
    sim.reporters.append(
        StateDataReporter(
            f"output/{prefix}_scalars.csv",
            stride,
            time=True,
            potentialEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
        )
    )

    # Write scalar properties to screen every 1000 steps.
    sim.reporters.append(
        StateDataReporter(
            sys.stdout,
            stride,
            step=True,
            temperature=True,
            volume=True,
            remainingTime=False,
            separator="\t",
        )
    )

    # Prepare the initial state.
    sim.context.reinitialize(True)
    sim.context.setPositions(atcoords)
    if do_opt:
        sim.minimizeEnergy()
    if atvels is not None:
        sim.context.setVelocities(atvels)
    elif temperature is not None:
        sim.context.setVelocitiesToTemperature(temperature, seed + 7973)

    # Actually run the molecular dynamics simulation.
    sim.step(nstep)

    # Write the final coordinates, and get velocities and cell vectors.
    state = sim.context.getState(getPositions=True, getVelocities=True)
    atcoords = state.getPositions(asNumpy=True)
    atvels = state.getVelocities(asNumpy=True)
    if pressure is not None:
        # Store possibly updated box vectors.
        topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    with open(f"output/{prefix}_final.pdb", "w") as f:
        PDBFile.writeFile(topology, atcoords, f)

    _convert_to_npz(prefix)

    return atcoords, atvels


def _convert_to_npz(prefix: str):
    """Convert the trajectory to an NPZ file for easier analysis.

    In the NPZ files, all quantities are stored in SI units.
    """
    traj = md.load(f"output/{prefix}_traj.dcd", top=f"output/{prefix}_final.pdb")
    atums = np.array([a.element.number for a in traj.top.atoms])
    data = np.loadtxt(f"output/{prefix}_scalars.csv", skiprows=1, delimiter=",")
    temperature = data[:, 3].mean()
    volume = data[:, 4].mean() * 1e-27
    time = data[:, 0] * 1e-12
    np.savez(
        f"output/{prefix}_traj.npz",
        atcoords=traj.xyz * 1e-9,
        time=time,
        atnums=atums,
        temperature=temperature,
        volume=volume,
    )


def make_plots(prefix: str):
    """Make plots of the temperature, energy, and volume of the simulation."""
    df = pd.read_csv(f"output/{prefix}_scalars.csv")

    plt.close(f"{prefix}_temperature")
    _, ax = plt.subplots(num=f"{prefix}_temperature")
    df.plot(kind="line", x='#"Time (ps)"', y="Temperature (K)", ax=ax)

    plt.close(f"{prefix}_energy")
    fig, ax = plt.subplots(num=f"{prefix}_energy")
    df.plot(kind="line", x='#"Time (ps)"', y="Total Energy (kJ/mole)", ax=ax)
    df.plot(kind="line", x='#"Time (ps)"', y="Potential Energy (kJ/mole)", ax=ax)

    plt.close(f"{prefix}_volume")
    fig, ax = plt.subplots(num=f"{prefix}_volume")
    df.plot(kind="line", x='#"Time (ps)"', y="Box Volume (nm^3)", ax=ax)
