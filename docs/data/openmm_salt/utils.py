"""Utility functions for the two simulation notebooks: exploration.ipynb and production.ipynb."""

import sys

import matplotlib.pyplot as plt
import pandas as pd
from numpy.typing import NDArray
from openmm import LangevinIntegrator, MonteCarloBarostat, System, VerletIntegrator, unit
from openmm.app import DCDReporter, PDBFile, Simulation, StateDataReporter, Topology

__all__ = ("make_plots", "runmd")


def runmd(
    prefix: str,
    system: System,
    topology: Topology,
    nstep: int,
    timestep: unit.Quantity,
    stride: int,
    *,
    load_checkpoint_from: str | None = None,
    atcoords: NDArray | None = None,
    atvels: NDArray | None = None,
    temperature: unit.Quantity | None = None,
    pressure: unit.Quantity | None = None,
    tau_thermostat: unit.Quantity = 1 * unit.picosecond,
    seed: int | None = None,
    reset_stepcounter: bool = False,
):
    """Simulate the system for a fixed number of steps.

    Parameters
    ----------
    prefix
        The prefix of the output files.
    system
        The OpenMM system object.
    topology
        The OpenMM topology object.
    nstep
        The number of steps to simulate.
    timestep
        The time step of the simulation.
    stride
        The number of steps between writing frames to the trajectory file.
    load_checkpoint_from
        If given, the initial state is loaded from this file.
        This overwrites the volume loaded from the topology.
    atcoords
        The initial coordinates of the atoms.
        Ignored if load_checkpoint_from is given.
    atvels
        If given, these initial velocities will be used.
        Ignored if load_checkpoint_from is given.
    temperature
        The temperature of the simulation in Kelvin.
        If None, the simulation will be performed in the NVE ensemble.
    pressure
        The pressure of the simulation in bar. If None, The volume is kept constant.
        (Note that the initial volume is a property of the topology.)
    tau_thermostat
        The relaxation time of the thermostat.
    seed
        The seed for the random number generator.
    reset_stepcounter
        If True, the step counter of the simulation will be reset to 0.
        This is useful if the simulation is continued from a checkpoint file.
    """
    if temperature is None:
        print("Verlet")
        integrator = VerletIntegrator(timestep)
    else:
        print("Langevin")
        integrator = LangevinIntegrator(temperature, 1 / tau_thermostat, timestep)
        integrator.setRandomNumberSeed(seed + 9273)

    # Define a simulation object.
    sim = Simulation(topology, system, integrator)

    # Add "forces" to the system.
    if pressure is not None:
        if temperature is None:
            raise ValueError("Pressure can only be set if temperature is also set.")
        mcb = MonteCarloBarostat(pressure, temperature)
        mcb.setRandomNumberSeed(seed + 746)
        system.addForce(mcb)

    # Write a frame to the DCD trajectory every 10 steps.
    sim.reporters.append(DCDReporter(f"output/{prefix}_traj.dcd", stride, enforcePeriodicBox=False))

    # Write scalar properties to a CSV file every 10 steps.
    sdrf = StateDataReporter(
        f"output/{prefix}_traj.csv",
        stride,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
    )
    sim.reporters.append(sdrf)

    # Write scalar properties to screen every 1000 steps.
    sdrs = StateDataReporter(
        sys.stdout,
        1000,
        step=True,
        temperature=True,
        volume=True,
        remainingTime=False,
        separator="\t",
    )
    sim.reporters.append(sdrs)

    sim.context.reinitialize(True)
    if load_checkpoint_from is not None:
        # Load the simulation state from a state file, if given.
        sim.loadCheckpoint(load_checkpoint_from)
    else:
        # Prepare the initial state.
        if pressure is None:
            sim.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
        if atcoords is not None:
            sim.context.setPositions(atcoords)
        if atvels is not None:
            sim.context.setVelocities(atvels)
        elif temperature is not None:
            sim.context.setVelocitiesToTemperature(temperature, seed + 32157)

    # Print the initial state, to validate the initial conditions.
    state = sim.context.getState(energy=True, parameters=True, positions=True, velocities=True)
    sdrf.report(sim, state)
    sdrs.report(sim, state)

    # Run the MD simulations
    if reset_stepcounter:
        sim.currentStep = 0
    sim.step(nstep)

    # Write the final coordinates, and get velocities and cell vectors.
    state = sim.context.getState(positions=True)
    if pressure is not None:
        # Store possibly updated box vectors.
        topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    atcoords = state.getPositions(asNumpy=True)
    with open(f"output/{prefix}_last.pdb", "w") as f:
        PDBFile.writeFile(topology, atcoords, f)

    # Remove the MCBarostat from the system, if it was added.
    for ifrc in range(system.getNumForces() - 1, -1, -1):
        force = system.getForce(ifrc)
        if isinstance(force, MonteCarloBarostat):
            system.removeForce(ifrc)

    # Write the final state to a file.
    sim.saveCheckpoint(f"output/{prefix}_last.chk")


def make_plots(prefix: str):
    """Make plots of the temperature, energy, and volume of the simulation."""
    df = pd.read_csv(f"output/{prefix}_traj.csv")

    plt.close(f"{prefix}_temperature")
    _, ax = plt.subplots(num=f"{prefix}_temperature")
    df.plot(kind="line", x='#"Time (ps)"', y="Temperature (K)", ax=ax)

    plt.close(f"{prefix}_energy")
    _, ax = plt.subplots(num=f"{prefix}_energy")
    df.plot(kind="line", x='#"Time (ps)"', y="Total Energy (kJ/mole)", ax=ax)
    df.plot(kind="line", x='#"Time (ps)"', y="Potential Energy (kJ/mole)", ax=ax)

    plt.close(f"{prefix}_volume")
    _, ax = plt.subplots(num=f"{prefix}_volume")
    df.plot(kind="line", x='#"Time (ps)"', y="Box Volume (nm^3)", ax=ax)
