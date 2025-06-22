#show link: set text(blue)
#set page("a4", margin: 2cm)

#align(center)[
  #text(size: 24pt)[
    *Output files of simulations analyzed in the examples in STACIE's documentation*
  ]

  Gözdenur Toraman#super[†] and Toon Verstraelen#super[✶¶]

  † Soete Laboratory, Ghent University, Technologiepark-Zwijnaarde 46, 9052 Ghent, Belgium\
  ¶ Center for Molecular Modeling (CMM), Ghent University, Technologiepark-Zwijnaarde
  46, B-9052, Ghent, Belgium

  ✶E-mail: #link("mailto:toon.verstraelen@ugent.be", "toon.verstraelen@ugent.be")
]

== Usage

To run the example notebooks, you need to:

1. Install STACIE and Jupyter Lab

    ```bash
    pip install stacie jupyterlab
    ```

2. Download and unpack the archive with notebooks and trajectory data.

    ```bash
    unzip examples.zip
    ```

3. Finally, you should be able to start Jupyter Lab and run the notebooks.

    ```bash
    jupyter lab
    ```

== Overview of included files

There are currently two sets of Molecular Dynamics (MD) simulations,
which are used as inputs:

- `lammps_lj3d`: LAMMPS simulations of Lennard-Jones 3D systems
- `openmm_salt`: OpenMM simulations of molten salt systems
