# OpenMM Molten Salt Simulations

All OpenMM simulations can be executed efficiently on a single compute node with StepUp.
You can install StepUp as follows:

```bash
pip install stepup stepup-reprep
```

It is also assumed that you have OpenMM, MDTraj, NGLView, NumPy and Pandas installed,
which is part of the documentation dependencies of Stacie.
You can install all required dependencies as follows:

```bash
pip install stacie[docs,tests]
```

Afterwards, you can run the OpenMM simulations as follows:

```bash
OPENMM_CPU_THREADS=1 stepup
```

Because StepUp already executes the workflow in parallel, multithreading in OpenMM is disabled.
The inputs were tested with OpenMM 8.2.0.
