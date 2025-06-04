# Worked Examples

All the examples are also available as Jupyter notebooks and can be downloaded as one ZIP archive here:

> <https://doi.org/10.5281/zenodo.15543903>

:::{warning}
The ZIP file will contain executable notebooks and simulation outputs used by the examples.
Hyperlinks from these notebooks to the rest of the documentation
and literature references will not work.
:::

This documentation contains the rendered notebooks, including all outputs, in the following sections.
We recommend starting with the minimal example, as it is the easiest to run and understand.
This example thoroughly explains STACIE's output and how to interpret the plots.
The other examples produce similar outputs and plots,
but the meaning of all outputs is not repeated in each example.

The first few notebooks are completely self-contained.
They generate the data and analyze it with STACIE:

```{toctree}
:maxdepth: 1

minimal.py
error_mean.py
applicability.py
surface_diffusion.py
```

The remaining notebooks process the output of external simulation codes.
Input files for these simulations can be found in the Git source repository of STACIE.
You can rerun these simulations to generate the required data
or use the data files from the ZIP archive mentioned above.

```{toctree}
:maxdepth: 1

lj_shear_viscosity.py
lj_bulk_viscosity.py
lj_thermal_conductivity.py
molten_salt.py
```

Some notebooks also use helper functions from the [`utils.py`](utils.py) module.

```{toctree}
:maxdepth: 1
:hidden:

utils.py
```
