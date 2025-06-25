selector_to_html = {"a[href=\"https://doi.org/10.5281/zenodo.15543903\"]": "\n<div>\n    <h3>Transport and Helfand moments in the Lennard-Jones fluid. I. Shear viscosity</h3>\n    \n    <p><b>Authors:</b> S. Viscardy, J. Servantie, P. Gaspard</p>\n    \n    <p><b>Publisher:</b> AIP Publishing</p>\n    <p><b>Published:</b> 2007-5-14</p>\n</div>", "a[href=\"lj_shear_viscosity.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Shear Viscosity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)<a class=\"headerlink\" href=\"#shear-viscosity-of-a-lennard-jones-liquid-near-the-triple-point-lammps\" title=\"Link to this heading\">\u00b6</a></h1><p>This example shows how to calculate viscosity of argon\nfrom pressure tensor data obtained from <a class=\"reference internal\" href=\"../glossary.html#term-LAMMPS\"><span class=\"xref std std-term\">LAMMPS</span></a> <a class=\"reference internal\" href=\"../glossary.html#term-MD\"><span class=\"xref std std-term\">MD</span></a> simulations.\nThe required theoretical background is explained the\n<a class=\"reference internal\" href=\"../properties/shear_viscosity.html\"><span class=\"std std-doc\">Shear Viscosity</span></a> section.\nThe same simulations are also used for the <a class=\"reference internal\" href=\"lj_bulk_viscosity.html\"><span class=\"std std-doc\">bulk viscosity</span></a>\nand <a class=\"reference internal\" href=\"lj_thermal_conductivity.html\"><span class=\"std std-doc\">thermal conductivity</span></a> examples in the following two notebooks.\nThe goal of the argon examples is to derive the three transport properties\nwith a relative error smaller than those found in the literature.</p><p>All argon MD simulations use the\n<a class=\"reference external\" href=\"https://en.wikipedia.org/wiki/Lennard-Jones_potential\">Lennard-Jones potential</a>\nwith reduced Lennard-Jones units.\nFor example, the reduced unit of viscosity is denoted as \u03b7*,\nand the reduced unit of time as \u03c4*.\nThe simulated system consists of 1372 argon atoms.\nThe thermodynamic state <span class=\"math notranslate nohighlight\">\\(\\rho=0.8442\\,\\mathrm{\\rho}^*\\)</span> and <span class=\"math notranslate nohighlight\">\\(T=0.722\\,\\mathrm{T}^*\\)</span>\ncorresponds to a liquid phase near the triple point\n(<span class=\"math notranslate nohighlight\">\\(\\rho=0.0845\\,\\mathrm{\\rho}^*\\)</span> and <span class=\"math notranslate nohighlight\">\\(T=0.69\\,\\mathrm{T}^*\\)</span>).\nThis liquid state is known to exhibit slow relaxation times,\nwhich complicates the convergence of transport properties and\nmakes it a popular test case for computational methods.</p>", "a[href=\"#worked-examples\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Worked Examples<a class=\"headerlink\" href=\"#worked-examples\" title=\"Link to this heading\">\u00b6</a></h1><p>All the examples are also available as Jupyter notebooks and can be downloaded as one ZIP archive here:</p>"}
skip_classes = ["headerlink", "sd-stretched-link"]

window.onload = function () {
    for (const [select, tip_html] of Object.entries(selector_to_html)) {
        const links = document.querySelectorAll(` ${select}`);
        for (const link of links) {
            if (skip_classes.some(c => link.classList.contains(c))) {
                continue;
            }

            tippy(link, {
                content: tip_html,
                allowHTML: true,
                arrow: true,
                placement: 'auto-start', maxWidth: 500, interactive: false,

            });
        };
    };
    console.log("tippy tips loaded!");
};
