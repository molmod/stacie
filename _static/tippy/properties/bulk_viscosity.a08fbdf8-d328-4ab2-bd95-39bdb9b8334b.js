selector_to_html = {"a[href=\"#how-to-compute-with-stacie\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">How to Compute with STACIE?<a class=\"headerlink\" href=\"#how-to-compute-with-stacie\" title=\"Link to this heading\">\u00b6</a></h2><p>It is assumed that you can load the diagonal, time-dependent pressure tensor components\ninto a 2D NumPy array <code class=\"docutils literal notranslate\"><span class=\"pre\">pcomps</span></code>.\n(The same array as for <a class=\"reference internal\" href=\"shear_viscosity.html\"><span class=\"std std-doc\">shear viscosity</span></a> can be used.)\nEach row of this array corresponds to one pressure tensor component in the order\n<span class=\"math notranslate nohighlight\">\\(\\hat{P}_{xx}\\)</span>, <span class=\"math notranslate nohighlight\">\\(\\hat{P}_{yy}\\)</span>, <span class=\"math notranslate nohighlight\">\\(\\hat{P}_{zz}\\)</span>, <span class=\"math notranslate nohighlight\">\\(\\hat{P}_{zx}\\)</span>, <span class=\"math notranslate nohighlight\">\\(\\hat{P}_{yz}\\)</span>, <span class=\"math notranslate nohighlight\">\\(\\hat{P}_{xy}\\)</span>.\n(This is the same order as in Voigt notation. The last three components are not used and can be omitted.)\nColumns correspond to time steps.\nYou also need to store the cell volume, temperature,\nBoltzmann constant, and time step in Python variables,\nall in consistent units.\nWith these requirements, the bulk viscosity can be computed as follows:</p>", "a[href=\"shear_viscosity.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Shear Viscosity<a class=\"headerlink\" href=\"#shear-viscosity\" title=\"Link to this heading\">\u00b6</a></h1><p>The shear viscosity of a fluid is related to the autocorrelation\nof microscopic off-diagonal pressure tensor fluctuations as follows:</p>", "a[href=\"../examples/lj_bulk_viscosity.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Bulk Viscosity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)<a class=\"headerlink\" href=\"#bulk-viscosity-of-a-lennard-jones-liquid-near-the-triple-point-lammps\" title=\"Link to this heading\">\u00b6</a></h1><p>This example demonstrates how to compute the bulk viscosity\nof a Lennard-Jones liquid near its triple point using LAMMPS.\nIt uses the same production runs and conventions\nas in the <a class=\"reference internal\" href=\"../examples/lj_shear_viscosity.html\"><span class=\"std std-doc\">Shear viscosity example</span></a>.\nThe required theoretical background is explained the section\n<a class=\"reference internal\" href=\"../properties/bulk_viscosity.html\"><span class=\"std std-doc\">Bulk Viscosity</span></a>.\nIn essence, it is computed in the same way as the shear viscosity,\nexcept that the isotropic pressure fluctuations are used as input.</p>", "a[href=\"#bulk-viscosity\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Bulk Viscosity<a class=\"headerlink\" href=\"#bulk-viscosity\" title=\"Link to this heading\">\u00b6</a></h1><p>The bulk viscosity of a fluid is related to the autocorrelation\nof isotropic pressure fluctuations as follows:</p>"}
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
