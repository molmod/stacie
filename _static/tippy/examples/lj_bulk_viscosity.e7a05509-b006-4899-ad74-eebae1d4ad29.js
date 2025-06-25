selector_to_html = {"a[href=\"#analysis-of-the-production-simulations\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Analysis of the Production Simulations<a class=\"headerlink\" href=\"#analysis-of-the-production-simulations\" title=\"Link to this heading\">\u00b6</a></h2><p>The following code cells define analysis functions used below.</p>", "a[href=\"#regression-tests\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Regression Tests<a class=\"headerlink\" href=\"#regression-tests\" title=\"Link to this heading\">\u00b6</a></h2><p>If you are experimenting with this notebook, you can ignore any exceptions below.\nThe tests are only meant to pass for the notebook in its original form.</p>", "a[href=\"#comparison-to-literature-results\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Comparison to Literature Results<a class=\"headerlink\" href=\"#comparison-to-literature-results\" title=\"Link to this heading\">\u00b6</a></h2><p>Computational estimates of the bulk viscosity of a Lennard-Jones fluid\ncan be found in <span id=\"id1\">[<a class=\"reference internal\" href=\"../references.html#id20\" title=\"Karsten Meier, Arno Laesecke, and Stephan Kabelac. Transport coefficients of the lennard-jones model fluid. iii. bulk viscosity. J. Chem. Phys., December 2004. URL: http://dx.doi.org/10.1063/1.1828040, doi:10.1063/1.1828040.\">MLK04b</a>]</span>.\nSince the simulation settings (<span class=\"math notranslate nohighlight\">\\(r_\\text{cut}^{*}=2.5\\)</span>, <span class=\"math notranslate nohighlight\">\\(N=1372\\)</span>, <span class=\"math notranslate nohighlight\">\\(T^*=0.722\\)</span> and <span class=\"math notranslate nohighlight\">\\(\\rho^{*}=0.8442\\)</span>)\nare identical to those used in this notebook, the reported values should be directly comparable.</p>", "a[href=\"#library-imports-and-matplotlib-configuration\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Library Imports and Matplotlib Configuration<a class=\"headerlink\" href=\"#library-imports-and-matplotlib-configuration\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"#bulk-viscosity-of-a-lennard-jones-liquid-near-the-triple-point-lammps\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Bulk Viscosity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)<a class=\"headerlink\" href=\"#bulk-viscosity-of-a-lennard-jones-liquid-near-the-triple-point-lammps\" title=\"Link to this heading\">\u00b6</a></h1><p>This example demonstrates how to compute the bulk viscosity\nof a Lennard-Jones liquid near its triple point using LAMMPS.\nIt uses the same production runs and conventions\nas in the <a class=\"reference internal\" href=\"lj_shear_viscosity.html\"><span class=\"std std-doc\">Shear viscosity example</span></a>.\nThe required theoretical background is explained the section\n<a class=\"reference internal\" href=\"../properties/bulk_viscosity.html\"><span class=\"std std-doc\">Bulk Viscosity</span></a>.\nIn essence, it is computed in the same way as the shear viscosity,\nexcept that the isotropic pressure fluctuations are used as input.</p>"}
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
