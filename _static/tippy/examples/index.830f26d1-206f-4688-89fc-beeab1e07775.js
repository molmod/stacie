selector_to_html = {"a[href=\"https://doi.org/10.5281/zenodo.15543903\"]": "\n<div>\n    <h3>Transport and Helfand moments in the Lennard-Jones fluid. I. Shear viscosity</h3>\n    \n    <p><b>Authors:</b> S. Viscardy, J. Servantie, P. Gaspard</p>\n    \n    <p><b>Publisher:</b> AIP Publishing</p>\n    <p><b>Published:</b> 2007-5-14</p>\n</div>", "a[href=\"lj_bulk_viscosity.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Bulk Viscosity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)<a class=\"headerlink\" href=\"#bulk-viscosity-of-a-lennard-jones-liquid-near-the-triple-point-lammps\" title=\"Link to this heading\">\u00b6</a></h1><p>This example demonstrates how to compute the bulk viscosity\nof a Lennard-Jones liquid near its triple point using LAMMPS.\nIt uses the same production runs and conventions\nas in the <a class=\"reference internal\" href=\"lj_shear_viscosity.html\"><span class=\"std std-doc\">Shear viscosity example</span></a>.\nThe required theoretical background is explained the section\n<a class=\"reference internal\" href=\"../properties/bulk_viscosity.html\"><span class=\"std std-doc\">Bulk Viscosity</span></a>.\nIn essence, it is computed in the same way as the shear viscosity,\nexcept that the isotropic pressure fluctuations are used as input.</p>", "a[href=\"#worked-examples\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Worked Examples<a class=\"headerlink\" href=\"#worked-examples\" title=\"Link to this heading\">\u00b6</a></h1><p>All the examples are also available as Jupyter notebooks and can be downloaded as one ZIP archive here:</p>"}
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
