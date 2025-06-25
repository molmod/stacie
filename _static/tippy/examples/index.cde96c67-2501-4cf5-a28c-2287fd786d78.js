selector_to_html = {"a[href=\"https://doi.org/10.5281/zenodo.15543903\"]": "\n<div>\n    <h3>Transport and Helfand moments in the Lennard-Jones fluid. I. Shear viscosity</h3>\n    \n    <p><b>Authors:</b> S. Viscardy, J. Servantie, P. Gaspard</p>\n    \n    <p><b>Publisher:</b> AIP Publishing</p>\n    <p><b>Published:</b> 2007-5-14</p>\n</div>", "a[href=\"water_diffusivity.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Diffusivity of TIP4P Water At Ambient Conditions (Gromacs)<a class=\"headerlink\" href=\"#diffusivity-of-tip4p-water-at-ambient-conditions-gromacs\" title=\"Link to this heading\">\u00b6</a></h1><h2>Library Imports and Matplotlib Configuration<a class=\"headerlink\" href=\"#library-imports-and-matplotlib-configuration\" title=\"Link to this heading\">\u00b6</a></h2><p>Manually fixed numbers of water molecules:</p>", "a[href=\"#worked-examples\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Worked Examples<a class=\"headerlink\" href=\"#worked-examples\" title=\"Link to this heading\">\u00b6</a></h1><p>All the examples are also available as Jupyter notebooks and can be downloaded as one ZIP archive here:</p>"}
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
