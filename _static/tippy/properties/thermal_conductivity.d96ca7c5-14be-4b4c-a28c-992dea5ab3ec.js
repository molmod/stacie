selector_to_html = {"a[href=\"../examples/lj_thermal_conductivity.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Thermal Conductivity of a Lennard-Jones Liquid Near the Triple Point (LAMMPS)<a class=\"headerlink\" href=\"#thermal-conductivity-of-a-lennard-jones-liquid-near-the-triple-point-lammps\" title=\"Link to this heading\">\u00b6</a></h1><p>This example shows how to derive the thermal conductivity\nusing heat flux data from a LAMMPS simulation.\nIt uses the same production runs and conventions\nas in the <a class=\"reference internal\" href=\"../examples/lj_shear_viscosity.html\"><span class=\"std std-doc\">Shear viscosity example</span></a>.\nThe required theoretical background is explained the section\n<a class=\"reference internal\" href=\"../properties/thermal_conductivity.html\"><span class=\"std std-doc\">Thermal Conductivity</span></a>.</p>", "a[href=\"#how-to-compute-with-stacie\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">How to Compute with STACIE?<a class=\"headerlink\" href=\"#how-to-compute-with-stacie\" title=\"Link to this heading\">\u00b6</a></h2><p>It is assumed that you can load the time-dependent heat flux components\ninto a 2D NumPy array <code class=\"docutils literal notranslate\"><span class=\"pre\">heatflux</span></code>,\nwhere each column corresponds to a time step.\nEach row corresponds to a single heat flux component:\n<span class=\"math notranslate nohighlight\">\\(\\hat{J}_x\\)</span>, <span class=\"math notranslate nohighlight\">\\(\\hat{J}_y\\)</span>, and <span class=\"math notranslate nohighlight\">\\(\\hat{J}_z\\)</span>.</p><p>You also need to store the cell volume, temperature,\nBoltzmann constant, and time step in Python variables,\nall in consistent units.\nWith these requirements, the thermal conductivity can be computed as follows:</p>", "a[href=\"#thermal-conductivity\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Thermal Conductivity<a class=\"headerlink\" href=\"#thermal-conductivity\" title=\"Link to this heading\">\u00b6</a></h1><p>The thermal conductivity of a system is related to the autocorrelation\nof the heat flux as follows:</p>"}
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
