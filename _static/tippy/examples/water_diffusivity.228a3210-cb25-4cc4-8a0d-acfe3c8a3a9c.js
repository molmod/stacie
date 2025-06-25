selector_to_html = {"a[href=\"#percentiles-of-the-instantenous-temperature-distribution\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Percentiles of the Instantenous Temperature Distribution<a class=\"headerlink\" href=\"#percentiles-of-the-instantenous-temperature-distribution\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"#analysis-of-correlations-between-atomic-displacements\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Analysis of correlations between atomic displacements<a class=\"headerlink\" href=\"#analysis-of-correlations-between-atomic-displacements\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"#diffusivity-of-tip4p-water-at-ambient-conditions-gromacs\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Diffusivity of TIP4P Water At Ambient Conditions (Gromacs)<a class=\"headerlink\" href=\"#diffusivity-of-tip4p-water-at-ambient-conditions-gromacs\" title=\"Link to this heading\">\u00b6</a></h1><h2>Library Imports and Matplotlib Configuration<a class=\"headerlink\" href=\"#library-imports-and-matplotlib-configuration\" title=\"Link to this heading\">\u00b6</a></h2><p>Manually fixed numbers of water molecules:</p>", "a[href=\"#validation-of-the-conserved-quantity\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Validation of the Conserved Quantity<a class=\"headerlink\" href=\"#validation-of-the-conserved-quantity\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"#cumulative-temperature-distribution\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Cumulative Temperature Distribution<a class=\"headerlink\" href=\"#cumulative-temperature-distribution\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"#id3\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Percentiles of the Instantenous Temperature Distribution<a class=\"headerlink\" href=\"#id3\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"#library-imports-and-matplotlib-configuration\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Library Imports and Matplotlib Configuration<a class=\"headerlink\" href=\"#library-imports-and-matplotlib-configuration\" title=\"Link to this heading\">\u00b6</a></h2><p>Manually fixed numbers of water molecules:</p>", "a[href=\"#diffusivity-with-uncertainty-quantification\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Diffusivity with Uncertainty Quantification<a class=\"headerlink\" href=\"#diffusivity-with-uncertainty-quantification\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"#extrapolation-to-infinite-box-size\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Extrapolation to Infinite Box Size<a class=\"headerlink\" href=\"#extrapolation-to-infinite-box-size\" title=\"Link to this heading\">\u00b6</a></h2>"}
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
