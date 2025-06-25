selector_to_html = {"a[href=\"https://doi.org/10.1021/acs.jcim.5c01475\"]": "\n<div>\n    <h3>STable AutoCorrelation Integral Estimator: Robust and Accurate Transport Properties from Molecular Dynamics Simulations</h3>\n    \n    <p><b>Authors:</b> G\u00f6zdenur Toraman, Dieter Fauconnier, Toon Verstraelen</p>\n    \n    <p><b>Publisher:</b> American Chemical Society (ACS)</p>\n    <p><b>Published:</b> 2025-9-16</p>\n</div>", "a[href=\"#how-to-cite\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">How to Cite<a class=\"headerlink\" href=\"#how-to-cite\" title=\"Link to this heading\">\u00b6</a></h1><p>When using STACIE in your research, please cite the STACIE paper in any resulting publication.\nThe reference is provided in several formats below:</p>", "a[href=\"#main-stacie-paper\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Main STACIE Paper<a class=\"headerlink\" href=\"#main-stacie-paper\" title=\"Link to this heading\">\u00b6</a></h2><p>This paper introduces STACIE and should be cited in any publication that relies on STACIE.\nThe manuscript has been submitted to The Journal of Chemical Information and Modeling,\nand the citation records below will be updated when appropriate.</p>", "a[href=\"#shear-viscosity-calculations\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Shear Viscosity Calculations<a class=\"headerlink\" href=\"#shear-viscosity-calculations\" title=\"Link to this heading\">\u00b6</a></h2><p>The following paper describes in detail the calculation of shear viscosity with STACIE.</p>"}
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
