selector_to_html = {"a[href=\"https://doi.org/10.1021/acs.jcim.5c01475\"]": "\n<div>\n    <h3>STable AutoCorrelation Integral Estimator: Robust and Accurate Transport Properties from Molecular Dynamics Simulations</h3>\n    \n    <p><b>Authors:</b> G\u00f6zdenur Toraman, Dieter Fauconnier, Toon Verstraelen</p>\n    \n    <p><b>Publisher:</b> American Chemical Society (ACS)</p>\n    <p><b>Published:</b> 2025-9-16</p>\n</div>", "a[href=\"getting_started/cite.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">How to Cite<a class=\"headerlink\" href=\"#how-to-cite\" title=\"Link to this heading\">\u00b6</a></h1><p>When using STACIE in your research, please cite the STACIE paper in any resulting publication.\nThe reference is provided in several formats below:</p>", "a[href=\"#welcome-to-stacies-documentation\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Welcome to STACIE\u2019s Documentation<a class=\"headerlink\" href=\"#welcome-to-stacies-documentation\" title=\"Link to this heading\">\u00b6</a></h1><p>STACIE is a Python package and algorithm that computes time integrals of autocorrelation functions.\nIt is primarily designed for post-processing molecular dynamics simulations.\nHowever, it can also be used for more general analysis of time-correlated data.\nTypical applications include estimating transport properties\nand the uncertainty of averages over time-correlated data,\nas well as analyzing characteristic timescales.</p><p><img alt=\"Graphical Summary\" src=\"_images/github_repo_card_dark.png\"/></p>"}
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
