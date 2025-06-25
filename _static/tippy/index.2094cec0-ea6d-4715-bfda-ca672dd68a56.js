selector_to_html = {"a[href=\"#welcome-to-stacies-documentation\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Welcome to STACIE\u2019s Documentation<a class=\"headerlink\" href=\"#welcome-to-stacies-documentation\" title=\"Link to this heading\">\u00b6</a></h1><p>STACIE is a <em>STable AutoCorrelation Integral Estimator</em>.</p><p>STACIE is developed in the context of a collaboration between\nthe <a class=\"reference external\" href=\"https://molmod.ugent.be/\">Center for Molecular Modeling</a>\nand the tribology group of <a class=\"reference external\" href=\"https://www.ugent.be/ea/emsme/en/research/soete\">Labo Soete</a>\nat <a class=\"reference external\" href=\"https://ugent.be/\">Ghent University</a>.\nSTACIE is open-source software (LGPL-v3 license) and is available on\n<a class=\"reference external\" href=\"https://github.com/molmod/stacie\">GitHub</a> and <a class=\"reference external\" href=\"https://pypi.org/project/stacie\">PyPI</a>.</p>", "a[href=\"getting_started/cite.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">How to Cite<a class=\"headerlink\" href=\"#how-to-cite\" title=\"Link to this heading\">\u00b6</a></h1><p>When using STACIE in your research, please cite the STACIE paper in any resulting publication.\nThe reference is provided in several formats below:</p>"}
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
