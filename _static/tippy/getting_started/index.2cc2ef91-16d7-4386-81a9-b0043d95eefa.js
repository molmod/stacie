selector_to_html = {"a[href=\"../examples/index.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Worked Examples<a class=\"headerlink\" href=\"#worked-examples\" title=\"Link to this heading\">\u00b6</a></h1><p>All the examples are also available as Jupyter notebooks and can be downloaded as one ZIP archive here:</p>", "a[href=\"#getting-started\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Getting Started<a class=\"headerlink\" href=\"#getting-started\" title=\"Link to this heading\">\u00b6</a></h1><p>STACIE is a Python software library that can be used interactively in a Jupyter Notebook\nor embedded non-interactively in larger computational workflows.</p><p>To get started:</p>", "a[href=\"cite.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">How to Cite<a class=\"headerlink\" href=\"#how-to-cite\" title=\"Link to this heading\">\u00b6</a></h1><p>When using STACIE in your research, please cite the STACIE paper in any resulting publication.\nThe reference is provided in several formats below:</p>"}
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
