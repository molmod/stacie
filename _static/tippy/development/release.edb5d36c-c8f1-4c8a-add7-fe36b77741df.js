selector_to_html = {"a[href=\"#how-to-make-a-release\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">How to Make a Release<a class=\"headerlink\" href=\"#how-to-make-a-release\" title=\"Link to this heading\">\u00b6</a></h1><h2>Software packaging and deployment<a class=\"headerlink\" href=\"#software-packaging-and-deployment\" title=\"Link to this heading\">\u00b6</a></h2><p>To make a new release of STACIE on PyPI, take the following steps</p>", "a[href=\"#software-packaging-and-deployment\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Software packaging and deployment<a class=\"headerlink\" href=\"#software-packaging-and-deployment\" title=\"Link to this heading\">\u00b6</a></h2><p>To make a new release of STACIE on PyPI, take the following steps</p>", "a[href=\"#documentation-build-and-deployment\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Documentation build and deployment<a class=\"headerlink\" href=\"#documentation-build-and-deployment\" title=\"Link to this heading\">\u00b6</a></h2><p>Take the following steps, starting from the root of the repository:</p>"}
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
