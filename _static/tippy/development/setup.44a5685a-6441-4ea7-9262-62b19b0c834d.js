selector_to_html = {"a[href=\"#development-setup\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Development Setup<a class=\"headerlink\" href=\"#development-setup\" title=\"Link to this heading\">\u00b6</a></h1><h2>Repository, Tests and Documentation Build<a class=\"headerlink\" href=\"#repository-tests-and-documentation-build\" title=\"Link to this heading\">\u00b6</a></h2><p>It is assumed that you have previously installed Python, Git, pre-commit and direnv.\nA local installation for testing and development can be installed as follows:</p>", "a[href=\"#documentation-live-preview\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Documentation Live Preview<a class=\"headerlink\" href=\"#documentation-live-preview\" title=\"Link to this heading\">\u00b6</a></h2><p>The documentation is created using <a class=\"reference external\" href=\"https://www.sphinx-doc.org/\">Sphinx</a>.</p><p>Edit the documentation Markdown files with a live preview\nby running the following command <em>in the root</em> of the repository:</p>", "a[href=\"#repository-tests-and-documentation-build\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Repository, Tests and Documentation Build<a class=\"headerlink\" href=\"#repository-tests-and-documentation-build\" title=\"Link to this heading\">\u00b6</a></h2><p>It is assumed that you have previously installed Python, Git, pre-commit and direnv.\nA local installation for testing and development can be installed as follows:</p>"}
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
