selector_to_html = {"a[href=\"#development\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Development<a class=\"headerlink\" href=\"#development\" title=\"Link to this heading\">\u00b6</a></h1><p>This section contains some technical details about the development of STACIE.</p>", "a[href=\"changelog.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Changelog<a class=\"headerlink\" href=\"#changelog\" title=\"Link to this heading\">\u00b6</a></h1><p>All notable changes to this project will be documented in this file.</p><p>The format is based on <a class=\"reference external\" href=\"https://keepachangelog.com/en/1.1.0/\">Keep a Changelog</a>,\nand this project adheres to <a class=\"reference external\" href=\"https://jacobtomlinson.dev/effver/\">Effort-based Versioning</a>.</p>", "a[href=\"contributing.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Contributor Guide<a class=\"headerlink\" href=\"#contributor-guide\" title=\"Link to this heading\">\u00b6</a></h1><p>First of all, thank you for considering contributing to STACIE!\nSTACIE is being developed by academics who also have many other responsibilities,\nand you are probably in a similar situation.\nThe purpose of this guide is to make efficient use of everyone\u2019s time.</p><p>STACIE has already been used for production simulations,\nbut we are always open to (suggestions for) improvements that fit within the goals of STACIE.\nNew worked examples that are not too computationally demanding are also highly appreciated!\nEven simple things like correcting typos or fixing minor mistakes are welcome.</p>", "a[href=\"setup.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Development Setup<a class=\"headerlink\" href=\"#development-setup\" title=\"Link to this heading\">\u00b6</a></h1><h2>Repository, Tests and Documentation Build<a class=\"headerlink\" href=\"#repository-tests-and-documentation-build\" title=\"Link to this heading\">\u00b6</a></h2><p>It is assumed that you have previously installed Python, Git, pre-commit and direnv.\nA local installation for testing and development can be installed as follows:</p>", "a[href=\"release.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">How to Make a Release<a class=\"headerlink\" href=\"#how-to-make-a-release\" title=\"Link to this heading\">\u00b6</a></h1><h2>Software packaging and deployment<a class=\"headerlink\" href=\"#software-packaging-and-deployment\" title=\"Link to this heading\">\u00b6</a></h2><p>To make a new release of STACIE on PyPI, take the following steps</p>"}
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
