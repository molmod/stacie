selector_to_html = {"a[href=\"release.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">How to Make a Release<a class=\"headerlink\" href=\"#how-to-make-a-release\" title=\"Link to this heading\">\u00b6</a></h1><h2>Software packaging and deployment<a class=\"headerlink\" href=\"#software-packaging-and-deployment\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"#development\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Development<a class=\"headerlink\" href=\"#development\" title=\"Link to this heading\">\u00b6</a></h1><p>This section contains some technical details about the development of STACIE.</p>", "a[href=\"changelog.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Changelog<a class=\"headerlink\" href=\"#changelog\" title=\"Link to this heading\">\u00b6</a></h1><p>All notable changes to this project will be documented in this file.</p><p>The format is based on <a class=\"reference external\" href=\"https://keepachangelog.com/en/1.1.0/\">Keep a Changelog</a>,\nand this project adheres to <a class=\"reference external\" href=\"https://jacobtomlinson.dev/effver/\">Effort-based Versioning</a>.</p>"}
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
