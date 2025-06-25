selector_to_html = {"a[href=\"#changelog\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Changelog<a class=\"headerlink\" href=\"#changelog\" title=\"Link to this heading\">\u00b6</a></h1><p>All notable changes to this project will be documented in this file.</p><p>The format is based on <a class=\"reference external\" href=\"https://keepachangelog.com/en/1.1.0/\">Keep a Changelog</a>,\nand this project adheres to <a class=\"reference external\" href=\"https://jacobtomlinson.dev/effver/\">Effort-based Versioning</a>.</p>", "a[href=\"#unreleased\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><a class=\"reference external\" href=\"https://github.com/molmod/stacie\">Unreleased</a><a class=\"headerlink\" href=\"#unreleased\" title=\"Link to this heading\">\u00b6</a></h2>"}
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
