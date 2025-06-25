selector_to_html = {"a[href=\"#contribution-workflow\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Contribution Workflow<a class=\"headerlink\" href=\"#contribution-workflow\" title=\"Link to this heading\">\u00b6</a></h2><p>Contributing to STACIE always involves the following steps:</p>", "a[href=\"#first-time-contributors\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">First-Time Contributors<a class=\"headerlink\" href=\"#first-time-contributors\" title=\"Link to this heading\">\u00b6</a></h2><p>If you have never contributed to an open source project before,\nyou may find the following online references helpful:</p>", "a[href=\"#how-to-report-a-bug\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">How to Report a Bug<a class=\"headerlink\" href=\"#how-to-report-a-bug\" title=\"Link to this heading\">\u00b6</a></h2><p>Create a new issue (or find an existing one) and include the following information:</p>", "a[href=\"../getting_started/licenses.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">License<a class=\"headerlink\" href=\"#license\" title=\"Link to this heading\">\u00b6</a></h1><p>STACIE is free software: you can redistribute it and/or modify it\nunder the terms of the GNU Lesser General Public License\nas published by the Free Software Foundation,\neither version 3 of the License, or (at your option) any later version.</p><p>STACIE is distributed in the hope that it will be useful,\nbut WITHOUT ANY WARRANTY;\nwithout even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\nSee the GNU Lesser General Public License for more details.</p>", "a[href=\"#contributor-guide\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Contributor Guide<a class=\"headerlink\" href=\"#contributor-guide\" title=\"Link to this heading\">\u00b6</a></h1><p>First of all, thank you for considering contributing to STACIE!\nSTACIE is being developed by academics who also have many other responsibilities,\nand you are probably in a similar situation.\nThe purpose of this guide is to make efficient use of everyone\u2019s time.</p><p>STACIE has already been used for production simulations,\nbut we are always open to (suggestions for) improvements that fit within the goals of STACIE.\nNew worked examples that are not too computationally demanding are also highly appreciated!\nEven simple things like correcting typos or fixing minor mistakes are welcome.</p>", "a[href=\"#ground-rules\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Ground Rules<a class=\"headerlink\" href=\"#ground-rules\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"../code_of_conduct.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Code of Conduct<a class=\"headerlink\" href=\"#code-of-conduct\" title=\"Link to this heading\">\u00b6</a></h1><h2>Our Pledge<a class=\"headerlink\" href=\"#our-pledge\" title=\"Link to this heading\">\u00b6</a></h2><p>We as members, contributors, and leaders pledge to make participation in our\ncommunity a harassment-free experience for everyone, regardless of age, body\nsize, visible or invisible disability, ethnicity, sex characteristics, gender\nidentity and expression, level of experience, education, socio-economic status,\nnationality, personal appearance, race, caste, color, religion, or sexual\nidentity and orientation.</p><p>We pledge to act and interact in ways that contribute to an open, welcoming,\ndiverse, inclusive, and healthy community.</p>"}
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
