selector_to_html = {"a[href=\"#license\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">License<a class=\"headerlink\" href=\"#license\" title=\"Link to this heading\">\u00b6</a></h1><p>STACIE is free software: you can redistribute it and/or modify it\nunder the terms of the GNU Lesser General Public License\nas published by the Free Software Foundation,\neither version 3 of the License, or (at your option) any later version.</p><p>STACIE is distributed in the hope that it will be useful,\nbut WITHOUT ANY WARRANTY;\nwithout even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\nSee the GNU Lesser General Public License for more details.</p>"}
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
