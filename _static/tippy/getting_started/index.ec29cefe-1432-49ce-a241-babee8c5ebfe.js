selector_to_html = {"a[href=\"installation.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Installation<a class=\"headerlink\" href=\"#installation\" title=\"Link to this heading\">\u00b6</a></h1><p>Before you begin, ensure that you have the following installed:</p>", "a[href=\"#getting-started\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Getting Started<a class=\"headerlink\" href=\"#getting-started\" title=\"Link to this heading\">\u00b6</a></h1><p>STACIE is a Python software library that can be used interactively in a Jupyter Notebook\nor embedded non-interactively in larger computational workflows.</p><p>To get started:</p>", "a[href=\"../theory/index.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Theory<a class=\"headerlink\" href=\"#theory\" title=\"Link to this heading\">\u00b6</a></h1><p>This section focuses solely on the autocorrelation integral itself.\nThe (physical) <a class=\"reference internal\" href=\"../properties/index.html\"><span class=\"std std-doc\">properties</span></a> associated with this integral\nare discussed later.</p><p>Some derivations presented here can also be found in other sources.\nThey are included to enhance accessibility\nand to provide all the necessary details for implementing STACIE.</p>", "a[href=\"../examples/index.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Worked Examples<a class=\"headerlink\" href=\"#worked-examples\" title=\"Link to this heading\">\u00b6</a></h1><p>All the examples are also available as Jupyter notebooks and can be downloaded as one ZIP archive here:</p>", "a[href=\"usage.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Usage Overview<a class=\"headerlink\" href=\"#usage-overview\" title=\"Link to this heading\">\u00b6</a></h1><p>This section provides an overview of how to use STACIE.\nMore detailed information can be found in the remaining sections of the documentation.</p><p>The STACIE algorithm provides robust and reliable estimates of autocorrelation integrals\nwithout requiring extensive adjustment of its settings.\nUsers simply provide the relevant inputs to STACIE:\nthe time-correlated sequences in the form of a NumPy array,\na few physical parameters (such as the time step),\nand a model to fit to the spectrum.\nThis can be done in a Jupyter notebook for interactive work or in a Python script.</p>", "a[href=\"cite.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">How to Cite<a class=\"headerlink\" href=\"#how-to-cite\" title=\"Link to this heading\">\u00b6</a></h1><p>When using STACIE in your research, please cite the STACIE paper in any resulting publication.\nThe reference is provided in several formats below:</p>", "a[href=\"licenses.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">License<a class=\"headerlink\" href=\"#license\" title=\"Link to this heading\">\u00b6</a></h1><p>STACIE is free software: you can redistribute it and/or modify it\nunder the terms of the GNU Lesser General Public License\nas published by the Free Software Foundation,\neither version 3 of the License, or (at your option) any later version.</p><p>STACIE is distributed in the hope that it will be useful,\nbut WITHOUT ANY WARRANTY;\nwithout even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\nSee the GNU Lesser General Public License for more details.</p>"}
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
