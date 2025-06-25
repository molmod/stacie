selector_to_html = {"a[href=\"../theory/index.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Theory<a class=\"headerlink\" href=\"#theory\" title=\"Link to this heading\">\u00b6</a></h1><p>This section focuses solely on the autocorrelation integral itself.\nThe (physical) <a class=\"reference internal\" href=\"../properties/index.html\"><span class=\"std std-doc\">properties</span></a> associated with this integral\nare discussed later.</p><p>Some derivations presented here can also be found in other sources.\nThey are included to enhance accessibility\nand to provide all the necessary details for implementing STACIE.</p>", "a[href=\"../examples/index.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Worked Examples<a class=\"headerlink\" href=\"#worked-examples\" title=\"Link to this heading\">\u00b6</a></h1><p>All the examples are also available as Jupyter notebooks and can be downloaded as one ZIP archive here:</p>", "a[href=\"../preparing_inputs/index.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Preparing Inputs<a class=\"headerlink\" href=\"#preparing-inputs\" title=\"Link to this heading\">\u00b6</a></h1><p>This section explains how to prepare input sequences for STACIE to ensure high-quality results.\nIt consists of three parts:</p>", "a[href=\"#usage-overview\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Usage Overview<a class=\"headerlink\" href=\"#usage-overview\" title=\"Link to this heading\">\u00b6</a></h1><p>This section provides an overview of how to use STACIE.\nMore detailed information can be found in the remaining sections of the documentation.</p><p>The STACIE algorithm provides robust and reliable estimates of autocorrelation integrals\nwithout requiring extensive adjustment of its settings.\nUsers simply provide the relevant inputs to STACIE:\nthe time-correlated sequences in the form of a NumPy array,\na few physical parameters (such as the time step),\nand a model to fit to the spectrum.\nThis can be done in a Jupyter notebook for interactive work or in a Python script.</p>", "a[href=\"../properties/index.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Properties Derived from the Autocorrelation Function<a class=\"headerlink\" href=\"#properties-derived-from-the-autocorrelation-function\" title=\"Link to this heading\">\u00b6</a></h1><p>This section outlines the statistical and physical quantities\nthat can be computed as the integral of an autocorrelation function.\nFor each property, a code skeleton is provided as a starting point for your calculations.\nAll skeletons assume that you can load the relevant input data into NumPy arrays.</p><p>First, we discuss a few properties that may be relevant to multiple scientific disciplines:</p>"}
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
