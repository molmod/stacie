selector_to_html = {"a[href=\"../theory/model.html#section-pade-target\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">2. Pade Model<a class=\"headerlink\" href=\"#pade-model\" title=\"Link to this heading\">\u00b6</a></h2><p>The <a class=\"reference internal\" href=\"../apidocs/stacie.model.html#stacie.model.PadeModel\" title=\"stacie.model.PadeModel\"><span class=\"xref myst py py-class\">PadeModel</span></a> is defined as:</p>", "a[href=\"#reducing-storage-requirements-with-block-averages\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Reducing Storage Requirements with Block Averages<a class=\"headerlink\" href=\"#reducing-storage-requirements-with-block-averages\" title=\"Link to this heading\">\u00b6</a></h1><p>When computer simulations generate time-dependent data,\nthey often use a discretization of the time axis with a resolution (much) higher\nthan needed for computing the autocorrelation integral with STACIE.\nStoring (and processing) all these data may require excessive resources.\nTo reduce the amount of data, we recommend taking block averages.\nThese block averages form a new time series with a time step equal to the block size\nmultiplied by the original time step.\nThey reduce storage requirements by a factor equal to the block size.\nIf the program generating the sequences does not support block averages,\nyou can use <a class=\"reference internal\" href=\"../apidocs/stacie.utils.html#stacie.utils.block_average\" title=\"stacie.utils.block_average\"><code class=\"xref py py-func docutils literal notranslate\"><span class=\"pre\">stacie.utils.block_average()</span></code></a>.</p><p>If the blocks are sufficiently small compared to the decay rate of the autocorrelation function,\nSTACIE will produce virtually the same results.\nThe effect of block averages can be understood by inserting them into the discrete power spectrum,\nusing STACIE\u2019s normalization convention to obtain the proper zero-frequency limit.\nLet <span class=\"math notranslate nohighlight\">\\(\\hat{a}_\\ell\\)</span> be the <span class=\"math notranslate nohighlight\">\\(\\ell\\)</span>\u2019th block average of <span class=\"math notranslate nohighlight\">\\(L\\)</span> blocks with block size <span class=\"math notranslate nohighlight\">\\(B\\)</span>.\nWe can start from the power spectrum of the original sequence, <span class=\"math notranslate nohighlight\">\\(\\hat{x}_n\\)</span>,\nand then introduce approximations to rewrite it in terms of the block averages:</p>", "a[href=\"../examples/surface_diffusion.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Diffusion on a Surface with Newtonian Dynamics<a class=\"headerlink\" href=\"#diffusion-on-a-surface-with-newtonian-dynamics\" title=\"Link to this heading\">\u00b6</a></h1><p>This example shows how to compute the diffusion coefficient\nof a particle adsorbed on a crystal surface.\nFor simplicity, the motion of the adsorbed particle is described\nby Newton\u2019s equations (without thermostat), i.e. in the <a class=\"reference internal\" href=\"../glossary.html#term-NVE\"><span class=\"xref std std-term\">NVE</span></a> ensemble,\nand the particle can only move in two dimensions.</p><p>This is a completely self-contained example that generates the input sequences\n(with numerical integration) and then analyzes them with STACIE.\nUnless otherwise noted, atomic units are used.</p>"}
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
