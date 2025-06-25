selector_to_html = {"a[href=\"#stacie.synthetic.generate\"]": "<dt class=\"sig sig-object py\" id=\"stacie.synthetic.generate\">\n<span class=\"sig-name descname\"><span class=\"pre\">generate</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">psd</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">timestep</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">nseq</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">nstep</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">rng</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../_modules/stacie/synthetic.html#generate\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd><p>Generate sequences with a given power spectral density.</p></dd>", "a[href=\"#module-stacie.synthetic\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">stacie.synthetic module<a class=\"headerlink\" href=\"#module-stacie.synthetic\" title=\"Link to this heading\">\u00b6</a></h1><p>Generate synthetic time-correlated data for algorithmic testing and validation.</p>"}
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
