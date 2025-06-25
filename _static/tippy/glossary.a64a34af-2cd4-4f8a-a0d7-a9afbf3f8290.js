selector_to_html = {"a[href=\"#term-PSD\"]": "<dt id=\"term-PSD\">PSD</dt><dd><p>Power spectral density.\nA measure of the power of a signal as a function of frequency.\nThe Fourier transform of the autocorrelation function.</p></dd>", "a[href=\"#term-LAMMPS\"]": "<dt id=\"term-LAMMPS\">LAMMPS</dt><dd><p>Large-scale atomic/molecular massively parallel simulator.\nA software package for simulating molecular dynamics.\nSee https://www.lammps.org/</p></dd>", "a[href=\"#glossary\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Glossary<a class=\"headerlink\" href=\"#glossary\" title=\"Link to this heading\">\u00b6</a></h1>", "a[href=\"#term-ACF\"]": "<dt id=\"term-ACF\">ACF</dt><dd><p>Autocorrelation function.\nA measure of the correlation of a signal with itself at different time lags.</p></dd>", "a[href=\"#term-NVE\"]": "<dt id=\"term-NVE\">NVE</dt><dd><p>Microcanonical ensemble.\nA statistical ensemble that represents a closed system\nwith fixed energy (E), volume (V), and number of particles (N).</p></dd>", "a[href=\"#term-Uncertainty\"]": "<dt id=\"term-Uncertainty\">Uncertainty</dt><dd><p>An estimate of the standard deviation of a result\nif the analysis would have been repeated many times with independent inputs.\nThis is also known as the standard error.</p></dd>", "a[href=\"#term-NVT\"]": "<dt id=\"term-NVT\">NVT</dt><dd><p>Canonical ensemble.\nA statistical ensemble that represents a system in thermal equilibrium\nwith a heat bath at constant temperature (T), volume (V), and number of particles (N).</p></dd>", "a[href=\"#term-NpT\"]": "<dt id=\"term-NpT\">NpT</dt><dd><p>Isothermal-isobaric ensemble.\nA statistical ensemble that represents a system in thermal equilibrium\nwith a heat bath at constant temperature (T), pressure (p), and number of particles (N).</p></dd>", "a[href=\"#term-MD\"]": "<dt id=\"term-MD\">MD</dt><dd><p>Molecular dynamics.\nA computational method used to simulate the physical movements of atoms and molecules.</p></dd>"}
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
