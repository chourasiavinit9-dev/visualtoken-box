/* GlassBox Logit Lens Strip — Highlights the FIRST layer that predicts final answer */
/* Made with 🔮 by Vivi */

const Lens = (() => {
    const strip = document.getElementById('lens-strip');

    function render(lensPredictions) {
        if (!lensPredictions || lensPredictions.length === 0) {
            strip.innerHTML = '<span style="color:#748096; font-family:JetBrains Mono; font-size:11px;">no lens data</span>';
            return;
        }

        // Final answer = last layer's top prediction token
        const finalAnswer = lensPredictions[lensPredictions.length - 1].token_str;

        // Highlight the beginning of the stable suffix matching the final prediction.
        let firstStableMatchIndex = lensPredictions.length - 1;
        for (let i = lensPredictions.length - 2; i >= 0; i--) {
            if (lensPredictions[i].token_str !== finalAnswer) break;
            firstStableMatchIndex = i;
        }

        let html = '';
        lensPredictions.forEach((pred, idx) => {
            const isFirstStableMatch = (idx === firstStableMatchIndex);
            const cls = isFirstStableMatch ? 'lens-cell lens-first-match' : 'lens-cell';
            
            const cleanStr = vis(pred.token_str);
            
            const pct = (pred.prob * 100).toFixed(1);
            const badge = isFirstStableMatch ? ' 🎯' : '';
            const blockLabel = `B${pred.layer + 1}`;

            html += `
                <div class="${cls}">
                    <div class="lens-layer" title="after transformer block ${pred.layer + 1}">${blockLabel}${badge}</div>
                    <div class="lens-token" title="${cleanStr}">${cleanStr}</div>
                    <div class="lens-prob">${pct}%</div>
                </div>
            `;
        });
        strip.innerHTML = html;
    }

    return { render };
})();