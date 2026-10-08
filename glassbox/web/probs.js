/* GlassBox Probability Bars — labels always live, optional sampled-token marker */
/* Made with 🔮 by Vivi */

const ProbBars = (() => {
    const svg = document.getElementById('prob-bars');
    const tempSlider = document.getElementById('temp-slider');
    const toppSlider = document.getElementById('topp-slider');
    const topkSlider = document.getElementById('topk-slider');
    const tempVal = document.getElementById('temp-val');
    const toppVal = document.getElementById('topp-val');
    const topkVal = document.getElementById('topk-val');

    let currentIds = [], currentValues = [], currentStrs = [], currentSampledId = null;
    let pickHandler = null;

    function pickFromTarget(target) {
        const item = target.closest('[data-pick-index]');
        if (!item || !pickHandler) return;
        const raw = currentStrs[Number(item.dataset.pickIndex)];
        if (typeof raw === 'string' && raw.length > 0) pickHandler(raw);
    }

    // sampledId is optional: pass the id of the token that was actually generated
    function setStep(ids, values, strs, sampledId = null) {
        currentIds = ids;
        currentValues = values;
        currentStrs = strs;
        currentSampledId = sampledId;
        rerender();
    }

    function rerender() {
        const temp = parseFloat(tempSlider.value);
        const topP = parseFloat(toppSlider.value);
        const topKSetting = parseInt(topkSlider.value);

        // Labels first, so they update even before the first generation
        tempVal.textContent = temp.toFixed(2);
        toppVal.textContent = topP.toFixed(2);
        topkVal.textContent = topKSetting;

        if (currentValues.length === 0) {
            svg.innerHTML = '<text x="50%" y="50%" text-anchor="middle" fill="#94a3b8" font-family="JetBrains Mono" font-size="11">generate to see distribution</text>';
            return;
        }

        const topK = Math.min(topKSetting, currentValues.length);

        // Softmax with temperature
        const scaled = currentValues.map(v => v / temp);
        const maxV = Math.max(...scaled);
        const exps = scaled.map(v => Math.exp(v - maxV));
        const sumExp = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map(v => v / sumExp);

        // Sort descending
        const items = probs.map((p, i) => ({
            p, id: currentIds[i], str: currentStrs[i], displayStr: currentStrs[i] || `[${currentIds[i]}]`, index: i
        })).sort((a, b) => b.p - a.p);

        // Top-P cutoff (number of tokens kept by nucleus sampling)
        let cumSum = 0;
        let pCutoffIndex = items.length;
        for (let i = 0; i < items.length; i++) {
            cumSum += items[i].p;
            if (cumSum >= topP && pCutoffIndex === items.length) {
                pCutoffIndex = i + 1;
            }
        }

        const top12 = items.slice(0, 12);
        const maxP = top12[0].p;

        // Is the sampled token outside the visible bars? If so, add one extra row.
        let extra = null;
        if (currentSampledId !== null) {
            const rank = items.findIndex(it => it.id === currentSampledId);
            if (rank >= 12) extra = { item: items[rank], rank: rank + 1 };
        }

        const barH = 14;
        const gap = 3;
        const rows = top12.length + (extra ? 1 : 0);
        const w = svg.clientWidth || 320;
        const h = rows * (barH + gap) + 16;
        svg.setAttribute('viewBox', `0 0 ${w} ${h}`);

        const labelW = 105;
        const barAreaW = w - labelW - 50;

        let html = '';
        top12.forEach((item, idx) => {
            const y = idx * (barH + gap) + 8;
            const barW = (item.p / maxP) * barAreaW;
            const isKept = (idx < topK) && (idx < pCutoffIndex);
            const isSampled = currentSampledId !== null && item.id === currentSampledId;

            const opacity = isKept ? (0.4 + 0.6 * (item.p / maxP)) : 0.12;
            const textColor = isKept ? '#f8fafc' : '#748096';
            const barColor = isKept ? '#7aa2ff' : '#4a5568';
            const outline = isSampled ? ' stroke="#fcc419" stroke-width="1.5"' : '';
            const tag = isSampled ? ' ◀ picked' : '';

            const isPickable = typeof item.str === 'string' && item.str.length > 0;
            html += `
                <g${isPickable ? ` data-pick-index="${item.index}" role="button" tabindex="0" aria-label="Force token ${vis(item.displayStr)}"` : ''}>
                <text x="${labelW - 6}" y="${y + 11}" text-anchor="end" fill="${textColor}" font-family="JetBrains Mono" font-size="10">${vis(item.displayStr)}</text>
                <rect x="${labelW}" y="${y}" width="${barW}" height="${barH}" fill="${barColor}" fill-opacity="${opacity}"${outline} rx="2"/>
                <text x="${labelW + barW + 5}" y="${y + 11}" fill="${isSampled ? '#fcc419' : textColor}" font-family="JetBrains Mono" font-size="9">${(item.p * 100).toFixed(1)}%${tag}</text>
                </g>
            `;

            // Dashed line marking where top-k / top-p cut the distribution
            if (idx === Math.min(topK, pCutoffIndex) - 1) {
                const lineY = y + barH + 1;
                html += `<line x1="0" y1="${lineY}" x2="${w}" y2="${lineY}" stroke="#fcc419" stroke-dasharray="3,3" stroke-width="1"/>`;
            }
        });

        // Extra row for a sampled token that fell outside the top 12
        if (extra) {
            const y = top12.length * (barH + gap) + 8;
            const barW = Math.max(2, (extra.item.p / maxP) * barAreaW);
            const isPickable = typeof extra.item.str === 'string' && extra.item.str.length > 0;
            html += `
                <g${isPickable ? ` data-pick-index="${extra.item.index}" role="button" tabindex="0" aria-label="Force token ${vis(extra.item.displayStr)}"` : ''}>
                <text x="${labelW - 6}" y="${y + 11}" text-anchor="end" fill="#fcc419" font-family="JetBrains Mono" font-size="10">${vis(extra.item.displayStr)}</text>
                <rect x="${labelW}" y="${y}" width="${barW}" height="${barH}" fill="#7aa2ff" fill-opacity="0.5" stroke="#fcc419" stroke-width="1.5" rx="2"/>
                <text x="${labelW + barW + 5}" y="${y + 11}" fill="#fcc419" font-family="JetBrains Mono" font-size="9">${(extra.item.p * 100).toFixed(1)}% ◀ picked (rank ${extra.rank})</text>
                </g>
            `;
        }

        svg.innerHTML = html;
    }

    tempSlider.addEventListener('input', rerender);
    toppSlider.addEventListener('input', rerender);
    topkSlider.addEventListener('input', rerender);
    svg.addEventListener('click', (event) => pickFromTarget(event.target));
    svg.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            pickFromTarget(event.target);
        }
    });

    rerender(); // show correct slider labels on page load

    return {
        setStep,
        rerender,
        onPick(callback) { pickHandler = callback; }
    };
})();