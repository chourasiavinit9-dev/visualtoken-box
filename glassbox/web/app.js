/* GlassBox App Controller — Bidirectional token chips without quotes */
/* Made with  by Vinit Chaurasia */

(() => {
    const generateBtn = document.getElementById('generate-btn');
    const promptInput = document.getElementById('prompt');
    const maxTokensInput = document.getElementById('max-tokens');
    const useCacheInput = document.getElementById('use-cache');
    const tokenStream = document.getElementById('token-stream');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const metricSpeed = document.getElementById('metric-speed');
    const metricCount = document.getElementById('metric-count');
    const metricCache = document.getElementById('metric-cache');
    const attnLayerSel = document.getElementById('attn-layer');
    const attnHeadSel = document.getElementById('attn-head');
    const attnAvgHeads = document.getElementById('attn-avg-heads');
    const attnRowNorm = document.getElementById('attn-row-norm');
    const attnHideSink = document.getElementById('attn-hide-sink');
    const stepSlider = document.getElementById('step-slider');
    const stepIndicator = document.getElementById('step-indicator');
    const entropyBadge = document.getElementById('entropy-badge');
    const branchBanner = document.getElementById('branch-banner');
    const branchLabel = document.getElementById('branch-label');
    const backToOriginalBtn = document.getElementById('back-to-original');
    const headGallery = document.getElementById('head-gallery');
    const headGalleryGrid = document.getElementById('head-gallery-grid');
    const headGalleryLayer = document.getElementById('head-gallery-layer');

    let steps = [];
    let config = null;
    let promptTokens = [];
    let genTokens = [];
    let allAttentionData = [];
    let ws = null;
    let selectedStepIdx = -1;
    let originalPrompt = '';
    let originalMaxTokens = maxTokensInput.value;
    let runPrompt = '';

    function setStatus(state, text) {
        statusDot.className = `dot dot-${state}`;
        statusText.textContent = text;
    }

    function renderTokenStream(activeIdx = -1) {
        let html = '';
        promptTokens.forEach(t => {
            html += `<span class="chip chip-prompt">${vis(t || '')}</span>`;
        });
        steps.forEach((s, idx) => {
            const isActive = idx === activeIdx;
            const cls = isActive ? 'chip chip-gen chip-active' : 'chip chip-gen';
            const surprise = surpriseForStep(s);
            html += `<span class="${cls} ${surprise.className}" data-step="${idx}" title="${surprise.title}">${vis(s.token_str || '')}</span>`;
        });
        tokenStream.innerHTML = html;
        tokenStream.scrollTop = tokenStream.scrollHeight;
    }

    function surpriseForStep(step) {
        const ids = step.top_logit_ids || [];
        const values = step.top_logit_values || [];
        const tokenIndex = ids.findIndex(id => Number(id) === Number(step.token_id));
        if (tokenIndex < 0 || !values.length) {
            return { className: 'chip-surprise-hot', title: 'Probability: not in top 200; highest surprise' };
        }
        const maxLogit = Math.max(...values);
        const expValues = values.map(value => Math.exp(value - maxLogit));
        const probability = expValues[tokenIndex] / expValues.reduce((sum, value) => sum + value, 0);
        const surpriseBits = -Math.log2(probability);
        const className = surpriseBits < 1
            ? 'chip-surprise-normal'
            : surpriseBits <= 3 ? 'chip-surprise-amber' : 'chip-surprise-hot';
        return {
            className,
            title: `Probability: ${(probability * 100).toFixed(3)}% · surprise: ${surpriseBits.toFixed(2)} bits`
        };
    }

    function populateSelectors() {
        attnLayerSel.innerHTML = ''; attnHeadSel.innerHTML = '';
        for (let i = 0; i < config.num_layers; i++) {
            const o = document.createElement('option');
            o.value = i; o.textContent = `L${i}`;
            attnLayerSel.appendChild(o);
        }
        for (let i = 0; i < config.num_heads; i++) {
            const o = document.createElement('option');
            o.value = i; o.textContent = `H${i}`;
            attnHeadSel.appendChild(o);
        }
        attnLayerSel.value = Math.floor(config.num_layers / 2);
    }

    function updateAttention() {
        if (config && allAttentionData.length > 0) {
            Heatmap.render(
                allAttentionData,
                promptTokens,
                genTokens,
                parseInt(attnLayerSel.value),
                parseInt(attnHeadSel.value),
                attnAvgHeads.checked,
                attnRowNorm.checked,
                attnHideSink.checked
            );
        }
        updateAttentionHighlights();
        renderHeadGallery();
    }

    function updateAttentionHighlights() {
        const chips = Array.from(tokenStream.querySelectorAll('.chip'));
        chips.forEach(chip => {
            chip.classList.remove('chip-attention');
            chip.style.removeProperty('background-color');
        });
        if (selectedStepIdx < 0 || !steps[selectedStepIdx]) return;

        const step = steps[selectedStepIdx];
        const layerRows = step.attention_new_rows && step.attention_new_rows[parseInt(attnLayerSel.value)];
        if (!Array.isArray(layerRows) || layerRows.length === 0) return;

        let row;
        if (attnAvgHeads.checked) {
            const validRows = layerRows.filter(Array.isArray);
            if (!validRows.length) return;
            const rowLength = Math.min(...validRows.map(values => values.length));
            row = Array.from({ length: rowLength }, (_, index) =>
                validRows.reduce((sum, values) => sum + values[index], 0) / validRows.length
            );
        } else {
            row = layerRows[parseInt(attnHeadSel.value)];
        }
        if (!Array.isArray(row)) return;

        const chipsBeforeStep = chips.filter((chip) =>
            chip.classList.contains('chip-prompt') || Number(chip.dataset.step) < selectedStepIdx
        );
        if (row.length !== chipsBeforeStep.length) {
            console.warn(`Attention row length ${row.length} does not match ${chipsBeforeStep.length} chips before step ${selectedStepIdx + 1}`);
            return;
        }

        const weights = [...row];
        if (weights.length) weights[0] = 0;
        const maxWeight = Math.max(0, ...weights);
        if (maxWeight <= 0) return;
        chipsBeforeStep.forEach((chip, index) => {
            const alpha = Math.min(0.55, 0.55 * weights[index] / maxWeight);
            if (alpha <= 0) return;
            chip.classList.add('chip-attention');
            chip.style.backgroundColor = `rgba(252, 196, 25, ${alpha})`;
        });
    }

    function renderHeadGallery() {
        if (!headGallery.open || !config || !allAttentionData.length) return;
        const layerIdx = parseInt(attnLayerSel.value);
        headGalleryLayer.textContent = `L${layerIdx}`;
        headGalleryGrid.innerHTML = '';
        for (let headIdx = 0; headIdx < config.num_heads; headIdx++) {
            const item = document.createElement('button');
            item.type = 'button';
            item.className = 'head-gallery-item';
            item.dataset.head = headIdx;
            item.setAttribute('aria-label', `Select head ${headIdx}`);

            const label = document.createElement('span');
            label.textContent = `H${headIdx}`;
            const canvas = document.createElement('canvas');
            canvas.width = 168;
            canvas.height = 72;
            canvas.setAttribute('aria-hidden', 'true');
            const matrix = allAttentionData.map(stepData =>
                stepData && stepData[layerIdx] && stepData[layerIdx][headIdx]
                    ? stepData[layerIdx][headIdx]
                    : []
            );
            Heatmap.drawMini(canvas, matrix, attnRowNorm.checked, attnHideSink.checked);
            item.append(label, canvas);
            item.addEventListener('click', () => {
                attnHeadSel.value = String(headIdx);
                attnAvgHeads.checked = false;
                attnHeadSel.dispatchEvent(new Event('change', { bubbles: true }));
                headGallery.close();
            });
            headGalleryGrid.appendChild(item);
        }
    }

    function selectStep(idx) {
        if (idx < 0 || idx >= steps.length) return;
        selectedStepIdx = idx;
        const step = steps[idx];
        stepIndicator.textContent = `${idx + 1} / ${steps.length}`;

        entropyBadge.textContent = `model uncertainty: ${step.entropy_bits.toFixed(2)} bits`;

        ProbBars.setStep(step.top_logit_ids, step.top_logit_values, step.top_logit_strs, step.token_id);
        Lens.render(step.lens_predictions);
        renderTokenStream(idx);
        updateAttention();
    }

    function selectHighestEntropyStep() {
        if (steps.length === 0) return;
        let maxH = -1;
        let maxIdx = 0;
        steps.forEach((s, idx) => {
            const h = s.entropy_bits;
            if (h > maxH) {
                maxH = h;
                maxIdx = idx;
            }
        });
        stepSlider.value = maxIdx;
        selectStep(maxIdx);
    }

    function startGeneration(prompt, { branch = false, resetOriginal = true } = {}) {
        if (resetOriginal) {
            originalPrompt = prompt;
            originalMaxTokens = maxTokensInput.value;
        }
        runPrompt = prompt;
        promptInput.value = prompt;
        branchBanner.hidden = !branch;
        if (ws) ws.close();
        steps = []; genTokens = []; allAttentionData = [];
        selectedStepIdx = -1;
        tokenStream.innerHTML = '';

        setStatus('active', 'connecting…');
        generateBtn.disabled = true;

        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const socket = new WebSocket(`${protocol}//${location.host}/ws/generate`);
        ws = socket;

        socket.onopen = () => {
            setStatus('active', 'generating…');
            socket.send(JSON.stringify({
                prompt: runPrompt,
                max_new_tokens: parseInt(maxTokensInput.value),
                temperature: parseFloat(document.getElementById('temp-slider').value),
                top_k: parseInt(document.getElementById('topk-slider').value),
                top_p: parseFloat(document.getElementById('topp-slider').value),
                use_cache: useCacheInput.checked,
                trace: true,
            }));
        };

        socket.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            if (msg.type === 'init') {
                config = msg.config;
                promptTokens = msg.prompt_token_strs;
                populateSelectors();
                renderTokenStream();
            } else if (msg.type === 'step') {
                steps.push(msg);
                genTokens.push(msg.token_str);
                allAttentionData.push(msg.attention_new_rows);
                stepSlider.max = steps.length - 1;
                stepSlider.value = steps.length - 1;

                metricSpeed.textContent = (steps.reduce((a, s) => a + s.tokens_per_sec, 0) / steps.length).toFixed(1);
                metricCount.textContent = steps.length;
                metricCache.textContent = msg.cache_active ? 'on' : 'off';

                renderTokenStream();
                selectStep(steps.length - 1);
            } else if (msg.type === 'done') {
                setStatus('done', `done · ${msg.total_steps} tokens`);
                generateBtn.disabled = false;
                selectHighestEntropyStep();
            }
        };
        socket.onerror = () => {
            if (ws !== socket) return;
            setStatus('error', 'connection error');
            generateBtn.disabled = false;
        };
    }

    generateBtn.addEventListener('click', () => startGeneration(promptInput.value));

    backToOriginalBtn.addEventListener('click', () => {
        if (!originalPrompt) return;
        maxTokensInput.value = originalMaxTokens;
        startGeneration(originalPrompt, { resetOriginal: false });
    });

    ProbBars.onPick((pickedToken) => {
        if (selectedStepIdx < 0 || !steps[selectedStepIdx]) return;
        const step = steps[selectedStepIdx];
        const branchPrompt = runPrompt + steps
            .slice(0, selectedStepIdx)
            .map(previousStep => previousStep.token_str || '')
            .join('') + pickedToken;
        branchLabel.textContent = `branched at step ${selectedStepIdx + 1}: ${step.token_str.trim() || '∅'} → ${pickedToken.trim() || '∅'}`;
        maxTokensInput.value = '15';
        startGeneration(branchPrompt, { branch: true, resetOriginal: false });
    });

    document.getElementById('head-gallery-open').addEventListener('click', () => {
        headGallery.showModal();
        renderHeadGallery();
    });
    document.getElementById('head-gallery-close').addEventListener('click', () => headGallery.close());
    headGallery.addEventListener('click', (event) => {
        if (event.target === headGallery) headGallery.close();
    });

    attnLayerSel.addEventListener('change', updateAttention);
    attnHeadSel.addEventListener('change', updateAttention);
    attnAvgHeads.addEventListener('change', updateAttention);
    attnRowNorm.addEventListener('change', updateAttention);
    attnHideSink.addEventListener('change', updateAttention);

    stepSlider.addEventListener('input', (e) => selectStep(parseInt(e.target.value)));

    tokenStream.addEventListener('click', (e) => {
        const stepAttr = e.target.getAttribute('data-step');
        if (stepAttr !== null) {
            const idx = parseInt(stepAttr);
            stepSlider.value = idx;
            selectStep(idx);
        }
    });

    setStatus('idle', 'idle');
})();