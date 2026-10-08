/* ═══════════════════════════════════════════════════════════════════════════
   GlassBox Features.js
   Implements 4 features:
   1. Activation Steering ("Brainwash Slider") — tone injection UI
   2. Attention Rollout ("Why Matrix") — cross-layer token influence
   3. Automatic Head Detection — Induction / Prev-token / BOS badges
   4. Tokenization Pictograph — visual token breakdown diagram
   ═══════════════════════════════════════════════════════════════════════════ */
/* Made with 🔮 by Vinit Chaurasia */

/* ─────────────────────────────────────────────────────────────────────────
   1. ACTIVATION STEERING MODULE
   ───────────────────────────────────────────────────────────────────────── */
const Steering = (() => {
    const PRESETS = [
        { label: '🏴‍☠️ Pirate',   concept: 'pirate',   alpha: 3.5 },
        { label: '📜 Poetic',    concept: 'poetic',   alpha: 3.0 },
        { label: '⚡ Angry',     concept: 'angry',    alpha: 3.0 },
        { label: '👔 Formal',    concept: 'formal',   alpha: 2.5 },
        { label: '😂 Funny',     concept: 'funny',    alpha: 3.0 },
        { label: '🧪 Technical', concept: 'technical',alpha: 2.5 },
    ];

    let activeConcept = null;
    let activeAlpha = 0;
    let activeBadge = null;

    function init() {
        const container = document.getElementById('steering-presets');
        if (!container) return;

        activeBadge = document.getElementById('steering-active-badge');

        PRESETS.forEach(p => {
            const btn = document.createElement('button');
            btn.className = 'preset-btn';
            btn.textContent = p.label;
            btn.title = `Steer toward "${p.concept}" (α=${p.alpha})`;
            btn.addEventListener('click', () => {
                const isActive = btn.classList.contains('active');
                container.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
                if (isActive) {
                    activeConcept = null; activeAlpha = 0;
                    document.getElementById('steering-slider').value = 0;
                    _updateVal(0);
                    _updateBadge();
                } else {
                    btn.classList.add('active');
                    activeConcept = p.concept;
                    activeAlpha = p.alpha;
                    document.getElementById('steering-slider').value = p.alpha;
                    _updateVal(p.alpha);
                    _updateBadge();
                }
            });
            container.appendChild(btn);
        });

        const slider = document.getElementById('steering-slider');
        if (slider) {
            slider.addEventListener('input', (e) => {
                activeAlpha = parseFloat(e.target.value);
                _updateVal(activeAlpha);
            });
        }

        const customInput = document.getElementById('steering-custom-input');
        if (customInput) {
            customInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    const val = customInput.value.trim();
                    if (val) {
                        container.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
                        activeConcept = val;
                        _updateBadge();
                        customInput.value = '';
                    }
                }
            });
        }

        const reset = document.getElementById('steering-reset');
        if (reset) {
            reset.addEventListener('click', () => {
                activeConcept = null; activeAlpha = 0;
                if (slider) { slider.value = 0; _updateVal(0); }
                container.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
                _updateBadge();
            });
        }
    }

    function _updateVal(val) {
        const el = document.getElementById('steering-val');
        if (el) el.textContent = (val >= 0 ? '+' : '') + Number(val).toFixed(1);
    }

    function _updateBadge() {
        if (!activeBadge) return;
        if (activeConcept && activeAlpha !== 0) {
            activeBadge.textContent = `✦ ${activeConcept} (α=${activeAlpha.toFixed(1)})`;
            activeBadge.classList.add('visible');
        } else {
            activeBadge.classList.remove('visible');
        }
    }

    /** Returns steering params to inject into WebSocket payload */
    function getParams() {
        if (!activeConcept || activeAlpha === 0) return {};
        return { steering_concept: activeConcept, steering_alpha: activeAlpha };
    }

    return { init, getParams };
})();


/* ─────────────────────────────────────────────────────────────────────────
   2. ATTENTION ROLLOUT MODULE ("Why Matrix")
   ───────────────────────────────────────────────────────────────────────── */
const Rollout = (() => {
    let _rolloutEnabled = false;
    let _rolloutMatrix = null;   // [genTokenIdx][promptTokenIdx] — influence scores
    let _promptTokens = [];
    let _genTokens = [];
    let _activeGenIdx = -1;

    /**
     * Compute attention rollout from raw per-step per-layer per-head attention data.
     * allAttentionData: array[steps] of array[layers] of array[heads][seqLen]
     * Returns: matrix[genStep][promptPos] with influence scores 0..1
     */
    function computeRollout(allAttentionData, numPromptTokens) {
        if (!allAttentionData || allAttentionData.length === 0) return null;

        const numSteps = allAttentionData.length;
        const numLayers = allAttentionData[0] ? allAttentionData[0].length : 0;
        if (numLayers === 0) return null;

        const result = [];

        allAttentionData.forEach((stepData, stepIdx) => {
            if (!stepData) { result.push(null); return; }

            const seqLen = numPromptTokens + stepIdx;
            if (seqLen === 0) { result.push(null); return; }

            // Start with identity
            let A = _eye(seqLen);

            // Multiply through each layer's avg attention
            for (let l = 0; l < numLayers; l++) {
                const layerData = stepData[l];
                if (!layerData || !layerData.length) continue;
                // Average over heads
                const nHeads = layerData.length;
                const len = layerData[0] ? layerData[0].length : 0;
                if (len === 0) continue;

                const avgAttn = new Array(len).fill(0);
                for (let h = 0; h < nHeads; h++) {
                    if (!layerData[h]) continue;
                    for (let i = 0; i < len; i++) {
                        avgAttn[i] += (layerData[h][i] || 0) / nHeads;
                    }
                }

                // Only use the last row (current query token → all key positions)
                const row = avgAttn;
                // Build a seqLen×seqLen matrix with rollout formula: 0.5*A + 0.5*I
                const rolloutLayer = _eye(seqLen);
                // Set last row to 0.5*attn + 0.5*identity
                for (let i = 0; i < Math.min(len, seqLen); i++) {
                    rolloutLayer[seqLen - 1][i] = 0.5 * (row[i] || 0);
                    if (i === seqLen - 1) rolloutLayer[seqLen - 1][i] += 0.5;
                }
                A = _matmul(rolloutLayer, A, seqLen);
            }

            // Extract influence on prompt tokens from the last row
            const influence = A[seqLen - 1].slice(0, numPromptTokens);
            const maxInf = Math.max(...influence, 1e-9);
            result.push(influence.map(v => v / maxInf));
        });

        return result;
    }

    function _eye(n) {
        return Array.from({ length: n }, (_, i) =>
            Array.from({ length: n }, (_, j) => i === j ? 1 : 0)
        );
    }

    function _matmul(A, B, n) {
        const C = _eye(n);
        // Only compute last row for efficiency (we only need result[n-1])
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let k = 0; k < n; k++) sum += A[n - 1][k] * B[k][j];
            C[n - 1][j] = sum;
        }
        return C;
    }

    function setData(allAttentionData, promptTokens, genTokens) {
        _promptTokens = promptTokens || [];
        _genTokens = genTokens || [];
        _rolloutMatrix = computeRollout(allAttentionData, _promptTokens.length);
        _activeGenIdx = -1;
        _renderTokensList();
    }

    function _renderTokensList() {
        const panel = document.getElementById('rollout-gen-tokens');
        if (!panel) return;
        panel.innerHTML = '';

        _genTokens.forEach((tok, idx) => {
            const el = document.createElement('span');
            el.className = 'rollout-token';
            el.textContent = (tok || '').replace(/\s/g, '·');
            el.dataset.idx = idx;
            el.addEventListener('click', () => highlightInfluence(idx));
            panel.appendChild(el);
        });
    }

    function highlightInfluence(genIdx) {
        _activeGenIdx = genIdx;
        const panel = document.getElementById('rollout-gen-tokens');
        if (panel) {
            panel.querySelectorAll('.rollout-token').forEach((el, i) => {
                el.classList.toggle('highlight', i === genIdx);
                el.style.setProperty('--rollout-alpha', i === genIdx ? '0.4' : '');
            });
        }

        // Highlight rollout-panel prompt tokens
        const promptPanel = document.getElementById('rollout-prompt-tokens');
        if (promptPanel && _rolloutMatrix && _rolloutMatrix[genIdx]) {
            const scores = _rolloutMatrix[genIdx];
            promptPanel.innerHTML = '';
            scores.forEach((score, i) => {
                const el = document.createElement('span');
                el.className = 'rollout-token';
                el.textContent = (_promptTokens[i] || '').replace(/\s/g, '·');
                if (score > 0.05) {
                    el.classList.add('highlight');
                    el.style.setProperty('--rollout-alpha', Math.min(score, 0.9).toFixed(2));
                    el.title = `${(score * 100).toFixed(1)}% influence on "${_genTokens[genIdx]}"`;
                }
                promptPanel.appendChild(el);
            });
        }

        // Highlight the chip in the token stream
        document.querySelectorAll('.chip').forEach(c => c.classList.remove('rollout-target'));
        const targetChip = document.querySelector(`.chip[data-step="${genIdx}"]`);
        if (targetChip) targetChip.classList.add('rollout-target');
    }

    function toggle() {
        _rolloutEnabled = !_rolloutEnabled;
        const panel = document.getElementById('rollout-panel');
        const btn = document.getElementById('rollout-btn');
        if (panel) panel.classList.toggle('visible', _rolloutEnabled);
        if (btn) btn.classList.toggle('active', _rolloutEnabled);
    }

    function init() {
        const btn = document.getElementById('rollout-btn');
        if (btn) btn.addEventListener('click', toggle);
    }

    return { init, setData, highlightInfluence };
})();


/* ─────────────────────────────────────────────────────────────────────────
   3. HEAD TYPE DETECTOR (Induction / Prev-token / BOS / Broad)
   ───────────────────────────────────────────────────────────────────────── */
const HeadDetector = (() => {
    const HEAD_TYPES = {
        BOS:       { label: 'BOS',       css: 'head-badge-bos',       emoji: '🔵' },
        PREV:      { label: 'Prev',      css: 'head-badge-prev',      emoji: '🟢' },
        INDUCTION: { label: 'Induction', css: 'head-badge-induction', emoji: '🟡' },
        BROAD:     { label: 'Broad',     css: 'head-badge-broad',     emoji: '⚪' },
    };

    // headTypes[layer][head] → HEAD_TYPES key
    let _headTypes = {};

    /**
     * Run heuristic detection over all collected attention data.
     * allAttentionData: array[steps] of array[layers] of array[heads][seqLen]
     */
    function detect(allAttentionData, numLayers, numHeads) {
        _headTypes = {};
        if (!allAttentionData || allAttentionData.length < 2) return;

        for (let l = 0; l < numLayers; l++) {
            _headTypes[l] = {};
            for (let h = 0; h < numHeads; h++) {
                _headTypes[l][h] = _classify(allAttentionData, l, h);
            }
        }
    }

    function _classify(data, layer, head) {
        let bosScore = 0, prevScore = 0, inductionScore = 0, total = 0;

        data.forEach((stepData, stepIdx) => {
            if (!stepData || !stepData[layer] || !stepData[layer][head]) return;
            const row = stepData[layer][head];
            if (!row || row.length === 0) return;
            total++;

            const sum = row.reduce((a, b) => a + b, 0) || 1;
            const norm = row.map(v => v / sum);

            // BOS: token 0 gets > 50% attention
            if (norm[0] > 0.5) bosScore++;

            // Previous-token: token at position [seqLen-2] (1-back) gets > 40%
            if (norm.length >= 2 && norm[norm.length - 2] > 0.4) prevScore++;

            // Induction: attention pattern is sharply peaked on a previous occurrence
            // Simple proxy: high attention on some non-BOS, non-current token with low entropy
            const entropy = -norm.reduce((e, p) => p > 0 ? e + p * Math.log2(p + 1e-9) : e, 0);
            if (entropy < 1.5 && norm[0] < 0.4) inductionScore++;
        });

        if (total === 0) return 'BROAD';
        const bos = bosScore / total;
        const prev = prevScore / total;
        const ind = inductionScore / total;

        if (bos > 0.6) return 'BOS';
        if (prev > 0.5) return 'PREV';
        if (ind > 0.4) return 'INDUCTION';
        return 'BROAD';
    }

    function getType(layer, head) {
        return _headTypes[layer] && _headTypes[layer][head]
            ? HEAD_TYPES[_headTypes[layer][head]]
            : HEAD_TYPES.BROAD;
    }

    function getBadgeHTML(layer, head) {
        const t = getType(layer, head);
        return `<span class="head-badge ${t.css}" title="${t.label} head">${t.label}</span>`;
    }

    return { detect, getType, getBadgeHTML };
})();


/* ─────────────────────────────────────────────────────────────────────────
   4. TOKENIZATION PICTOGRAPH MODULE
   ───────────────────────────────────────────────────────────────────────── */
const TokenPictograph = (() => {
    const COLORS = 8; // number of tc-N classes
    let debounceTimer = null;
    let cachedPromptTokens = [];
    let cachedPromptTokenIds = [];
    let cachedPromptText = '';

    function init(inputEl) {
        if (!inputEl) inputEl = document.getElementById('prompt');
        if (inputEl) {
            // Live update as user types in prompt box
            inputEl.addEventListener('input', () => {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(() => {
                    const txt = inputEl.value.trim();
                    if (txt) fetchAndRender(txt);
                }, 180);
            });
            // Fetch & render initial prompt immediately on page load
            const initial = inputEl.value.trim() || 'What is the capital of India?';
            fetchAndRender(initial);
        }
    }

    async function fetchAndRender(text) {
        try {
            const res = await fetch(`/api/tokenize?text=${encodeURIComponent(text)}`);
            if (res.ok) {
                const data = await res.json();
                cachedPromptText = data.text || text;
                cachedPromptTokens = data.token_strs || [];
                cachedPromptTokenIds = data.token_ids || [];
                render(cachedPromptText, cachedPromptTokens, cachedPromptTokenIds, []);
            }
        } catch (e) {
            console.warn('Tokenize fetch error:', e);
        }
    }

    function render(promptText, promptTokens, promptTokenIds = [], genTokens = []) {
        const container = document.getElementById('pictograph-stage');
        if (!container) return;
        container.innerHTML = '';

        if (!promptTokens || promptTokens.length === 0) {
            container.innerHTML = '<div style="font-size:0.72rem;color:var(--text-muted);padding:4px 0;">Type a prompt above to see how it splits into tokens ↓</div>';
            return;
        }

        cachedPromptText = promptText || cachedPromptText;
        cachedPromptTokens = promptTokens;
        if (promptTokenIds && promptTokenIds.length > 0) {
            cachedPromptTokenIds = promptTokenIds;
        }

        // Row 1: Words breakdown
        const words = (promptText || '').trim().split(/\s+/).filter(Boolean);
        const wordItems = words.map(w => ({ type: 'word', text: w }));
        _addRow(container, 'words', wordItems.length > 0 ? wordItems : [{ type: 'word', text: promptText || '—' }]);

        // Row 2: Prompt BPE tokens with Token IDs
        const pTokenEls = promptTokens.map((tok, i) => ({
            type: 'token',
            text: tok,
            id: (cachedPromptTokenIds && cachedPromptTokenIds[i] !== undefined) ? `#${cachedPromptTokenIds[i]}` : `t${i}`,
            colorIdx: i % COLORS
        }));
        _addRow(container, 'subwords', pTokenEls, true);

        // Row 3: Generated tokens (if present)
        if (genTokens && genTokens.length > 0) {
            const gTokenEls = genTokens.map((tok, i) => ({
                type: 'token',
                text: tok,
                id: `+${i + 1}`,
                colorIdx: (promptTokens.length + i) % COLORS,
                isGen: true
            }));
            _addRow(container, 'generated', gTokenEls, true);
        }

        // Row 4: Summary stats
        const total = promptTokens.length + (genTokens ? genTokens.length : 0);
        const numWords = Math.max(1, words.length);
        const ratio = (promptTokens.length / numWords).toFixed(2);
        const statsRow = document.createElement('div');
        statsRow.className = 'pictograph-row';
        statsRow.style.justifyContent = 'space-between';
        statsRow.style.paddingTop = '4px';
        statsRow.style.borderTop = '1px solid rgba(255,255,255,0.06)';
        statsRow.innerHTML = `
            <span style="font-size:0.65rem;color:var(--text-muted);font-family:var(--font-mono);">
                ratio: <b style="color:#7aa2ff;">${ratio}</b> tok/word
            </span>
            <span style="font-size:0.68rem;color:var(--text-muted);font-family:var(--font-mono);">
                ${promptTokens.length} prompt + ${genTokens ? genTokens.length : 0} gen
                = <b style="color:var(--text);">${total} tokens</b>
            </span>`;
        container.appendChild(statsRow);
    }

    function _addRow(container, label, items, showIds = false) {
        const row = document.createElement('div');
        row.className = 'pictograph-row';

        const lbl = document.createElement('span');
        lbl.className = 'pictograph-row-label';
        lbl.textContent = label;
        row.appendChild(lbl);

        const arrow = document.createElement('span');
        arrow.className = 'pictograph-arrow';
        arrow.textContent = '→';
        row.appendChild(arrow);

        const itemsWrap = document.createElement('div');
        itemsWrap.style.display = 'flex';
        itemsWrap.style.flexWrap = 'wrap';
        itemsWrap.style.gap = '4px';
        itemsWrap.style.alignItems = 'center';
        itemsWrap.style.flex = '1';

        items.forEach(item => {
            if (item.type === 'word') {
                const el = document.createElement('span');
                el.className = 'pic-word';
                el.textContent = item.text;
                itemsWrap.appendChild(el);
            } else {
                const wrap = document.createElement('span');
                wrap.className = 'pic-token';
                wrap.title = `Token ID: ${item.id} | String: "${item.text}"`;

                const txt = document.createElement('span');
                txt.className = `pic-token-text tc-${item.colorIdx}`;
                // Visualise leading/trailing space with ·
                let display = (item.text || '').replace(/ /g, '·');
                if (!display) display = '␣';
                txt.textContent = display;

                if (item.isGen) {
                    txt.style.opacity = '0.85';
                    txt.style.borderStyle = 'dashed';
                }

                const idEl = document.createElement('span');
                idEl.className = 'pic-token-id';
                idEl.textContent = item.id;

                wrap.append(txt);
                if (showIds) wrap.append(idEl);
                itemsWrap.appendChild(wrap);
            }
        });

        row.appendChild(itemsWrap);
        container.appendChild(row);
    }

    return { init, render, fetchAndRender };
})();
