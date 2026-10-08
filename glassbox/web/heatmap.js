/* GlassBox Attention Heatmap — Full Width Canvas + Row Norm without Sink Bias */
/* Made with by Vinit Chaurasia */

const Heatmap = (() => {
    const canvas = document.getElementById('attn-canvas');
    const ctx = canvas.getContext('2d');
    const info = document.getElementById('attn-info');

    let stepsData = [];
    let promptTokens = [];
    let genTokens = [];

    function viridis(t) {
        const stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
        t = Math.max(0, Math.min(1, t));
        const idx = t * (stops.length - 1);
        const i0 = Math.floor(idx), i1 = Math.min(i0 + 1, stops.length - 1);
        const fraction = idx - i0;
        const start = stops[i0], end = stops[i1];
        return `rgb(${Math.round(start[0] + (end[0] - start[0]) * fraction)},${Math.round(start[1] + (end[1] - start[1]) * fraction)},${Math.round(start[2] + (end[2] - start[2]) * fraction)})`;
    }

    function drawMini(miniCanvas, sourceMatrix, rowNorm = true, hideSink = true) {
        const miniCtx = miniCanvas.getContext('2d');
        const width = miniCanvas.width;
        const height = miniCanvas.height;
        const totalCols = Math.max(0, ...sourceMatrix.map(row => row.length));
        miniCtx.fillStyle = '#090a0f';
        miniCtx.fillRect(0, 0, width, height);
        if (!sourceMatrix.length || !totalCols) return;

        const cellW = width / totalCols;
        const cellH = height / sourceMatrix.length;
        sourceMatrix.forEach((sourceRow, rowIdx) => {
            const row = [...sourceRow];
            if (hideSink && row.length) row[0] = 0;
            const scaleValues = hideSink && row.length > 1 ? row.slice(1) : row;
            const maxValue = Math.max(0, ...scaleValues);
            row.forEach((value, colIdx) => {
                const normalized = rowNorm && !(hideSink && colIdx === 0) && maxValue > 0
                    ? value / maxValue
                    : value;
                miniCtx.fillStyle = hideSink && colIdx === 0 ? '#292d35' : viridis(Math.sqrt(normalized));
                miniCtx.fillRect(colIdx * cellW, rowIdx * cellH, cellW + 0.5, cellH + 0.5);
            });
        });
    }

    function render(data, pTokens, gTokens, layerIdx, headIdx, avgHeads, rowNorm, hideSink) {
        stepsData = data; promptTokens = pTokens; genTokens = gTokens;
        if (!stepsData || stepsData.length === 0) return;

        // Build raw matrix
        let matrix = [];
        for (const stepAttn of stepsData) {
            if (!stepAttn || !stepAttn[layerIdx]) continue;
            const layer = stepAttn[layerIdx];
            let row;
            if (avgHeads) {
                const nHeads = layer.length, len = layer[0].length;
                row = new Array(len).fill(0);
                for (let h = 0; h < nHeads; h++) {
                    for (let i = 0; i < len; i++) row[i] += layer[h][i] / nHeads;
                }
            } else {
                row = layer[headIdx] || [];
            }
            matrix.push([...row]);
        }

        if (matrix.length === 0) return;

        // 1. Hide Sink Option (Zero out Column 0)
        if (hideSink) {
            matrix = matrix.map(row => {
                const r = [...row];
                if (r.length > 0) r[0] = 0;
                return r;
            });
        }

        // 2. Row Normalization (EXCLUDING Column 0 if sink is hidden/ignored)
        if (rowNorm) {
            matrix = matrix.map(row => {
                // If hideSink is ON, exclude col 0 from max calculation so non-sink weights scale 0..1!
                const searchSlice = (hideSink && row.length > 1) ? row.slice(1) : row;
                const maxVal = Math.max(...searchSlice);
                if (maxVal > 0) {
                    return row.map((v, colIdx) => (colIdx === 0 && hideSink) ? 0 : v / maxVal);
                }
                return row;
            });
        }

        // Apply Sqrt scaling so subtle attention signals pop out vividly
        const scaledMatrix = matrix.map(row => row.map(v => Math.sqrt(v)));

        // Setup Canvas Size (Full Width Stretch)
        const totalRows = matrix.length;
        const totalCols = matrix[matrix.length - 1].length;

        const parentW = canvas.parentElement.clientWidth;
        const parentH = canvas.parentElement.clientHeight;

        canvas.width = parentW;
        canvas.height = parentH;

        const marginLeft = 65; // Left margin for row token labels
        const marginTop = 50;  // Space for rotated column labels and sink marker

        const drawW = parentW - marginLeft - 10;
        const drawH = parentH - marginTop - 5;

        const cellW = drawW / totalCols;
        const cellH = drawH / totalRows;

        ctx.fillStyle = '#090a0f';
        ctx.fillRect(0, 0, parentW, parentH);

        // Render Heatmap Cells
        for (let i = 0; i < totalRows; i++) {
            for (let j = 0; j < scaledMatrix[i].length; j++) {
                ctx.fillStyle = viridis(scaledMatrix[i][j]);
                ctx.fillRect(marginLeft + j * cellW, marginTop + i * cellH, cellW + 0.5, cellH + 0.5);
            }
        }

        if (hideSink && totalCols > 0) {
            const sinkX = marginLeft;
            ctx.save();
            ctx.beginPath();
            ctx.rect(sinkX, marginTop, cellW, drawH);
            ctx.clip();
            ctx.fillStyle = '#292d35';
            ctx.fillRect(sinkX, marginTop, cellW, drawH);
            ctx.strokeStyle = '#737b88';
            ctx.lineWidth = 1;
            for (let offset = -drawH; offset < cellW; offset += 8) {
                ctx.beginPath();
                ctx.moveTo(sinkX + offset, marginTop + drawH);
                ctx.lineTo(sinkX + offset + drawH, marginTop);
                ctx.stroke();
            }
            ctx.restore();
        }

        // Draw Axis Token Labels
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '10px JetBrains Mono';

        // Column Labels (Top)
        const allTokens = [...promptTokens, ...genTokens];
        const colStep = Math.max(1, Math.ceil(totalCols / 16));
        for (let j = 0; j < totalCols; j += colStep) {
            const label = vis(allTokens[j] || '').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
            ctx.save();
            ctx.translate(marginLeft + (j + 0.5) * cellW, marginTop - 7);
            ctx.rotate(-Math.PI / 4);
            ctx.fillText(label.slice(0, 5), 0, 0);
            ctx.restore();
        }
        if (hideSink) {
            ctx.fillStyle = '#aab1bc';
            ctx.font = '9px JetBrains Mono';
            ctx.fillText('sink hidden', marginLeft, 12);
        }

        // Each attention row queries the previous token and predicts genTokens[i].
        const rowStep = Math.max(1, Math.ceil(totalRows / 20));
        for (let i = 0; i < totalRows; i += rowStep) {
            const queryIndex = promptTokens.length + i - 1;
            const queryToken = queryIndex >= 0 ? allTokens[queryIndex] || '' : '';
            const cleanStr = vis(queryToken).replace(/&lt;/g, '<').replace(/&gt;/g, '>');
            ctx.fillText(cleanStr.slice(0, 8), 4, marginTop + (i + 0.8) * cellH);
        }

        // Hover Tooltip Logic
        canvas.onmousemove = (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left - marginLeft;
            const y = e.clientY - rect.top - marginTop;
            const col = Math.floor(x / cellW);
            const row = Math.floor(y / cellH);

            if (row >= 0 && row < matrix.length && col >= 0 && col < matrix[row].length) {
                const layerRows = stepsData[row] && stepsData[row][layerIdx] ? stepsData[row][layerIdx] : [];
                const selectedRow = avgHeads
                    ? (layerRows[0] || []).map((_, index) => layerRows.reduce((sum, values) => sum + values[index], 0) / layerRows.length)
                    : (layerRows[headIdx] || []);
                const rawVal = selectedRow[col] || 0;
                const rowMaxValues = hideSink ? selectedRow.slice(1) : selectedRow;
                const rowMax = rowMaxValues.length ? Math.max(...rowMaxValues) : 0;
                const percentOfMax = rowMax > 0 ? (rawVal / rowMax) * 100 : 0;
                const queryIndex = promptTokens.length + row - 1;
                const queryTok = queryIndex >= 0 ? allTokens[queryIndex] || '' : `step ${row}`;
                const predictedTok = genTokens[row] || '';
                const toTok = allTokens[col] || `col ${col}`;
                const cleanQuery = vis(queryTok).replace(/&lt;/g, '<').replace(/&gt;/g, '>');
                const cleanPrediction = vis(predictedTok).replace(/&lt;/g, '<').replace(/&gt;/g, '>');
                const cleanKey = vis(toTok).replace(/&lt;/g, '<').replace(/&gt;/g, '>');
                info.textContent = `Step ${row}: query '${cleanQuery}' predicts '${cleanPrediction}' → key ${col} '${cleanKey}'; raw ${rawVal.toFixed(4)}, ${percentOfMax.toFixed(1)}% of row max`;
            }
        };
    }

    return { render, drawMini };
})();