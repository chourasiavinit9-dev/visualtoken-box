# Interpreting GlassBox Output

A guide to reading each visualization panel in the dashboard.

---

## 🔥 Attention Heatmap

**What it shows**: For each generated token, this grid shows where each
transformer layer was "looking" in the input sequence.

- **Rows** = transformer layers (bottom = early, top = final)
- **Columns** = input + previously generated tokens
- **Bright cells** = high attention weight (the model focused heavily here)

### What to look for:

| Pattern | Meaning |
|---|---|
| Bright column on token `1` (BOS) | Normal — "BOS sink heads" dump baseline attention here |
| Attention spreading to many tokens | Model uncertain, broad context needed |
| Sharp spike on 1-2 tokens | Model very confident about specific context |
| Lower layers attending locally | Common — early layers handle syntax |
| Upper layers attending globally | Common — later layers handle semantics |

### Controls:
- **Normalize rows** — makes each row sum to 1 (easier to compare layers)
- **Hide BOS** — removes the first-token spike for clearer view of other tokens
- **Head selector** — inspect a specific head vs. the average

---

## 🔭 Logit Lens

**What it shows**: At each layer's hidden state, GlassBox projects it through
the final norm + LM head to predict: *"If generation stopped here, what word
would be produced?"*

- **Left** = early layers (raw, often garbage)
- **Right** = late layers (progressively confident)
- **Color intensity** = probability of that layer's top-1 prediction

### What to look for:

| Pattern | Meaning |
|---|---|
| Final prediction crystallizes at layer 15+ | The upper half of the model does most "reasoning" |
| Prediction unchanged from layer 5 → 26 | This was an "easy" token (e.g., punctuation) |
| Prediction flips in the last 3 layers | The model second-guessed itself at the end |
| Gibberish predictions in early layers | Normal — early layers haven't yet integrated full context |

---

## 📊 Top-K Candidates

**What it shows**: The top-10 tokens the model was considering at this
generation step, with their raw logit scores.

- **High score difference** between #1 and #2 → model is very confident
- **Tight spread** → model was uncertain, many tokens were plausible

### Entropy indicator:
- **< 3 bits** → Very confident (predictable text)
- **3-8 bits** → Normal uncertainty
- **> 10 bits** → Very uncertain (creative or ambiguous context)

---

## 🔀 What-If Branching

Click any candidate token in the Top-K panel to **force** the model to select
that token and continue generation from there.

This lets you explore:
- *"What if the model had chosen a different word?"*
- *"Does the output collapse to the same answer anyway?"*
- *"Which token choices lead to hallucinations?"*

Each branch creates an independent generation thread visible in the chat panel.

---

## Reading Attention Per Head

When you click on a specific head in the head selector:

- **Induction-like heads**: Strong diagonal stripes — copying recent patterns
- **Previous-token heads**: Bright cell exactly 1 position back
- **BOS heads**: Almost all weight on token position 0
- **Broad heads**: Diffuse, uniform attention — integrating context
