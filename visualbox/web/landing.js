(() => {
  const el = document.getElementById('landing');
  if (!el) return;

  // After the first launch in this tab, skip the intro on refresh (handy while developing)
  let seen = false;
  try { seen = sessionStorage.getItem('gb-intro-seen') === '1'; } catch (e) {}
  if (seen) { el.remove(); return; }

  // Decorative causal-attention grid (purely visual, not real data)
  const grid = document.getElementById('lp-grid');
  const N = 14, palette = ['#440154', '#3b528b', '#21908d', '#5ec962', '#fde725'];
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const d = document.createElement('div');
      d.className = 'lp-cell';
      if (c > r) {
        d.style.visibility = 'hidden';
      } else {
        let v = Math.min(1, 0.15 * rnd() + Math.exp(-(r - c) / 2.5) * (0.4 + 0.6 * rnd()));
        if (c === 0) v = 0.02;   // dark first column, like the attention sink
        d.style.background = palette[Math.min(4, Math.floor(v * 5))];
        d.style.animationDelay = (r * 40 + c * 8) + 'ms';
      }
      grid.appendChild(d);
    }
  }

  function onKey(e) { if (e.key === 'Enter') launch(); }
  function launch() {
    try { sessionStorage.setItem('gb-intro-seen', '1'); } catch (e) {}
    document.removeEventListener('keydown', onKey);
    el.classList.add('lp-hide');
    setTimeout(() => { el.remove(); window.dispatchEvent(new Event('resize')); }, 500);
  }
  document.getElementById('lp-launch').addEventListener('click', launch);
  document.addEventListener('keydown', onKey);
  document.getElementById('lp-launch').focus();
})();