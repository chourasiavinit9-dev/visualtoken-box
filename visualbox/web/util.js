function vis(str) {
    const s = String(str)
        .replace(/\n/g, '↵').replace(/\t/g, '⇥').replace(/ /g, '·')
        .replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return s === '' ? '∅' : s;
}