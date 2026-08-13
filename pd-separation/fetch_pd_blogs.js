async (page) => {
  const url = 'https://www.lmsys.org/blog/2025-05-05-large-scale-ep';
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);
  const extractor = () => {
    const h2s = Array.from(document.querySelectorAll('h2, h3'));
    const starts = ['prefill and decode disaggregation', 'issues with unified scheduling', 'implementation details'];
    const output = [];
    for (const h of h2s) {
      const t = h.textContent.toLowerCase();
      if (starts.some(s => t.includes(s))) {
        output.push('\n=== ' + h.textContent.trim().toUpperCase() + ' ===\n');
        let node = h.nextElementSibling;
        while (node && node.tagName !== 'H2' && node.tagName !== 'H3') {
          output.push(node.innerText || '');
          node = node.nextElementSibling;
        }
      }
    }
    return output.join('\n');
  };
  const result = await page.evaluate(extractor);
  return result;
}
