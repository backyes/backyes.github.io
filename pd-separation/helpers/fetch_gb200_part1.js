async (page) => {
  const url = 'https://www.lmsys.org/blog/2025-06-16-gb200-part-1';
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2500);
  return await page.evaluate(() => {
    const targets = ['prefill', 'decode', 'disagg', 'transfer', 'architecture', 'implementation'];
    const h2s = Array.from(document.querySelectorAll('h2, h3'));
    const output = [];
    for (const h of h2s) {
      const t = h.textContent.toLowerCase();
      if (targets.some(s => t.includes(s))) {
        output.push('\n=== ' + h.textContent.trim().toUpperCase() + ' ===\n');
        let node = h.nextElementSibling;
        while (node && node.tagName !== 'H2' && node.tagName !== 'H3') {
          output.push(node.innerText || '');
          node = node.nextElementSibling;
        }
      }
    }
    return output.join('\n');
  });
}
