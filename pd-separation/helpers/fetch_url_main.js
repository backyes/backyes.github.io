async function main(page, url) {
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);
  var text = await page.evaluate(function() {
    var el = document.querySelector('article') || document.querySelector('div.repository-content') || document.querySelector('[class*="markdown"]') || document.body;
    return el.innerText;
  });
  return text.substring(0, 15000);
}
