async (page) => {
  await page.goto(URL_TO_FETCH, { waitUntil: 'networkidle' });
  await page.waitForTimeout(4000);
  const text = await page.evaluate(function() { return document.querySelector('main').innerText; });
  return text.substring(0, 18000);
}
