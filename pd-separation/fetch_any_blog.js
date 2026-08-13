async (page) => {
  const url = process.argv[1] || PAGE_URL;
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);
  return await page.evaluate(() => {
    return document.querySelector('main') ? document.querySelector('main').innerText : document.body.innerText;
  });
}
