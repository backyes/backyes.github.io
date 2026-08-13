async (page) => {
  const text = await page.evaluate(() => document.querySelector('main').innerText);
  return text;
}
