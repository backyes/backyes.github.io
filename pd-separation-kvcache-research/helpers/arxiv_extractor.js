async function main(page) {
  var rawResults = await page.evaluate(function() {
    var out = [];
    var items = document.querySelectorAll('li.arxiv-result');
    for (var i = 0; i < items.length; i++) {
      var r = items[i];
      var titleEl = r.querySelector('p.title');
      var linkEl = r.querySelector('p.list-title a[href*="abs"]');
      var abstractEl = r.querySelector('p.abstract span');
      var metaEl = r.querySelector('p.is-size-7');
      out.push({
        title: titleEl ? titleEl.innerText.trim().substring(0,200) : '',
        link: linkEl ? linkEl.href : '',
        abstract: abstractEl ? abstractEl.innerText.trim().substring(0, 600) : '',
        submitted: metaEl ? metaEl.innerText.trim().substring(0, 200) : ''
      });
    }
    return out;
  });
  return JSON.stringify(rawResults.slice(0, 30));
}
