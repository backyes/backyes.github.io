async function main(page) {
  var rawResults = await page.evaluate(function() {
    var out = [];
    var items = document.querySelectorAll('li.arxiv-result');
    for (var i = 0; i < items.length; i++) {
      var r = items[i];
      var titleEl = r.querySelector('p.title');
      var linkEl = r.querySelector('p.list-title a[href*="abs"]');
      var abstractParagraph = r.querySelector('p.abstract');
      var abstractText = '';
      if (abstractParagraph) {
        var html = abstractParagraph.innerHTML;
        var idx = html.indexOf('</span>');
        if (idx > -1) html = html.substring(idx + 7);
        abstractText = html.replace(/<[^>]*>/g, '').trim().substring(0, 800);
      }
      var metaEl = r.querySelector('p.is-size-7');
      out.push({
        title: titleEl ? titleEl.innerText.trim().substring(0,200) : '',
        link: linkEl ? linkEl.href : '',
        abstract: abstractText,
        submitted: metaEl ? metaEl.innerText.trim().substring(0, 200) : ''
      });
    }
    return out;
  });
  return JSON.stringify(rawResults.slice(0, 30));
}
