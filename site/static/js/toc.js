document.addEventListener('DOMContentLoaded', function() {
  var tocLinks = document.querySelectorAll('.toc-link');
  if (!tocLinks.length) return;

  var headings = [];
  tocLinks.forEach(function(link) {
    var id = link.getAttribute('href').slice(1);
    var el = document.getElementById(id);
    if (el) headings.push({ el: el, link: link });
  });

  function updateActive() {
    var scrollY = window.scrollY + 100;
    var active = null;
    for (var i = headings.length - 1; i >= 0; i--) {
      if (headings[i].el.offsetTop <= scrollY) {
        active = headings[i];
        break;
      }
    }
    tocLinks.forEach(function(l) { l.classList.remove('active'); });
    if (active) active.link.classList.add('active');
  }

  window.addEventListener('scroll', updateActive, { passive: true });
  updateActive();
});
