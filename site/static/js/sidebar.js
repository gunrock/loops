document.addEventListener('DOMContentLoaded', function() {
  var toggle = document.querySelector('.sidebar-toggle');
  var sidebar = document.querySelector('.sidebar');
  var overlay = document.querySelector('.sidebar-overlay');

  if (toggle && sidebar) {
    toggle.addEventListener('click', function() {
      sidebar.classList.toggle('open');
      if (overlay) overlay.classList.toggle('open');
    });
    if (overlay) {
      overlay.addEventListener('click', function() {
        sidebar.classList.remove('open');
        overlay.classList.remove('open');
      });
    }
  }

  document.querySelectorAll('.sidebar-expand').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var expanded = btn.getAttribute('aria-expanded') === 'true';
      btn.setAttribute('aria-expanded', !expanded);
      var children = btn.nextElementSibling;
      if (children) children.classList.toggle('open');
    });
  });

  var current = window.location.pathname;
  document.querySelectorAll('.sidebar-link').forEach(function(link) {
    if (link.getAttribute('href') === current || link.getAttribute('href') === current + 'index.html') {
      link.classList.add('active');
    }
  });
});
