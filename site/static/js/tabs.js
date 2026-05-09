document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.tabbed-code').forEach(function(container) {
    var buttons = container.querySelectorAll('.tab-btn');
    var panels = container.querySelectorAll('.tab-panel');
    buttons.forEach(function(btn) {
      btn.addEventListener('click', function() {
        var target = btn.getAttribute('data-tab');
        buttons.forEach(function(b) { b.classList.remove('active'); });
        panels.forEach(function(p) { p.classList.remove('active'); });
        btn.classList.add('active');
        container.querySelector('#' + target).classList.add('active');
      });
    });
  });
});
