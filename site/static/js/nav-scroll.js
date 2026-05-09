document.addEventListener('DOMContentLoaded', function() {
  var nav = document.querySelector('.nav-bar');
  if (!nav) return;
  window.addEventListener('scroll', function() {
    nav.classList.toggle('scrolled', window.scrollY > 50);
  }, { passive: true });
});
