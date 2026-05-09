document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.code-copy').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var text = '';
      var cite = btn.closest('.cite-block');
      if (cite) {
        var clone = cite.cloneNode(true);
        clone.querySelectorAll('button').forEach(function(b) { b.remove(); });
        text = clone.textContent.trim();
      } else {
        var block = btn.closest('.code-block') || btn.closest('.tabbed-code');
        var activePanel = block.querySelector('.tab-panel.active pre code') || block.querySelector('pre code');
        if (!activePanel) return;
        text = activePanel.textContent;
      }
      navigator.clipboard.writeText(text).then(function() {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(function() {
          btn.textContent = 'Copy';
          btn.classList.remove('copied');
        }, 1500);
      });
    });
  });
});
