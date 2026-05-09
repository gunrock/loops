(function() {
  var canvas = document.getElementById('hero-canvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var N = 120;
  var CONNECTION_DIST = 80;
  var HEARTBEAT_INTERVAL = 3000;
  var HEARTBEAT_DURATION = 1200;
  var dots = [];
  var easterEggActive = false;
  var easterEggTime = 0;
  var lastHeartbeat = 0;
  var heartbeatIdx = -1;
  var w, h;

  var RAINBOW = ['#C0392B','#D4841A','#B8860B','#2D8659','#2B6CB0','#7B4BB3'];
  var heartbeatColorIdx = 0;
  var PROC_COLORS = [
    '#C0392B','#D4841A','#B8860B','#2D8659','#1A8A8A','#2B6CB0',
    '#7B4BB3','#8B2252','#6B8E23','#16A085','#3498DB','#C0588A'
  ];

  function resize() {
    var rect = canvas.parentElement.getBoundingClientRect();
    var oldW = w, oldH = h;
    w = rect.width;
    h = rect.height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (dots.length === 0) {
      initDots();
    } else if (oldW && oldH) {
      var sx = w / oldW, sy = h / oldH;
      for (var i = 0; i < dots.length; i++) {
        dots[i].homeX *= sx; dots[i].homeY *= sy;
        dots[i].x *= sx; dots[i].y *= sy;
      }
    }
  }

  function initDots() {
    dots = [];
    for (var i = 0; i < N; i++) {
      var cx = w * 0.5;
      var cy = h * 0.45;
      var angle = Math.random() * Math.PI * 2;
      var radius = Math.random() * Math.min(w, h) * 0.42;

      dots.push({
        homeX: cx + Math.cos(angle) * radius,
        homeY: cy + Math.sin(angle) * radius,
        x: cx + Math.cos(angle) * radius,
        y: cy + Math.sin(angle) * radius,
        size: 2 + Math.random() * 2.5,
        phase: Math.random() * Math.PI * 2,
        procId: i % PROC_COLORS.length,
        glow: 0
      });
    }
  }

  function hexToRgba(hex, a) {
    var r = parseInt(hex.slice(1,3),16);
    var g = parseInt(hex.slice(3,5),16);
    var b = parseInt(hex.slice(5,7),16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
  }

  function update(t) {
    if (t - lastHeartbeat > HEARTBEAT_INTERVAL) {
      lastHeartbeat = t;
      heartbeatIdx = Math.floor(Math.random() * N);
      heartbeatColorIdx = (heartbeatColorIdx + 1) % RAINBOW.length;
    }

    for (var i = 0; i < dots.length; i++) {
      var d = dots[i];
      d.x = d.homeX + Math.sin(t * 0.0008 + d.phase) * 12;
      d.y = d.homeY + Math.cos(t * 0.0006 + d.phase * 1.3) * 8;

      if (i === heartbeatIdx) {
        var elapsed = t - lastHeartbeat;
        if (elapsed < HEARTBEAT_DURATION) {
          var p = elapsed / HEARTBEAT_DURATION;
          d.glow = p < 0.3 ? p / 0.3 : 1 - (p - 0.3) / 0.7;
        } else {
          d.glow = 0;
        }
      } else {
        d.glow *= 0.95;
      }
    }
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);

    for (var i = 0; i < dots.length; i++) {
      for (var j = i + 1; j < dots.length; j++) {
        var dx = dots[j].x - dots[i].x;
        var dy = dots[j].y - dots[i].y;
        var dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < CONNECTION_DIST) {
          var alpha = (1 - dist / CONNECTION_DIST) * 0.07;
          var glow = Math.max(dots[i].glow, dots[j].glow);
          if (glow > 0.1) {
            ctx.strokeStyle = hexToRgba(RAINBOW[heartbeatColorIdx], alpha + glow * 0.15);
          } else {
            ctx.strokeStyle = 'rgba(26, 26, 26, ' + alpha + ')';
          }
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(dots[i].x, dots[i].y);
          ctx.lineTo(dots[j].x, dots[j].y);
          ctx.stroke();
        }
      }
    }

    for (var i = 0; i < dots.length; i++) {
      var d = dots[i];
      var fade = 1;
      var distFromCenter = Math.sqrt(
        Math.pow(d.x - w * 0.5, 2) + Math.pow(d.y - h * 0.45, 2)
      );
      var maxDist = Math.min(w, h) * 0.48;
      if (distFromCenter > maxDist * 0.65) {
        fade = 1 - (distFromCenter - maxDist * 0.65) / (maxDist * 0.35);
        fade = Math.max(0, fade);
      }
      if (fade <= 0) continue;

      ctx.globalAlpha = fade;

      if (easterEggActive) {
        var elapsed = (performance.now() - (startTime || 0)) - easterEggTime;
        var delay = (i / N) * 600;
        var fadeIn = Math.min(1, Math.max(0, (elapsed - delay) / 500));
        if (elapsed > 12000) {
          var fadeOut = Math.min(1, (elapsed - 12000) / 1500);
          ctx.fillStyle = hexToRgba(PROC_COLORS[d.procId], (1 - fadeOut) * 0.75);
          if (fadeOut >= 1 && i === N - 1) easterEggActive = false;
        } else {
          ctx.fillStyle = hexToRgba(PROC_COLORS[d.procId], fadeIn * 0.75);
        }
      } else if (d.glow > 0.01) {
        ctx.fillStyle = hexToRgba(RAINBOW[heartbeatColorIdx], 0.3 + d.glow * 0.6);
        var glowSize = d.size + d.glow * 4;
        ctx.beginPath();
        ctx.arc(d.x, d.y, glowSize, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = hexToRgba(RAINBOW[heartbeatColorIdx], 0.6 + d.glow * 0.4);
      } else {
        ctx.fillStyle = 'rgba(26, 26, 26, 0.25)';
      }

      ctx.beginPath();
      ctx.arc(d.x, d.y, d.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  var startTime = null;
  function loop(ts) {
    if (!startTime) startTime = ts;
    var t = ts - startTime;
    if (!reducedMotion) {
      update(t);
      draw();
    }
    requestAnimationFrame(loop);
  }

  canvas.addEventListener('click', function() {
    if (reducedMotion) {
      easterEggActive = true;
      draw();
      return;
    }
    if (!easterEggActive) {
      easterEggActive = true;
      easterEggTime = performance.now() - startTime;
    }
  });

  canvas.addEventListener('mousemove', function() {
    canvas.style.cursor = easterEggActive ? 'default' : 'pointer';
  });

  if (reducedMotion) {
    resize();
    draw();
  } else {
    resize();
    requestAnimationFrame(loop);
  }

  window.addEventListener('resize', resize);
})();
