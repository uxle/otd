(function () {
  "use strict";

  /* ---- mobile nav toggle ---- */
  var toggle = document.getElementById("nav-toggle");
  var links = document.getElementById("nav-links");
  if (toggle && links) {
    toggle.addEventListener("click", function () {
      var open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    links.querySelectorAll(".nav-link").forEach(function (link) {
      link.addEventListener("click", function () {
        links.classList.remove("open");
        toggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  /* ---- copy-to-clipboard for the terminal commands + code sample ---- */
  document.querySelectorAll(".copy-btn").forEach(function (btn) {
    var targetId = btn.getAttribute("data-copy-target");
    var target = targetId ? document.getElementById(targetId) : null;
    if (!target) return;
    var defaultLabel = btn.textContent;
    btn.addEventListener("click", function () {
      var text = target.textContent;
      var done = function () {
        btn.textContent = "Copied";
        btn.classList.add("copied");
        setTimeout(function () {
          btn.textContent = defaultLabel;
          btn.classList.remove("copied");
        }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(function () {
          fallbackCopy(text);
          done();
        });
      } else {
        fallbackCopy(text);
        done();
      }
    });
  });

  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
  }

  /* ---- eight-camera ring diagram: real geometry, not a static image ---- */
  var ringGroup = document.querySelector(".ring-cams");
  if (ringGroup) {
    var cx = 140, cy = 140, r = 96, count = 8, size = 14;
    var svgNS = "http://www.w3.org/2000/svg";
    for (var i = 0; i < count; i++) {
      var angle = (i / count) * Math.PI * 2 - Math.PI / 2;
      var x = cx + r * Math.cos(angle);
      var y = cy + r * Math.sin(angle);

      var line = document.createElementNS(svgNS, "line");
      line.setAttribute("x1", cx);
      line.setAttribute("y1", cy);
      line.setAttribute("x2", x.toFixed(1));
      line.setAttribute("y2", y.toFixed(1));
      line.setAttribute("class", "ring-line");
      ringGroup.appendChild(line);

      var rect = document.createElementNS(svgNS, "rect");
      rect.setAttribute("x", (x - size / 2).toFixed(1));
      rect.setAttribute("y", (y - size / 2).toFixed(1));
      rect.setAttribute("width", size);
      rect.setAttribute("height", size);
      rect.setAttribute("rx", 3);
      rect.setAttribute("class", "ring-cam");
      ringGroup.appendChild(rect);
    }
  }
})();
