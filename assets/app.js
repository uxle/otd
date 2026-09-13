// OTD viewer — vanilla JavaScript, zero libraries. The server renders every
// pixel; this file is just an editor, an <img>, and an input relay.
(function () {
  "use strict";

  var editor = document.getElementById("editor");
  var highlight = document.getElementById("highlight");
  var preview = document.getElementById("preview");
  var consoleEl = document.getElementById("console");
  var flag = document.getElementById("rendering-flag");
  var statTitle = document.getElementById("stat-title");
  var statParts = document.getElementById("stat-parts");
  var statMass = document.getElementById("stat-mass");
  var statLines = document.getElementById("stat-lines");
  var statStatus = document.getElementById("stat-status");
  var examplesSel = document.getElementById("examples");
  var modal = document.getElementById("modal");
  var modalTitle = document.getElementById("modal-title");
  var modalBody = document.getElementById("modal-body");

  var cam = null;          // {yaw, pitch, dist, target}
  var lastCodeHash = null;
  var renderTimer = null;
  var inFlight = false;
  var inFlightAbort = null;
  var interacting = false; // true while a drag/touch/zoom is in progress
  var settleTimer = null;  // fires one full-quality render after interaction stops

  // ---------- syntax highlight ----------
  var KEYWORDS = "scene unit version gravity camera hide show sphere cube cylinder cone torus pyramid prism capsule wedge plane tube helix rope extrude revolve sweep loft text import terrain metaball hollow group blend add subtract intersect at rotate scale mirror smooth subdiv repeat grid ring scatter define use material color simulate ask export print if else end for to by while break continue and or not true false is mod assert in";
  var KW = {};
  KEYWORDS.split(" ").forEach(function (k) { KW[k] = 1; });

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function highlightCode(src) {
    var out = "";
    var lines = src.split("\n");
    lines.forEach(function (line, li) {
      // comment split: the first '#' that does NOT start a #rrggbb color
      var codePart = line;
      var comPart = "";
      var hash = -1;
      for (var h = line.indexOf("#"); h >= 0; h = line.indexOf("#", h + 1)) {
        var six = line.slice(h + 1, h + 7);
        var after = line.charAt(h + 7);
        if (/^[0-9a-fA-F]{6}$/.test(six) && !/[0-9a-zA-Z_]/.test(after)) {
          continue; // this # starts a hex color — keep scanning
        }
        hash = h;
        break;
      }
      if (hash >= 0) {
        codePart = line.slice(0, hash);
        comPart = line.slice(hash);
      }
      var html = "";
      var re = /("(?:[^"\\]|\\.)*")|(\#[0-9a-fA-F]{6}(?![0-9a-zA-Z_]))|(\d*\.?\d+\s*(?:mm|cm|m|in|ft|deg|°)?)|([A-Za-z_][A-Za-z0-9_]*)/g;
      var m;
      var last = 0;
      while ((m = re.exec(codePart)) !== null) {
        html += esc(codePart.slice(last, m.index));
        if (m[1]) html += '<span class="tk-str">' + esc(m[1]) + "</span>";
        else if (m[2]) html += '<span class="tk-hex">' + esc(m[2]) + "</span>";
        else if (m[3]) html += '<span class="tk-num">' + esc(m[3]) + "</span>";
        else if (m[4] && KW[m[4]]) html += '<span class="tk-kw">' + esc(m[4]) + "</span>";
        else if (m[4]) html += esc(m[4]);
        last = m.index + m[0].length;
      }
      html += esc(codePart.slice(last));
      if (comPart) html += '<span class="tk-com">' + esc(comPart) + "</span>";
      out += html + "\n";
    });
    highlight.innerHTML = out;
  }

  function syncScroll() {
    highlight.scrollTop = editor.scrollTop;
    highlight.scrollLeft = editor.scrollLeft;
  }

  editor.addEventListener("input", function () {
    highlightCode(editor.value);
    scheduleRender(280);
    updateLines();
  });
  editor.addEventListener("scroll", syncScroll);
  editor.addEventListener("keydown", function (e) {
    if (e.key === "Tab") {
      e.preventDefault();
      var s = editor.selectionStart;
      editor.setRangeText("  ", s, editor.selectionEnd, "end");
      highlightCode(editor.value);
      scheduleRender(280);
    } else if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      scheduleRender(0);
    }
  });

  // ---------- render loop ----------
  function scheduleRender(ms) {
    clearTimeout(renderTimer);
    renderTimer = setTimeout(doRender, ms);
  }

  // PERF: while the user is actively dragging/zooming, ask the server for
  // a smaller, ssaa=1 preview instead of the full ssaa=2 frame — on a slow
  // device or a slow connection the full-quality render can easily take
  // longer than the gap between drag frames, and rubber-banding through a
  // queue of stale full-quality frames is exactly what "lag" feels like.
  // The moment interaction stops, one full-quality frame is requested to
  // settle on a crisp final image (see the mouseup/touchend/wheel-idle
  // handlers below). This mirrors how Blender/Fusion360/SketchUp-style
  // viewers drop resolution during orbit and refine once you let go.
  function doRender() {
    // A newer render supersedes whatever's in flight — abort it instead of
    // queuing behind it, so responsiveness during a drag is never bounded
    // by a stale frame the user has already moved past.
    if (inFlightAbort) {
      inFlightAbort.abort();
      inFlightAbort = null;
    }
    var code = editor.value;
    var scale = interacting ? 0.6 : 1.0;
    var body = {
      code: code,
      ssaa: interacting ? 1 : 2,
      width: Math.floor(Math.min(1100, Math.max(320, Math.floor(window.innerWidth * 0.52))) * scale),
      height: Math.floor(Math.min(760, Math.max(220, Math.floor(window.innerHeight * 0.6))) * scale),
    };
    if (cam) body.cam = cam;
    var controller = ("AbortController" in window) ? new AbortController() : null;
    inFlightAbort = controller;
    inFlight = true;
    flag.style.display = "block";
    fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller ? controller.signal : undefined,
    })
      .then(function (r) { return r.json(); })
      .then(function (j) { handleRender(j, code); })
      .catch(function (e) {
        if (e && e.name === "AbortError") return; // superseded — not a real error
        setStatus("network error", "bad");
      })
      .finally(function () {
        if (inFlightAbort === controller) {
          inFlightAbort = null;
          inFlight = false;
          flag.style.display = "none";
        }
      });
  }

  // Called once interaction (drag/zoom/touch) stops: drop back to full
  // quality and request one settled frame.
  function settleRender() {
    interacting = false;
    scheduleRender(0);
  }

  function handleRender(j, code) {
    if (j.png) preview.src = "data:image/png;base64," + j.png;
    if (j.cam) cam = j.cam;
    if (j.fit) {
      cam = { yaw: j.fit.yaw, pitch: j.fit.pitch, dist: j.fit.dist, target: j.fit.target };
    }
    // stats
    if (j.stats) {
      statTitle.textContent = j.stats.title || "—";
      statParts.textContent = j.stats.parts + " object" + (j.stats.parts === 1 ? "" : "s");
      statMass.textContent = fmtMass(j.stats.mass_g);
    }
    renderConsole(j.console || [], j.errors || []);
    var errCount = (j.errors || []).length;
    var empty = !j.stats || !j.stats.total_parts;
    setStatus(errCount ? errCount + " error" + (errCount > 1 ? "s" : "") : (empty ? "empty scene" : "ok"), errCount ? "bad" : (empty ? "" : "ok"));
  }

  function fmtMass(g) {
    if (g >= 1000000) return (g / 1000000).toFixed(2) + " t";
    if (g >= 1000) return (g / 1000).toFixed(2) + " kg";
    if (g >= 1) return g.toFixed(1) + " g";
    return (g * 1000).toFixed(0) + " mg";
  }

  function setStatus(text, cls) {
    statStatus.textContent = text;
    statStatus.className = "chip " + (cls || "");
  }

  function updateLines() {
    var n = editor.value.split("\n").length;
    statLines.textContent = n + " line" + (n === 1 ? "" : "s");
  }

  function renderConsole(lines, errors) {
    var html = "";
    errors.forEach(function (e) {
      html += '<div class="cline error" data-line="' + e.line + '">✗ line ' + e.line + ": " + esc(e.msg) +
        (e.hint ? " — <i>" + esc(e.hint) + "</i>" : "") + "</div>";
    });
    lines.forEach(function (c) {
      var k = c.kind === "warn" ? "⚠ " : c.kind === "answer" ? "→ " : c.kind === "error" ? "✗ " : "";
      html += '<div class="cline ' + c.kind + '">' + k + esc(c.text) + "</div>";
    });
    if (!html) html = '<div class="cline info">console — answers to ask "…" appear here</div>';
    consoleEl.innerHTML = html;
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }

  // click an error line → jump to it
  consoleEl.addEventListener("click", function (e) {
    var el = e.target.closest(".error");
    if (!el) return;
    var line = parseInt(el.getAttribute("data-line"), 10);
    if (!line) return;
    var lines = editor.value.split("\n");
    var pos = 0;
    for (var i = 0; i < line - 1 && i < lines.length; i++) pos += lines[i].length + 1;
    editor.focus();
    editor.setSelectionRange(pos, pos + (lines[line - 1] || "").length);
  });

  // ---------- camera controls ----------
  var drag = null;
  var holder = document.getElementById("canvas-holder");
  holder.addEventListener("mousedown", function (e) {
    drag = { x: e.clientX, y: e.clientY, yaw: cam ? cam.yaw : 40, pitch: cam ? cam.pitch : 28 };
    interacting = true;
    e.preventDefault();
  });
  window.addEventListener("mousemove", function (e) {
    if (!drag) return;
    var dx = e.clientX - drag.x;
    var dy = e.clientY - drag.y;
    cam.yaw = drag.yaw - dx * 0.4;
    cam.pitch = Math.max(-88, Math.min(88, drag.pitch + dy * 0.4));
    scheduleRender(30);
  });
  window.addEventListener("mouseup", function () {
    if (!drag) return;
    drag = null;
    settleRender();
  });
  holder.addEventListener("wheel", function (e) {
    e.preventDefault();
    if (cam && cam.dist) {
      cam.dist = Math.max(5, cam.dist * (e.deltaY > 0 ? 1.12 : 0.89));
      interacting = true;
      clearTimeout(settleTimer);
      settleTimer = setTimeout(settleRender, 220); // wheel has no natural "end" event
      scheduleRender(30);
    }
  }, { passive: false });
  // touch orbit
  var touch = null;
  holder.addEventListener("touchstart", function (e) {
    if (e.touches.length === 1) {
      touch = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      interacting = true;
    }
  });
  holder.addEventListener("touchmove", function (e) {
    if (touch && e.touches.length === 1 && cam) {
      var dx = e.touches[0].clientX - touch.x;
      var dy = e.touches[0].clientY - touch.y;
      cam.yaw -= dx * 0.5;
      cam.pitch = Math.max(-88, Math.min(88, cam.pitch + dy * 0.5));
      touch = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      scheduleRender(60);
      e.preventDefault();
    }
  }, { passive: false });
  holder.addEventListener("touchend", function () {
    if (!touch) return;
    touch = null;
    settleRender();
  });
  holder.addEventListener("dblclick", function () {
    cam = null; // refit
    scheduleRender(0);
  });

  document.querySelectorAll(".vw").forEach(function (b) {
    b.addEventListener("click", function () {
      // apply a view preset relative to current target
      fetch("/api/render", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: editor.value, ssaa: 2 }),
      })
        .then(function (r) { return r.json(); })
        .then(function (j) {
          if (j.fit) {
            var dist = j.fit.dist;
            var target = j.fit.target;
            var v = b.getAttribute("data-view");
            var yaw = { iso: 40, front: 0, side: 90, top: 0 }[v] || 0;
            var pitch = { iso: 28, front: 8, side: 8, top: 88 }[v] || 28;
            cam = { yaw: yaw, pitch: pitch, dist: dist, target: target };
            scheduleRender(0);
          }
        });
    });
  });

  // ---------- exports ----------
  document.querySelectorAll(".ex").forEach(function (b) {
    b.addEventListener("click", function () {
      var fmt = b.getAttribute("data-fmt");
      fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: editor.value, fmt: fmt }),
      })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (j) { throw new Error(j.errors[0].msg); });
          return r.blob().then(function (blob) {
            var cd = r.headers.get("Content-Disposition") || "";
            var m = /filename="([^"]+)"/.exec(cd);
            var a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = m ? m[1] : "model." + fmt;
            a.click();
            URL.revokeObjectURL(a.href);
          });
        })
        .catch(function (e) { setStatus("export failed: " + e.message, "bad"); });
    });
  });

  // ---------- examples & lessons & help ----------
  fetch("/api/examples").then(function (r) { return r.json(); }).then(function (j) {
    (j.examples || []).forEach(function (ex) {
      var o = document.createElement("option");
      o.value = ex.file;
      o.textContent = ex.title;
      examplesSel.appendChild(o);
    });
    if (j.examples && j.examples.length) {
      loadCode(j.examples[0].code);
    }
  });

  examplesSel.addEventListener("change", function () {
    if (!examplesSel.value) return;
    fetch("/api/examples").then(function (r) { return r.json(); }).then(function (j) {
      var ex = (j.examples || []).find(function (e) { return e.file === examplesSel.value; });
      if (ex) {
        loadCode(ex.code);
        statTitle.textContent = ex.title;
      }
    });
  });

  function loadCode(code) {
    editor.value = code;
    highlightCode(code);
    cam = null; // refit to new scene
    updateLines();
    scheduleRender(0);
  }

  document.getElementById("btn-lessons").addEventListener("click", function () {
    modalTitle.textContent = "20 lessons — from hello cube to a robot arm";
    modalBody.innerHTML = "loading…";
    modal.showModal();
    fetch("/api/lessons").then(function (r) { return r.json(); }).then(function (j) {
      var html = "";
      (j.lessons || []).forEach(function (l) {
        html += '<div class="lesson" data-n="' + l.n + '">' +
          '<span class="ln">' + l.n + "</span>" +
          '<span><div class="lt">' + esc(l.title) + '</div><div class="lg">' + esc(l.goal) + "</div></span></div>";
      });
      modalBody.innerHTML = html;
      modalBody.querySelectorAll(".lesson").forEach(function (el) {
        el.addEventListener("click", function () {
          var n = parseInt(el.getAttribute("data-n"), 10);
          var l = j.lessons.find(function (x) { return x.n === n; });
          modalBody.innerHTML =
            "<h3>Lesson " + l.n + " — " + esc(l.title) + "</h3>" +
            "<p>" + esc(l.goal) + "</p>" +
            '<pre class="codeblock">' + esc(l.code) + "</pre>" +
            '<p><b>Try this:</b> ' + esc(l.tryit) + "</p>" +
            '<button id="lesson-load">load this code ↵</button> <button id="lesson-back">← back to the list</button>';
          document.getElementById("lesson-load").addEventListener("click", function () {
            modal.close();
            loadCode(l.code);
          });
          document.getElementById("lesson-back").addEventListener("click", function () {
            document.getElementById("btn-lessons").click();
          });
        });
      });
    });
  });

  document.getElementById("btn-help").addEventListener("click", function () {
    modalTitle.textContent = "OTD cheat sheet — 75 keywords + 25 functions, 27 materials, 147 colors";
    var html =
      "<table class='kw-table'>" +
      "<tr><th>shapes</th><td><code>sphere cube cylinder cone torus pyramid prism capsule wedge plane tube helix</code></td></tr>" +
      "<tr><th>builders</th><td><code>extrude revolve sweep loft text import terrain metaball hollow group</code></td></tr>" +
      "<tr><th>booleans</th><td><code>a + b</code> fuse · <code>a - b</code> cut · <code>a & b</code> overlap</td></tr>" +
      "<tr><th>place</th><td><code>at (x, y, z)</code> stand there · <code>rotate (x, y, z)</code> spin · <code>scale 2</code> grow · <code>mirror x</code></td></tr>" +
      "<tr><th>patterns</th><td><code>repeat(n: 4, step: (2cm, 0, 0))</code> · <code>grid(nx: 3, nz: 3)</code> · <code>ring(n: 8, radius: 5cm)</code> · <code>scatter(n: 20)</code></td></tr>" +
      "<tr><th>decide (2.1)</th><td><code>if x > 5cm: cube 1cm else: sphere 1cm</code> · <code>if … else if … else … end</code></td></tr>" +
      "<tr><th>repeat (2.1)</th><td><code>for i = 1 to 10 by 2</code> · <code>for r in [1cm, 2cm]</code> · <code>while x < 5cm</code> · <code>break</code> · <code>continue</code></td></tr>" +
      "<tr><th>logic (2.1)</th><td><code>and or not</code> · <code>is / is not / in</code> · <code>x += 1cm</code> · <code>assert x > 0, \"msg\"</code></td></tr>" +
      "<tr><th>parts</th><td><code>define wheel(r) = …</code> then <code>use wheel(r: 3cm)</code></td></tr>" +
      "<tr><th>looks</th><td><code>material: steel</code> · <code>color: ivory</code> · 27 materials, 147 colors</td></tr>" +
      "<tr><th>science</th><td><code>ask \"mass?\"</code> · <code>simulate: drop / float / collapse</code> · <code>gravity: moon</code></td></tr>" +
      "<tr><th>units</th><td><code>mm cm m in ft deg</code> — bare numbers are centimeters</td></tr>" +
      "</table>" +
      "<p style='color:var(--dim)'>The cup in 8 lines: <code>cylinder(top: 4cm, bottom: 3cm, height: 10cm) - hollow(wall: 3mm)</code></p>";
    modalBody.innerHTML = html;
    modal.showModal();
  });

  document.getElementById("modal-close").addEventListener("click", function () { modal.close(); });

  // local storage persistence
  try {
    var saved = localStorage.getItem("otd-code");
    if (saved) {
      // only use saved code after examples fetch (first example wins unless saved)
      setTimeout(function () {
        if (editor.value && localStorage.getItem("otd-chose")) {
          loadCode(saved);
        }
      }, 50);
      editor.addEventListener("input", function () {
        localStorage.setItem("otd-code", editor.value);
        localStorage.setItem("otd-chose", "1");
      });
    }
  } catch (e) {}

  updateLines();
})();
