/* binference docs theme: theme toggle, phone menu, sidebar groups and outline marker
   (Sphinx's doctools.js handles the "/" search key). The saved theme is applied before paint by an inline script in layout.html. */
(function () {
  var root = document.documentElement;

  function isDark() {
    var t = root.dataset.theme;
    return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function label(button) {
    button.setAttribute("aria-label", isDark() ? "Switch to light theme" : "Switch to dark theme");
  }
  document.querySelectorAll("[data-theme-toggle]").forEach(function (button) {
    label(button);
    button.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("binference-theme", next); } catch (e) {}
      label(button);
    });
  });

  /* Sidebar groups: the one holding the current page is open; the others open on click. */
  document.querySelectorAll(".side p.caption").forEach(function (caption) {
    var list = caption.nextElementSibling, text = caption.querySelector(".caption-text");
    if (!list || !text) return;
    var button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-expanded", list.querySelector(".current") ? "true" : "false");
    caption.replaceChild(button, text);
    button.appendChild(text);
    button.addEventListener("click", function () {
      button.setAttribute("aria-expanded", button.getAttribute("aria-expanded") === "true" ? "false" : "true");
    });
  });

  /* Phone menu. */
  var menu = document.querySelector("[data-menu]"), nav = document.getElementById("sidebar");
  if (menu && nav) {
    menu.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      menu.setAttribute("aria-expanded", open ? "true" : "false");
      menu.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && nav.classList.contains("open")) { menu.click(); menu.focus(); }
    });
  }

  /* On this page: mark the last section whose heading has passed under the top bar. */
  var links = Array.from(document.querySelectorAll(".toc > ul > li > ul > li > a")), shown = -1;
  if (!links.length) return;
  var targets = links.map(function (a) { return document.getElementById(decodeURIComponent(a.hash.slice(1))); });
  function mark() {
    var current = 0;
    targets.forEach(function (t, i) { if (t && t.getBoundingClientRect().top < 140) current = i; });
    if (innerHeight + scrollY >= document.body.scrollHeight - 2) current = links.length - 1;
    if (current === shown) return;
    shown = current;
    links.forEach(function (a, i) {
      a.classList.toggle("on", i === current);
      if (i === current) a.setAttribute("aria-current", "location"); else a.removeAttribute("aria-current");
    });
  }
  var pending = false;
  addEventListener("scroll", function () {
    if (pending) return;
    pending = true;
    requestAnimationFrame(function () { pending = false; mark(); });
  }, { passive: true });
  mark();
})();

/* After sphinx-copybutton has added its buttons, name them; once MathJax has typeset, let the
   keyboard reach every block that scrolls sideways (the blocks binference.css lets scroll). */
addEventListener("load", function () {
  document.querySelectorAll("button.copybtn:not([aria-label])").forEach(function (b) {
    b.setAttribute("aria-label", "Copy code");
  });
  var typeset = window.MathJax && MathJax.startup ? MathJax.startup.promise : Promise.resolve();
  typeset.then(function () {
    document.querySelectorAll("main div.highlight, main div.math, main table.docutils, main dl.py > dt, " +
      "div.cell_output .output pre, div.cell_output .output.text_html").forEach(function (el) {
      if (el.scrollWidth > el.clientWidth && !el.hasAttribute("tabindex")) el.setAttribute("tabindex", "0");
    });
  });
});

/* The home page's mark resamples its draws unless the reader prefers reduced motion; a button
   stops it (WCAG 2.2.2). */
(function () {
  var hero = document.querySelector(".hero");
  if (!hero || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  var button = document.createElement("button");
  button.type = "button";
  button.textContent = "Pause animation";
  button.addEventListener("click", function () {
    button.textContent = hero.classList.toggle("paused") ? "Play animation" : "Pause animation";
  });
  hero.appendChild(button);
  hero.classList.add("moving");
})();
