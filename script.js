// OTD6 website — smooth scroll + nav highlighting
(function () {
  "use strict";

  // Smooth scroll for nav links
  document.querySelectorAll('.nav-links a[href^="#"]').forEach(function (link) {
    link.addEventListener('click', function (e) {
      e.preventDefault();
      var target = document.querySelector(this.getAttribute('href'));
      if (target) {
        var offset = 70; // nav height
        var top = target.getBoundingClientRect().top + window.pageYOffset - offset;
        window.scrollTo({ top: top, behavior: 'smooth' });
      }
    });
  });

  // Nav link highlighting on scroll
  var sections = document.querySelectorAll('section[id]');
  var navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
  window.addEventListener('scroll', function () {
    var scrollPos = window.pageYOffset + 100;
    sections.forEach(function (section) {
      var top = section.offsetTop;
      var height = section.offsetHeight;
      var id = section.getAttribute('id');
      if (scrollPos >= top && scrollPos < top + height) {
        navLinks.forEach(function (link) {
          if (link.getAttribute('href') === '#' + id) {
            link.style.color = 'var(--accent)';
          } else {
            link.style.color = '';
          }
        });
      }
    });
  });

  // Animate hero code typing effect (optional, subtle)
  var heroCode = document.querySelector('.hero-code pre');
  if (heroCode) {
    heroCode.style.opacity = '0';
    heroCode.style.transform = 'translateY(20px)';
    heroCode.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
    setTimeout(function () {
      heroCode.style.opacity = '1';
      heroCode.style.transform = 'translateY(0)';
    }, 300);
  }

  // Animate stat numbers counting up
  var stats = document.querySelectorAll('.stat-num');
  var animated = false;
  function animateStats() {
    if (animated) return;
    var heroStats = document.querySelector('.hero-stats');
    if (!heroStats) return;
    var rect = heroStats.getBoundingClientRect();
    if (rect.top < window.innerHeight && rect.bottom > 0) {
      animated = true;
      stats.forEach(function (stat) {
        var target = parseInt(stat.textContent, 10);
        if (isNaN(target) || target === 0) return;
        var current = 0;
        var step = Math.max(1, Math.ceil(target / 30));
        var interval = setInterval(function () {
          current += step;
          if (current >= target) {
            stat.textContent = target;
            clearInterval(interval);
          } else {
            stat.textContent = current;
          }
        }, 30);
      });
    }
  }
  window.addEventListener('scroll', animateStats);
  animateStats(); // also try on load
})();
