(() => {
  // Tables retain native semantics; the wrapper provides keyboard-accessible
  // horizontal scrolling on a narrow screen.
  document.querySelectorAll(".research-article table").forEach((table, index) => {
    const wrapper = document.createElement("div");
    wrapper.className = "table-scroll";
    wrapper.tabIndex = 0;
    wrapper.setAttribute("role", "region");
    wrapper.setAttribute("aria-label", `Research table ${index + 1}; scroll horizontally if needed`);
    table.before(wrapper);
    wrapper.append(table);
  });

  const links = [...document.querySelectorAll(".contents a[href^='#']")];
  const sections = links.map((link) => document.getElementById(link.hash.slice(1))).filter(Boolean);
  if (!sections.length || !("IntersectionObserver" in window)) return;

  const activate = (id) => {
    links.forEach((link) => {
      if (link.hash === `#${id}`) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  };
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting);
    if (visible.length) activate(visible[0].target.id);
  }, { rootMargin: "-8% 0px -67% 0px", threshold: 0 });
  sections.forEach((section) => observer.observe(section));
  links.forEach((link) => link.addEventListener("click", () => activate(link.hash.slice(1))));
})();
