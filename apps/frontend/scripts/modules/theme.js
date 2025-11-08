import { el } from './dom.js';

// Theme Management
export const theme = {
  KEY: "bluez-ui-theme",
  
  apply(themeName) {
    const normalized = themeName === "light" ? "light" : "dark";
    document.body.dataset.theme = normalized;
    document.body.style.colorScheme = normalized;
    
    if (el.themeToggle) {
      const label = el.themeToggle.querySelector(".theme-label");
      const icon = el.themeToggle.querySelector("span[aria-hidden='true']");
      if (label) label.textContent = normalized === "dark" ? "Dark" : "Light";
      if (icon) icon.textContent = normalized === "dark" ? "🌙" : "☀️";
    }
  },
  
  init() {
    if (el.themeToggle) {
      this.apply(document.body.dataset.theme);
      el.themeToggle.onclick = () => {
        const next = document.body.dataset.theme === "dark" ? "light" : "dark";
        this.apply(next);
        localStorage.setItem(this.KEY, next);
      };
    } else {
      document.body.dataset.theme = "dark";
      document.body.style.colorScheme = "dark";
    }
  }
};
