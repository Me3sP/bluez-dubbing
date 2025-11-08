export const models = {
  refresh(select, models = [], langCode) {
    if (!select) return;

    const prev = select.value;
    const frag = document.createDocumentFragment();
    frag.appendChild(new Option("Auto", "auto"));

    const code = langCode?.toLowerCase() || "";
    let foundPrev = prev === "auto";

    models.forEach(model => {
      const langs = model?.languages;
      if (!Array.isArray(langs) || !langs.length) return;

      const supports = !code || langs.some(l => String(l).toLowerCase() === code);
      if (!supports) return;

      const opt = new Option(model.key, model.key);
      if (model.key === prev) {
        opt.selected = true;
        foundPrev = true;
      }
      frag.appendChild(opt);
    });

    select.replaceChildren(frag);
    if (!foundPrev) select.value = "auto";
  }
};
