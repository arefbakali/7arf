// 7arf — interface. Aucune bibliothèque : DOM, fetch et un peu de SVG.
(() => {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const REFERENCE = [["2", "ء"], ["3", "ع"], ["5", "خ"], ["6", "ط"], ["7", "ح"], ["8", "ه"], ["9", "ق"]];
  const state = { tab: "text", ocrLines: [], last: null };

  // ---------- API ----------
  async function api(path, body) {
    const r = await fetch(path, body ? {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    } : undefined);
    const data = await r.json().catch(() => ({ error: "Réponse illisible du serveur." }));
    if (!r.ok || data.error) throw new Error(data.error || `Erreur ${r.status}`);
    return data;
  }

  function showError(msg) {
    const e = $("error");
    e.textContent = msg || "";
    e.hidden = !msg;
  }

  // ---------- état du modèle ----------
  async function loadInfo() {
    const m = $("mode");
    try {
      const info = await api("/api/info");
      if (info.mode === "neural") {
        const a = info.attention_model;
        m.className = "mode neural";
        m.textContent = `Réseau chargé : ${a.run} (${a.attention ? "avec attention" : "sans attention"}${a.bidirectional ? ", encodeur bidirectionnel" : ""})`;
      } else {
        m.className = "mode demo";
        m.textContent = "Mode démo : aucun modèle entraîné trouvé, la traduction utilise le dictionnaire et des règles.";
      }
      renderTiles(info.alphabet);
    } catch (err) {
      m.className = "mode demo";
      m.textContent = "Serveur injoignable : lancez python -m src.server.";
      renderTiles(null);
    }
  }

  // ---------- onglets ----------
  const tabs = [$("tab-text"), $("tab-shot")];
  function selectTab(name, focus) {
    state.tab = name;
    tabs.forEach((t) => {
      const on = t.id === `tab-${name}`;
      t.setAttribute("aria-selected", on);
      t.tabIndex = on ? 0 : -1;
      $(t.getAttribute("aria-controls")).hidden = !on;
      if (on && focus) t.focus();
    });
  }
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => selectTab(t.id.slice(4)));
    t.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const n = tabs[(i + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length];
        selectTab(n.id.slice(4), true);
      }
    });
  });

  // ---------- saisie ----------
  const input = $("input");
  const updateCounter = () => { $("counter").textContent = `${input.value.length}/160`; };
  input.addEventListener("input", updateCounter);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
  });
  document.querySelectorAll(".ex").forEach((b) => b.addEventListener("click", () => {
    input.value = b.textContent;
    updateCounter();
    selectTab("text");
    run();
  }));

  // ---------- capture d'écran ----------
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) readImage(f); });
  $("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) readImage(f); });
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) { e.preventDefault(); selectTab("shot"); readImage(item.getAsFile()); }
  });
  $("shot-reset").addEventListener("click", () => {
    $("shot").hidden = true; drop.hidden = false; $("file").value = ""; state.ocrLines = [];
  });

  function readImage(file) {
    if (!file.type.startsWith("image/")) return showError("Ce fichier n'est pas une image. Choisissez une capture PNG ou JPG.");
    if (file.size > 12 * 1024 * 1024) return showError("Image trop lourde : 12 Mo maximum.");
    showError("");
    const reader = new FileReader();
    reader.onload = async () => {
      drop.hidden = true;
      $("shot").hidden = false;
      $("shot-img").src = reader.result;
      $("ocr-lines").replaceChildren();
      $("shot-boxes").replaceChildren();
      $("ocr-status").textContent = "Lecture de l'image…";
      try {
        const res = await api("/api/ocr", { image: reader.result });
        state.ocrLines = res.lines;
        renderOcr(res);
      } catch (err) {
        $("ocr-status").textContent = "";
        showError(err.message);
      }
    };
    reader.readAsDataURL(file);
  }

  function renderOcr(res) {
    const list = $("ocr-lines");
    const svg = $("shot-boxes");
    svg.setAttribute("viewBox", `0 0 ${res.width} ${res.height}`);
    svg.setAttribute("preserveAspectRatio", "none");
    if (!res.lines.length) {
      $("ocr-status").textContent = "Aucun texte trouvé. Essayez une capture plus nette, ou recopiez le message dans l'onglet Écrire.";
      return;
    }
    $("ocr-status").textContent = "Lignes lues. Corrigez-les si besoin et décochez celles à ignorer.";
    res.lines.forEach((l, i) => {
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      const [x0, y0, x1, y1] = l.box;
      Object.entries({ x: x0 - 4, y: y0 - 4, width: x1 - x0 + 8, height: y1 - y0 + 8, rx: 6, fill: "none",
        "stroke-width": Math.max(2, res.width / 220) }).forEach(([k, v]) => rect.setAttribute(k, v));
      svg.append(rect);

      const li = el("li");
      const cb = el("input"); cb.type = "checkbox"; cb.checked = l.keep;
      cb.setAttribute("aria-label", `Garder la ligne ${i + 1}`);
      const tx = el("input"); tx.type = "text"; tx.value = l.text; tx.spellcheck = false;
      tx.setAttribute("aria-label", `Texte de la ligne ${i + 1}`);
      const sync = () => {
        l.keep = cb.checked; l.text = tx.value;
        li.classList.toggle("off", !cb.checked);
        rect.setAttribute("stroke", cb.checked ? "#1D5C9E" : "#9AA7AE");
        rect.setAttribute("stroke-dasharray", cb.checked ? "" : "6 5");
      };
      cb.addEventListener("change", sync); tx.addEventListener("input", sync);
      li.append(cb, tx);
      if (l.reason) li.append(el("span", "why", `Ignorée par défaut : ${l.reason}`));
      list.append(li);
      sync();
    });
  }

  // ---------- translittération ----------
  $("go").addEventListener("click", run);
  async function run() {
    showError("");
    const text = state.tab === "text"
      ? input.value
      : state.ocrLines.filter((l) => l.keep && l.text.trim()).map((l) => l.text).join("\n");
    if (!text.trim()) {
      return showError(state.tab === "text"
        ? "Écrivez au moins un mot en Arabizi, par exemple 3aslema."
        : "Déposez une capture d'écran et gardez au moins une ligne.");
    }
    const go = $("go");
    go.disabled = true; go.textContent = "Écriture…";
    const ring = startRing();
    try {
      // la requête part en même temps que la sonnerie ; on affiche le résultat quand la sonnerie est finie
      const [res] = await Promise.all([
        api("/api/transliterate", { text, beam: $("beam").checked ? 5 : 0, compare: $("compare").checked }),
        ring,
      ]);
      state.last = res;
      renderOutput(res);
      renderHeatOptions(res);
    } catch (err) {
      showError(err.message);
    } finally {
      stopRing();
      go.disabled = false; go.textContent = "Écrire en arabe";
    }
  }

  // ---------- sonnerie du Nokia : l'extrait du milieu de la Nokia Tune, vibration et secousse ----------
  const RING_MS = 3350;                              // durée de l'extrait : « tiri riri tiri riri tiri riri ten »
  const tune = new Audio("sound/nokia-tune-mid.mp3");
  tune.preload = "auto";
  let ringTimer = null;

  function startRing() {
    stopRing();
    const nokia = $("nokia");
    tune.currentTime = 0;
    tune.volume = 1;
    tune.play().catch(() => {});                       // si le navigateur bloque le son, on continue sans
    if (!reduceMotion) nokia.classList.add("ringing");
    if (navigator.vibrate && !reduceMotion) navigator.vibrate(Array.from({ length: 21 }, (_, i) => (i % 2 ? 60 : 100)));
    return new Promise((resolve) => {
      // fondu sur les 250 dernières millisecondes, puis arrêt net
      const fade = setTimeout(() => {
        const t0 = performance.now();
        const step = () => {
          const k = Math.min(1, (performance.now() - t0) / 250);
          tune.volume = 1 - k;
          if (k < 1) requestAnimationFrame(step);
        };
        step();
      }, RING_MS - 250);
      ringTimer = setTimeout(() => { clearTimeout(fade); stopRing(); resolve(); }, RING_MS);
    });
  }

  function stopRing() {
    clearTimeout(ringTimer);
    tune.pause();
    $("nokia").classList.remove("ringing");
    if (navigator.vibrate) navigator.vibrate(0);
  }

  // le français reste en lettres latines : on l'isole pour le styler sans casser le sens d'écriture
  function arabicLine(text, cls) {
    const p = el("p", cls);
    const re = /[A-Za-zÀ-ÿœ'’-]+(?:[ ,.!?]+[A-Za-zÀ-ÿœ'’-]+)*/g;  // une suite de mots français = un seul bloc LTR
    let last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) p.append(document.createTextNode(text.slice(last, m.index)));
      const s = el("span", "lat", m[0]); s.dir = "ltr"; p.append(s);
      last = m.index + m[0].length;
    }
    if (last < text.length) p.append(document.createTextNode(text.slice(last)));
    return p;
  }

  function renderOutput(res) {
    const out = $("output");
    out.replaceChildren();
    res.lines.forEach((l, i) => {
      const p = arabicLine(l.output, "line");
      p.lang = "ar";
      if (!reduceMotion) { p.classList.add("reveal"); p.style.animationDelay = `${i * 120}ms`; }
      out.append(p);
      if (l.baseline !== undefined) {
        const b = arabicLine(l.baseline, "base");
        const tag = el("small", "", ` (${l.baseline_name})`);
        b.append(tag);
        out.append(b);
      }
    });
    $("copy").hidden = false;
  }

  $("copy").addEventListener("click", async () => {
    if (!state.last) return;
    const txt = state.last.lines.map((l) => l.output).join("\n");
    try { await navigator.clipboard.writeText(txt); $("copy").textContent = "Texte copié"; }
    catch { $("copy").textContent = "Copie impossible : sélectionnez le texte"; }
    setTimeout(() => { $("copy").textContent = "Copier le texte arabe"; }, 1800);
  });

  // ---------- carte d'attention ----------
  function renderHeatOptions(res) {
    const sel = $("chunk-select");
    sel.replaceChildren();
    const items = [];
    res.lines.forEach((l) => l.chunks.forEach((c) => { if (c.att) items.push(c); }));
    if (!items.length) {
      $("look-controls").hidden = true;
      $("heat").replaceChildren();
      $("heat-empty").hidden = false;
      $("heat-empty").textContent = res.mode === "demo"
        ? "La carte d'attention apparaît quand un modèle avec attention est chargé. Le mode démo n'en a pas."
        : "Ce modèle n'a pas d'attention : il résume toute l'entrée dans un seul vecteur, il n'y a donc rien à afficher.";
      return;
    }
    items.forEach((c, i) => { const o = el("option", "", c.src); o.value = i; sel.append(o); });
    sel.onchange = () => drawHeat(items[+sel.value]);
    $("look-controls").hidden = false;
    $("heat-empty").hidden = true;
    drawHeat(items[0]);
  }

  function drawHeat(c) {
    const heat = $("heat");
    heat.replaceChildren();
    const src = [...c.src], out = [...c.out];
    heat.style.gridTemplateColumns = `26px repeat(${src.length}, 26px)`;
    heat.setAttribute("role", "img");
    heat.setAttribute("aria-label", `Carte d'attention : ${out.length} lettres arabes sur ${src.length} caractères Arabizi`);
    const show = (ch) => (ch === " " ? "␣" : ch);
    heat.append(el("span"));
    const xl = src.map((ch) => { const s = el("span", "xl", show(ch)); heat.append(s); return s; });
    out.forEach((oc, t) => {
      const yl = el("span", "yl", show(oc)); heat.append(yl);
      const row = c.att[t] || [];
      src.forEach((sc, i) => {
        const v = row[i] || 0;
        const cell = el("span", "c");
        cell.style.opacity = (0.06 + 0.94 * v).toFixed(3);
        cell.addEventListener("mouseenter", () => {
          cell.classList.add("hot"); xl[i].classList.add("hot"); yl.classList.add("hot");
          const pair = $("pair");
          pair.replaceChildren(document.createTextNode("Pour écrire "), el("b", "", show(oc)),
            document.createTextNode(" le modèle regarde "), el("code", "", show(sc)),
            document.createTextNode(` à ${Math.round(v * 100)} %`));
        });
        cell.addEventListener("mouseleave", () => {
          cell.classList.remove("hot"); xl[i].classList.remove("hot"); yl.classList.remove("hot");
        });
        heat.append(cell);
      });
    });
  }

  // ---------- alphabet ----------
  function renderTiles(learned) {
    const ul = $("tiles");
    ul.replaceChildren();
    const fromModel = learned && Object.keys(learned).length;
    if (fromModel) {
      $("alpha-note").textContent = "Ce que le modèle a retrouvé seul : pour chaque chiffre, la lettre arabe qu'il écrit le plus souvent quand son attention se pose sur ce chiffre (mesuré sur le jeu de test).";
    }
    const rows = fromModel
      ? Object.entries(learned).map(([d, list]) => {
          const tot = list.reduce((s, [, n]) => s + n, 0);
          return [d, list[0][0], `${Math.round((100 * list[0][1]) / tot)} % des cas`];
        })
      : REFERENCE.map(([d, l]) => [d, l, "table usuelle"]);
    rows.forEach(([d, l, cap]) => {
      const li = el("li", "tile");
      const pr = el("span", "pairing");
      pr.append(el("span", "d", d), el("span", "arrow", "="), el("span", "l", l));
      pr.lastChild.lang = "ar";
      li.append(pr, el("small", "", cap));
      ul.append(li);
    });
  }

  // heure sur l'iPhone
  const tick = () => { $("ios-time").textContent = new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" }); };
  tick(); setInterval(tick, 20000);

  updateCounter();
  loadInfo();
})();
