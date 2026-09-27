// Secure MOM — interfața pentru medici. Fără biblioteci externe (funcționează offline).
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const ICON = {
  upload: '<path d="M12 15V4M7 9l5-5 5 5"/><path d="M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/>',
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/>',
  file: '<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
  mail: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/>',
  lock: '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  check: '<path d="m5 12 5 5 9-10"/>',
  alert: '<circle cx="12" cy="12" r="9"/><path d="M12 7v6M12 16.5v.5"/>',
  chev: '<path d="m9 6 6 6-6 6"/>',
  medical: '<path d="M3 12h4l2-5 4 10 2-5h6"/>',
  executive: '<rect x="3" y="7" width="18" height="13" rx="2"/><path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M3 13h18"/>',
  administrative: '<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M9 7h1M14 7h1M9 11h1M14 11h1M9 15h1M14 15h1"/>',
};
const svg = (name, size = 20) =>
  `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="2" ` +
  `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON[name] || ""}</svg>`;

// ponderea etapelor în bara de progres (timpi măsurați: transcrierea și LLM-ul domină)
const WEIGHTS = { audio: 0.03, asr: 0.45, diarization: 0.04, llm: 0.44, report: 0.02, delivery: 0.02 };
const STATUS = { queued: "în așteptare", running: "în lucru", error: "eroare" };
const MONTHS = ["ian", "feb", "mar", "apr", "mai", "iun", "iul", "aug", "sep", "oct", "nov", "dec"];
const START_LABEL = `Generează procesul-verbal ${svg("arrow", 18)}`;

let CFG = null, view = null, blob = null, blobName = null, meetingType = "medical", docLang = "ro";
let currentJob = null, lastJob = null, resultLang = null, pollTimer = null;
const prog = { shown: 0, target: 0, ceil: 0, done: false, start: 0, timer: null };

// ------------------------------------------------------------------ pornire
async function init() {
  $("#drop-ic").innerHTML = svg("upload", 26);
  $("#rec-ic").innerHTML = svg("mic", 26);
  $("#clock-ic").innerHTML = svg("clock", 15);
  $("#dl-ic").innerHTML = svg("download", 17);
  $("#mail-ic").innerHTML = svg("mail", 17);
  $("#lock-ic").innerHTML = svg("lock", 14);
  $("#ok-ic").innerHTML = svg("check", 18);
  $("#start").innerHTML = START_LABEL;

  CFG = await (await fetch("/api/config")).json();
  $("#types").innerHTML = Object.entries(CFG.meeting_types).map(([k, v]) =>
    `<button type="button" data-k="${k}">${svg(k, 17)}<span>${esc(v.ro)}</span></button>`).join("");
  $$("#types button").forEach((b) => (b.onclick = () => selectType(b.dataset.k)));
  selectType(CFG.meeting_types.medical ? "medical" : Object.keys(CFG.meeting_types)[0]);

  docLang = CFG.default_language;
  $("#langs").innerHTML = Object.entries(CFG.languages).map(([k, v]) => `<button type="button" data-l="${k}">${esc(v)}</button>`).join("");
  $$("#langs button").forEach((b) => (b.onclick = () => selectLang(b.dataset.l)));
  selectLang(docLang);

  const d = new Date();
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
  $("#date").value = d.toISOString().slice(0, 10);

  const legacy = new URLSearchParams(location.search).get("job");
  if (legacy) history.replaceState(null, "", `/#job/${legacy}`);
  window.addEventListener("hashchange", route);
  route();
}

function selectType(k) {
  meetingType = k;
  $$("#types button").forEach((b) => b.classList.toggle("active", b.dataset.k === k));
}
function selectLang(l) {
  docLang = l;
  $$("#langs button").forEach((b) => b.classList.toggle("active", b.dataset.l === l));
}

// ------------------------------------------------------------------ navigare
function route() {
  const h = decodeURIComponent(location.hash.slice(1));
  stopPolling();
  if (h === "istoric") { show("history"); loadHistory(); }
  else if (h.startsWith("job/")) openJob(h.slice(4));
  else show("new");
}

function show(v) {
  if (v === view) return;
  view = v;
  for (const x of ["new", "progress", "result", "history"]) $(`#view-${x}`).hidden = x !== v;
  $$("nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === (v === "history" ? "history" : "new")));
  scrollTo(0, 0);
}

// ------------------------------------------------------------------ fișier / drag & drop
const drop = $("#drop");
["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (ev) => { const f = ev.dataTransfer.files[0]; if (f) setFile(f, f.name); });
drop.addEventListener("click", () => $("#file").click());
$("#file").addEventListener("change", (ev) => { const f = ev.target.files[0]; if (f) setFile(f, f.name); });
for (const t of [drop, $("#rec")]) {
  t.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); t.click(); } });
}

function setFile(f, name) {
  blob = f; blobName = name;
  const chip = $("#file-chip");
  chip.hidden = false;
  chip.innerHTML = `${svg("file", 18)}<span class="name">${esc(name)}</span><span class="size">${(f.size / 1048576).toFixed(1)} MB</span>` +
    `<button type="button" title="Elimină fișierul" aria-label="Elimină fișierul">×</button>`;
  chip.querySelector("button").onclick = clearFile;
  $("#start").disabled = false;
}
function clearFile() {
  blob = null; blobName = null; $("#file").value = "";
  $("#file-chip").hidden = true;
  $("#start").disabled = true;
}

// ------------------------------------------------------------------ înregistrare
let rec = null, recStart = 0, recTimer = null, audioCtx = null, analyser = null, stream = null;
$("#rec").onclick = async () => {
  if (rec && rec.state === "recording") { rec.stop(); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
  } catch (e) { alert("Microfonul nu este disponibil: " + e.message); return; }
  const chunks = [];
  const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "";
  rec = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    clearInterval(recTimer);
    $("#rec").classList.remove("on");
    $("#rec-ic").innerHTML = svg("mic", 26);
    $("#rec-label").textContent = "Înregistrați ședința";
    $("#rec-sub").textContent = "Direct de la microfonul calculatorului";
    $("#meter").hidden = true;
    const type = rec.mimeType || "audio/webm";
    const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
    const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
    setFile(new Blob(chunks, { type }), `sedinta-${stamp}.${ext}`);
    if (audioCtx) audioCtx.close();
  };
  rec.start(1000);
  recStart = Date.now();
  $("#rec").classList.add("on");
  $("#rec-ic").innerHTML = svg("stop", 24);
  $("#rec-label").textContent = "Opriți înregistrarea";
  $("#rec-sub").textContent = "00:00";
  $("#meter").hidden = false;
  recTimer = setInterval(() => { $("#rec-sub").textContent = mmss((Date.now() - recStart) / 1000); }, 500);
  audioCtx = new AudioContext(); analyser = audioCtx.createAnalyser(); analyser.fftSize = 1024;
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  drawMeter();
};

function drawMeter() {
  const c = $("#meter"), g = c.getContext("2d"), data = new Uint8Array(analyser.fftSize);
  const bars = 40, hist = new Array(bars).fill(0);
  let last = 0;
  (function loop(ts) {
    if (!rec || rec.state !== "recording") { g.clearRect(0, 0, c.width, c.height); return; }
    if (ts - last > 70) {
      last = ts;
      analyser.getByteTimeDomainData(data);
      let sum = 0; for (const v of data) sum += (v - 128) ** 2;
      hist.shift(); hist.push(Math.min(1, Math.sqrt(sum / data.length) / 40));
      g.clearRect(0, 0, c.width, c.height);
      const w = c.width / bars;
      hist.forEach((v, i) => {
        const h = Math.max(4, v * c.height);
        g.fillStyle = `rgba(217,59,59,${0.35 + 0.65 * (i / bars)})`;
        g.beginPath(); g.roundRect(i * w + 2, (c.height - h) / 2, w - 4, h, 3); g.fill();
      });
    }
    requestAnimationFrame(loop);
  })(0);
}

// ------------------------------------------------------------------ trimiterea spre procesare
$("#start").onclick = () => {
  if (!blob) return;
  const b = $("#start");
  b.disabled = true;
  b.innerHTML = `<i class="spin"></i> Se încarcă…`;
  const fd = new FormData();
  fd.append("file", blob, blobName);
  fd.append("meeting_type", meetingType);
  fd.append("output_language", docLang);
  fd.append("meeting_date", $("#date").value);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) b.innerHTML = `<i class="spin"></i> Se încarcă… ${Math.round((e.loaded / e.total) * 100)}%`;
  };
  const done = () => { b.innerHTML = START_LABEL; b.disabled = !blob; };
  xhr.onload = () => {
    let d = {};
    try { d = JSON.parse(xhr.responseText); } catch { /* răspuns gol */ }
    if (xhr.status >= 200 && xhr.status < 300 && d.id) { clearFile(); done(); location.hash = `job/${d.id}`; }
    else { done(); alert("Nu s-a putut încărca: " + (d.detail || xhr.statusText)); }
  };
  xhr.onerror = () => { done(); alert("Serverul intern nu răspunde."); };
  xhr.send(fd);
};

// ------------------------------------------------------------------ procesare: o singură bară 0–100%
function openJob(id) {
  currentJob = id; lastJob = null; resultLang = null;
  Object.assign(prog, { shown: 0, target: 0, ceil: 0, done: false, start: 0 });
  poll(true);
  pollTimer = setInterval(poll, 1500);
}

function stopPolling() {
  clearInterval(pollTimer); pollTimer = null;
  clearInterval(prog.timer); prog.timer = null;
}

async function poll(first = false) {
  const id = currentJob;
  let j;
  try {
    const r = await fetch(`/api/jobs/${id}`);
    if (!r.ok) throw new Error(String(r.status));
    j = await r.json();
  } catch {
    if (first) location.hash = "";
    return;
  }
  if (id !== currentJob) return;
  if (j.status === "done") {
    clearInterval(pollTimer); pollTimer = null;
    if (first || view !== "progress") return showResult(j);
    prog.done = true; prog.target = 1;
    setTimeout(() => { if (currentJob === id) showResult(j); }, 1000);
    return;
  }
  show("progress");
  renderProgress(j);
}

function overall(j) {
  let done = 0, ceil = 0;
  for (const [k, w] of Object.entries(WEIGHTS)) {
    const s = (j.stages || {})[k] || {};
    if (s.status === "done" || s.status === "skipped") { done += w; ceil = done; continue; }
    if (s.status === "running" || s.status === "error") { ceil = done + w * 0.92; done += w * (s.progress || 0); }
    break;
  }
  return { target: done, ceil: Math.max(ceil, done) };
}

function renderProgress(j) {
  prog.start = (j.started || j.created) * 1000;
  const o = overall(j);
  prog.target = Math.max(prog.target, o.target);
  prog.ceil = o.ceil;
  if (!prog.timer) prog.timer = setInterval(tickProgress, 100);
  $("#p-kicker").textContent = CFG.meeting_types[j.meeting_type]?.ro || "";
  $("#p-file").textContent = j.source_name || "";
  const failed = j.status === "error";
  $("#view-progress .panel").classList.toggle("failed", failed);
  $("#p-error").hidden = !failed;
  $("#p-back").hidden = !failed;
  if (failed) {
    tickProgress();
    stopPolling();
    $("#p-title").textContent = "Procesarea nu a reușit";
    $("#p-sub").textContent = "Verificați înregistrarea și încercați din nou.";
    $("#p-error").textContent = j.error || "";
    return;
  }
  if (j.status === "queued") {
    $("#p-title").textContent = "În așteptare…";
    $("#p-sub").textContent = j.queue_position ? `Poziția ${j.queue_position} în coadă` : "";
  } else {
    $("#p-title").textContent = "Se pregătește procesul-verbal…";
    $("#p-sub").textContent = j.send_email
      ? "La final se trimite automat pe email. Puteți închide această pagină."
      : "Puteți închide această pagină — documentul rămâne în Istoric.";
  }
}

function tickProgress() {
  const lim = prog.done ? 1 : 0.99;
  if (prog.shown < prog.target) {
    const step = Math.max(0.002, (prog.target - prog.shown) * (prog.done ? 0.25 : 0.12));
    prog.shown = Math.min(prog.target, prog.shown + step);
  } else if (prog.ceil > prog.shown) {
    prog.shown += (prog.ceil - prog.shown) * 0.0022;  // avans lent în etapa curentă, fără să treacă de ea
  }
  prog.shown = Math.min(prog.shown, lim);
  $("#p-pct").textContent = Math.floor(prog.shown * 100 + 1e-6);
  $("#p-bar").style.width = `${(prog.shown * 100).toFixed(2)}%`;
  $("#view-progress .track").setAttribute("aria-valuenow", Math.floor(prog.shown * 100));
  if (prog.start && !prog.done) $("#p-elapsed").textContent = mmss((Date.now() - prog.start) / 1000);
}

// ------------------------------------------------------------------ rezultat
function showResult(j) {
  stopPolling();
  lastJob = j;
  show("result");
  resultLang = resultLang || j.output_language;
  $("#r-meta").textContent = [CFG.meeting_types[j.meeting_type]?.ro || j.meeting_type, fmtDate(j.meeting_date),
    j.duration_s ? durText(j.duration_s) : ""].filter(Boolean).join(" · ");
  renderSent(j);
  $("#r-langs").innerHTML = Object.entries(CFG.languages).map(([k, v]) =>
    `<button type="button" data-l="${k}" title="${esc(v)}" class="${k === resultLang ? "active" : ""}">${k.toUpperCase()}</button>`).join("");
  $$("#r-langs button").forEach((b) => (b.onclick = () => {
    if (b.dataset.l === resultLang) return;
    resultLang = b.dataset.l;
    showResult(lastJob);
  }));
  $("#dl-docx").href = `/api/jobs/${j.id}/download/docx?lang=${resultLang}`;
  loadDoc(j, resultLang);
}

function renderSent(j) {
  const box = $("#r-sent");
  const rows = (j.deliveries || []).slice(-3).map((d) =>
    `<div>${svg("check", 16)}<span>${d.manual ? "Trimis către" : "Trimis automat către"} <b>${esc(d.recipients.join(", "))}</b></span></div>`);
  if (j.delivery_error && !(j.deliveries || []).length) {
    rows.push(`<div class="bad">${svg("alert", 16)}<span>Emailul nu a putut fi trimis automat. Folosiți „Trimite pe email”.</span></div>`);
  }
  box.innerHTML = rows.join("");
  box.hidden = !rows.length;
}

async function loadDoc(j, lang) {
  const f = $("#mom-frame"), ld = $("#doc-loading");
  ld.hidden = false;
  ld.innerHTML = `<i class="spin"></i><span>${lang === j.output_language ? "Se deschide documentul…" : "Se traduce documentul…"}</span>`;
  f.style.visibility = "hidden";
  let d;
  try {
    const r = await fetch(`/api/jobs/${j.id}/mom?lang=${lang}`);
    d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
  } catch (e) {
    if (j.id === currentJob && lang === resultLang) ld.innerHTML = `<span>Documentul nu poate fi afișat: ${esc(e.message)}</span>`;
    return;
  }
  if (j.id !== currentJob || lang !== resultLang) return;
  f.onload = () => { fitDoc(); setTimeout(fitDoc, 300); f.style.visibility = ""; ld.hidden = true; };
  // în aplicație documentul stă direct pe fundalul paginii (în email are fundalul lui)
  f.srcdoc = d.html.replace("</head>", "<style>body{background:transparent!important;padding:2px 12px 30px!important}</style></head>");
}

function fitDoc() {
  const f = $("#mom-frame");
  try {
    const doc = f.contentDocument.documentElement;
    f.style.height = "0px";  // altfel scrollHeight nu scade sub înălțimea curentă
    f.style.height = `${doc.scrollHeight}px`;
  } catch { /* încă se încarcă */ }
}
window.addEventListener("resize", fitDoc);

// ------------------------------------------------------------------ email la cerere
$("#r-send").onclick = () => {
  $("#mail-err").hidden = true;
  $("#mail").showModal();
  $("#mail-to").focus();
};
$("#mail-cancel").onclick = () => $("#mail").close();
$("#mail-form").onsubmit = async (e) => {
  e.preventDefault();
  const b = $("#mail-send");
  b.disabled = true; b.innerHTML = `<i class="spin"></i> Se trimite…`;
  try {
    const r = await fetch(`/api/jobs/${currentJob}/send`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lang: resultLang, recipients: $("#mail-to").value }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Trimiterea a eșuat.");
    $("#mail").close();
    $("#mail-to").value = "";
    (lastJob.deliveries ||= []).push(d);
    renderSent(lastJob);
  } catch (err) {
    $("#mail-err").hidden = false;
    $("#mail-err").textContent = err.message;
  }
  b.disabled = false; b.textContent = "Trimite";
};

// ------------------------------------------------------------------ istoric
async function loadHistory() {
  const box = $("#history");
  let jobs = [];
  try { jobs = await (await fetch("/api/jobs")).json(); } catch { /* server indisponibil */ }
  if (view !== "history") return;
  if (!jobs.length) { box.innerHTML = `<div class="empty">Nicio ședință procesată încă.</div>`; return; }
  box.innerHTML = jobs.map((j) => {
    const [, m, d] = (j.meeting_date || "").split("-");
    const c = j.counts;
    const meta = [j.duration_s ? durText(j.duration_s) : "",
      c ? `${plural(c.decisions, "decizie", "decizii")} · ${plural(c.action_items, "sarcină", "sarcini")}` : ""].filter(Boolean).join(" · ");
    return `<a class="item" href="#job/${encodeURIComponent(j.id)}">
      <div class="date"><b>${d ? +d : ""}</b><span>${m ? MONTHS[+m - 1] : ""}</span></div>
      <div class="body"><div class="t">${esc(j.title || j.source_name)}</div>
        <div class="m"><span class="tag ${esc(j.meeting_type)}">${esc(CFG.meeting_types[j.meeting_type]?.ro || j.meeting_type)}</span>${esc(meta)}</div></div>
      ${j.status !== "done" ? `<span class="st ${esc(j.status)}">${STATUS[j.status] || esc(j.status)}</span>` : ""}
      <span class="chev">${svg("chev", 18)}</span></a>`;
  }).join("");
}

// ------------------------------------------------------------------ utilitare
function mmss(sec) {
  sec = Math.max(0, Math.floor(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return (h ? `${h}:${String(m).padStart(2, "0")}` : String(m).padStart(2, "0")) + `:${String(s).padStart(2, "0")}`;
}
function durText(sec) {
  const m = Math.max(1, Math.round(sec / 60));
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`;
}
function fmtDate(iso) { const p = (iso || "").split("-"); return p.length === 3 ? `${p[2]}.${p[1]}.${p[0]}` : iso || ""; }
function plural(n, one, many) { return `${n} ${n === 1 ? one : many}`; }
function esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]); }

init();
