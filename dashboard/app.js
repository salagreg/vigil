const detectionList = document.getElementById("detection-list");
const todayCountEl = document.getElementById("today-count");
const bannerEl = document.getElementById("status-banner");
const bannerTextEl = document.getElementById("status-banner-text");
const radarBlipsEl = document.getElementById("radar-blips");
const radarDefsEl = document.getElementById("radar-defs");
const cameraToggleEl = document.getElementById("camera-toggle");
const historyFiltersEl = document.getElementById("history-filters");
const historyCountEl = document.getElementById("history-count");
const historySortEl = document.getElementById("history-sort");
const cloudToggleEl = document.getElementById("cloud-toggle");
const SVG_NS = "http://www.w3.org/2000/svg";

const MAX_ITEMS = 100;
const LEVEL_PRIORITY = { CRITIQUE: 0, HAUT: 1, MOYEN: 2, BAS: 3 };
let todayCount = 0;
let currentRange = "all";
let currentSort = "time";
let currentEvents = [];
const knownGradientIds = new Set();

// Meme formule que cote serveur (detector.py, _color_for_id) : angle dore,
// deux ids consecutifs ont toujours des teintes bien differentes. Permet
// d'associer visuellement le meme rocher entre video, radar et historique.
function hueForId(objectId) {
  return (objectId * 137.508) % 360;
}

function gradientCss(objectId) {
  const hue = hueForId(objectId);
  return `linear-gradient(135deg, hsl(${hue}, 85%, 68%), hsl(${hue}, 85%, 38%))`;
}

function ensureRadarGradient(objectId) {
  const gradientId = `blip-grad-${objectId}`;
  if (!knownGradientIds.has(gradientId)) {
    const hue = hueForId(objectId);
    const gradient = document.createElementNS(SVG_NS, "radialGradient");
    gradient.setAttribute("id", gradientId);

    const stopCenter = document.createElementNS(SVG_NS, "stop");
    stopCenter.setAttribute("offset", "0%");
    stopCenter.setAttribute("stop-color", `hsl(${hue}, 90%, 78%)`);

    const stopEdge = document.createElementNS(SVG_NS, "stop");
    stopEdge.setAttribute("offset", "100%");
    stopEdge.setAttribute("stop-color", `hsl(${hue}, 90%, 42%)`);

    gradient.appendChild(stopCenter);
    gradient.appendChild(stopEdge);
    radarDefsEl.appendChild(gradient);
    knownGradientIds.add(gradientId);
  }
  return `url(#${gradientId})`;
}

function isToday(isoTs) {
  const d = new Date(isoTs);
  const now = new Date();
  return (
    d.getUTCFullYear() === now.getUTCFullYear() &&
    d.getUTCMonth() === now.getUTCMonth() &&
    d.getUTCDate() === now.getUTCDate()
  );
}

function formatTime(isoTs) {
  const d = new Date(isoTs);
  return d.toLocaleTimeString();
}

function levelBadgeClass(level) {
  return `level-badge level-badge--${(level || "").toLowerCase()}`;
}

function buildDetectionLi(detection) {
  const li = document.createElement("li");
  li.style.borderColor = `hsl(${hueForId(detection.object_id)}, 85%, 55%)`;

  let thumb;
  if (detection.snapshot) {
    thumb = document.createElement("img");
    thumb.className = "thumb";
    thumb.src = `/snapshots/${detection.snapshot}`;
    thumb.alt = `${detection.label} #${detection.object_id}`;
    thumb.style.borderColor = `hsl(${hueForId(detection.object_id)}, 85%, 55%)`;
  } else {
    thumb = document.createElement("span");
    thumb.className = "swatch";
    thumb.style.background = gradientCss(detection.object_id);
  }

  const level = document.createElement("span");
  level.className = levelBadgeClass(detection.level);
  level.textContent = detection.level || "?";

  const label = document.createElement("span");
  label.className = "label";
  label.textContent = `${detection.label} #${detection.object_id} - ${detection.zone} (${Math.round(
    detection.confidence * 100
  )}%)`;

  const meta = document.createElement("span");
  meta.className = "meta";
  const durationText = detection.duration_seconds ? ` - ${Math.round(detection.duration_seconds)}s` : "";
  meta.textContent = `${formatTime(detection.ts)}${durationText}`;

  li.appendChild(thumb);
  li.appendChild(level);
  li.appendChild(label);
  li.appendChild(meta);
  return li;
}

function sortedEvents() {
  const events = [...currentEvents];
  if (currentSort === "level") {
    events.sort((a, b) => {
      const pa = LEVEL_PRIORITY[a.level] ?? 99;
      const pb = LEVEL_PRIORITY[b.level] ?? 99;
      if (pa !== pb) return pa - pb;
      return new Date(b.ts) - new Date(a.ts);
    });
  }
  // "time" : /events renvoie deja du plus recent au plus ancien, et les
  // nouvelles detections live sont ajoutees en tete (voir plus bas).
  return events;
}

function renderDetectionList() {
  detectionList.innerHTML = "";
  for (const detection of sortedEvents()) {
    detectionList.appendChild(buildDetectionLi(detection));
  }
}

function addLiveDetection(detection) {
  currentEvents.unshift(detection);
  if (currentEvents.length > MAX_ITEMS) {
    currentEvents.length = MAX_ITEMS;
  }
  renderDetectionList();
}

function rangeLabel(range) {
  if (range === "1h") return "derniere heure";
  if (range === "24h") return "dernieres 24h";
  return "tout l'historique";
}

function sinceForRange(range) {
  const now = Date.now();
  if (range === "1h") return new Date(now - 60 * 60 * 1000).toISOString();
  if (range === "24h") return new Date(now - 24 * 60 * 60 * 1000).toISOString();
  return null;
}

function updateHistoryCount() {
  const suffix = currentEvents.length > 1 ? "s" : "";
  historyCountEl.textContent = `${currentEvents.length} detection${suffix} - ${rangeLabel(currentRange)}`;
}

async function loadHistoryForRange(range) {
  currentRange = range;
  try {
    const since = sinceForRange(range);
    const url = since ? `/events?since=${encodeURIComponent(since)}` : "/events";
    const res = await fetch(url);
    currentEvents = await res.json();
    renderDetectionList();
    updateHistoryCount();
  } catch (err) {
    console.error("Impossible de charger l'historique des detections", err);
  }
}

async function loadHistory() {
  // Calcule le compteur "aujourd'hui" une seule fois au demarrage, a partir
  // de l'historique complet - independant du filtre affiche ensuite.
  try {
    const res = await fetch("/events");
    const events = await res.json();
    for (const event of events) {
      if (isToday(event.ts)) {
        todayCount += 1;
      }
    }
    todayCountEl.textContent = todayCount;
  } catch (err) {
    console.error("Impossible de calculer le total du jour", err);
  }

  await loadHistoryForRange(currentRange);
}

historyFiltersEl.addEventListener("click", (event) => {
  const button = event.target.closest(".filter-btn");
  if (!button) return;

  historyFiltersEl.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("active"));
  button.classList.add("active");
  loadHistoryForRange(button.dataset.range);
});

historySortEl.addEventListener("click", (event) => {
  const button = event.target.closest(".filter-btn");
  if (!button) return;

  historySortEl.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("active"));
  button.classList.add("active");
  currentSort = button.dataset.sort;
  renderDetectionList();
});

// Web Audio API : pas de fichier son a heberger, tout est synthetise.
// Les navigateurs bloquent l'audio automatique tant qu'il n'y a pas eu
// d'interaction utilisateur : on cree/debloque le contexte au premier
// clic ou touche pressee sur la page.
let audioCtx = null;

function unlockAudio() {
  if (audioCtx) return;
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  } catch (err) {
    console.error("Audio non supporte par ce navigateur", err);
  }
}

document.addEventListener("click", unlockAudio, { once: true });
document.addEventListener("keydown", unlockAudio, { once: true });

function playAlertSound() {
  if (!audioCtx) return;
  const now = audioCtx.currentTime;

  // Deux notes courtes et montantes ("bip-bip") : plus reconnaissable comme
  // alarme qu'un simple bip unique, sans etre agressif en demo.
  [0, 0.18].forEach((offset, i) => {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "square";
    osc.frequency.value = i === 0 ? 740 : 988;

    gain.gain.setValueAtTime(0.0001, now + offset);
    gain.gain.exponentialRampToValueAtTime(0.15, now + offset + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.15);

    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start(now + offset);
    osc.stop(now + offset + 0.16);
  });
}

function setIdleBanner() {
  bannerEl.className = "status-banner status-banner--idle";
  bannerEl.style.borderColor = "";
  bannerTextEl.textContent = "SECTEUR SURVEILLE - RAS";
}

// Pilote directement par le flux radar (objets actuellement dans le champ) :
// le bandeau reste en alerte en continu tant qu'au moins un rocher est
// visible, et revient a RAS des qu'il n'y en a plus aucun - plus de
// minuterie qui masque une menace toujours presente.
//
// Delai de grace avant de repasser a RAS : le radar est reconstruit a partir
// de zero a chaque detection (nouvel id = nouvelle couleur), donc il peut y
// avoir un tout petit trou (une ou deux diffusions radar, ~200-400ms) entre
// la perte de l'ancien id et la confirmation du nouveau. Sans ce delai, le
// bandeau clignoterait sur RAS a chaque micro-coupure au lieu de rester en
// alerte comme demande.
const RAS_GRACE_MS = 1500;
let rasGraceTimer = null;

function renderPresenceBanner(objects) {
  if (!objects || objects.length === 0) {
    if (rasGraceTimer === null) {
      rasGraceTimer = setTimeout(() => {
        rasGraceTimer = null;
        setIdleBanner();
      }, RAS_GRACE_MS);
    }
    return;
  }

  clearTimeout(rasGraceTimer);
  rasGraceTimer = null;

  const worst = [...objects].sort((a, b) => {
    const pa = LEVEL_PRIORITY[a.level] ?? 99;
    const pb = LEVEL_PRIORITY[b.level] ?? 99;
    return pa - pb;
  })[0];

  const extra = objects.length > 1 ? ` (+${objects.length - 1} autre${objects.length > 2 ? "s" : ""})` : "";

  bannerEl.className = "status-banner status-banner--alert";
  bannerEl.style.borderColor = `hsl(${hueForId(worst.object_id)}, 85%, 55%)`;
  bannerTextEl.textContent = `OBJET DETECTE : ${worst.label.toUpperCase()} #${worst.object_id} - ZONE ${
    worst.zone
  } [${worst.level}]${extra}`;
}

// rel_x = 0 (bord gauche de l'image) a 1 (bord droit) -> angle de -70 a +70
// degres, 0.5 (centre de l'image) = tout droit (en haut du radar).
function renderRadar(objects) {
  radarBlipsEl.innerHTML = "";

  for (const obj of objects) {
    const angleDeg = (obj.rel_x - 0.5) * 140;
    const angleRad = (angleDeg * Math.PI) / 180;
    // Plus l'objet est gros dans l'image, plus il est proche : rayon plus petit.
    const radius = Math.max(12, 85 - Math.min(obj.rel_size * 250, 70));

    const x = 100 + radius * Math.sin(angleRad);
    const y = 100 - radius * Math.cos(angleRad);

    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "radar-blip");

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", x);
    circle.setAttribute("cy", y);
    circle.setAttribute("r", 6);
    circle.setAttribute("fill", ensureRadarGradient(obj.object_id));

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", x);
    text.setAttribute("y", y - 8);
    text.textContent = `#${obj.object_id}`;

    group.appendChild(circle);
    group.appendChild(text);
    radarBlipsEl.appendChild(group);
  }
}

function renderCameraButton(connected) {
  cameraToggleEl.textContent = connected ? "Deconnecter la camera" : "Connecter la camera";
  cameraToggleEl.classList.toggle("camera-toggle--off", !connected);
  cameraToggleEl.dataset.connected = connected ? "1" : "0";
}

async function loadCameraStatus() {
  try {
    const res = await fetch("/camera/status");
    const data = await res.json();
    renderCameraButton(data.connected);
  } catch (err) {
    console.error("Impossible de recuperer l'etat de la camera", err);
  }
}

async function toggleCamera() {
  const currentlyConnected = cameraToggleEl.dataset.connected === "1";
  const endpoint = currentlyConnected ? "/camera/disconnect" : "/camera/connect";

  cameraToggleEl.disabled = true;
  try {
    const res = await fetch(endpoint, { method: "POST" });
    const data = await res.json();
    renderCameraButton(data.connected);
  } catch (err) {
    console.error("Impossible de changer l'etat de la camera", err);
  } finally {
    cameraToggleEl.disabled = false;
  }
}

cameraToggleEl.addEventListener("click", toggleCamera);

function renderCloudPill(statusData) {
  if (statusData.autonomous) {
    const elapsed = Math.round(statusData.elapsed_seconds || 0);
    cloudToggleEl.textContent = `MODE AUTONOME - ${elapsed}s`;
    cloudToggleEl.classList.add("cloud-pill--autonomous");
  } else {
    cloudToggleEl.textContent = "CLOUD : JOIGNABLE";
    cloudToggleEl.classList.remove("cloud-pill--autonomous");
  }
  cloudToggleEl.dataset.enabled = statusData.enabled ? "1" : "0";
}

async function loadCloudStatus() {
  try {
    const res = await fetch("/cloud/status");
    renderCloudPill(await res.json());
  } catch (err) {
    console.error("Impossible de recuperer l'etat du cloud", err);
  }
}

async function toggleCloud() {
  cloudToggleEl.disabled = true;
  try {
    const res = await fetch("/cloud/toggle", { method: "POST" });
    renderCloudPill(await res.json());
  } catch (err) {
    console.error("Impossible de changer l'etat du cloud", err);
  } finally {
    cloudToggleEl.disabled = false;
  }
}

cloudToggleEl.addEventListener("click", toggleCloud);

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

  ws.onclose = () => {
    setTimeout(connectWebSocket, 2000);
  };
  ws.onerror = () => ws.close();

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "detection") {
      addLiveDetection(data);
      playAlertSound();
      updateHistoryCount();
      if (isToday(data.ts)) {
        todayCount += 1;
        todayCountEl.textContent = todayCount;
      }
    } else if (data.type === "radar") {
      renderRadar(data.objects);
      renderPresenceBanner(data.objects);
    } else if (data.type === "cloud_status") {
      renderCloudPill(data);
    }
    // Le heartbeat sert seulement a confirmer que la connexion est vivante.
  };
}

loadHistory().then(connectWebSocket);
loadCameraStatus();
loadCloudStatus();
