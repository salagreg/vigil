const detectionList = document.getElementById("detection-list");
const todayCountEl = document.getElementById("today-count");
const statusEl = document.getElementById("connection-status");
const bannerEl = document.getElementById("status-banner");
const bannerTextEl = document.getElementById("status-banner-text");
const radarBlipsEl = document.getElementById("radar-blips");

const MAX_ITEMS = 100;
const IDLE_DELAY_MS = 6000; // temps sans nouvelle detection avant de revenir a "RAS"
let todayCount = 0;
let idleTimer = null;

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

function addDetectionToList(detection, { prepend = true } = {}) {
  const li = document.createElement("li");

  const label = document.createElement("span");
  label.className = "label";
  label.textContent = `${detection.label} #${detection.object_id} - ${detection.zone} (${Math.round(
    detection.confidence * 100
  )}%)`;

  const meta = document.createElement("span");
  meta.className = "meta";
  meta.textContent = formatTime(detection.ts);

  li.appendChild(label);
  li.appendChild(meta);

  if (prepend) {
    detectionList.prepend(li);
    while (detectionList.children.length > MAX_ITEMS) {
      detectionList.removeChild(detectionList.lastChild);
    }
  } else {
    detectionList.appendChild(li);
  }
}

async function loadHistory() {
  try {
    const res = await fetch("/events");
    const events = await res.json();
    // /events renvoie deja du plus recent au plus ancien.
    for (const event of events) {
      addDetectionToList(event, { prepend: false });
      if (isToday(event.ts)) {
        todayCount += 1;
      }
    }
    todayCountEl.textContent = todayCount;
  } catch (err) {
    console.error("Impossible de charger l'historique des detections", err);
  }
}

function setIdleBanner() {
  bannerEl.className = "status-banner status-banner--idle";
  bannerTextEl.textContent = "SECTEUR SURVEILLE - RAS";
}

function setDetectionBanner(detection) {
  clearTimeout(idleTimer);

  bannerEl.className = "status-banner status-banner--alert";
  bannerTextEl.textContent = `OBJET DETECTE : ${detection.label.toUpperCase()} #${detection.object_id} - ZONE ${
    detection.zone
  } (${Math.round(detection.confidence * 100)}%)`;

  idleTimer = setTimeout(setIdleBanner, IDLE_DELAY_MS);
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
    circle.setAttribute("r", 5);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", x);
    text.setAttribute("y", y - 8);
    text.textContent = `#${obj.object_id}`;

    group.appendChild(circle);
    group.appendChild(text);
    radarBlipsEl.appendChild(group);
  }
}

function setStatus(online) {
  statusEl.textContent = online ? "connecte" : "deconnecte";
  statusEl.classList.toggle("status--online", online);
  statusEl.classList.toggle("status--offline", !online);
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

  ws.onopen = () => setStatus(true);
  ws.onclose = () => {
    setStatus(false);
    setTimeout(connectWebSocket, 2000);
  };
  ws.onerror = () => ws.close();

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "detection") {
      addDetectionToList(data);
      setDetectionBanner(data);
      if (isToday(data.ts)) {
        todayCount += 1;
        todayCountEl.textContent = todayCount;
      }
    } else if (data.type === "radar") {
      renderRadar(data.objects);
    }
    // Le heartbeat sert seulement a confirmer que la connexion est vivante.
  };
}

loadHistory().then(connectWebSocket);
