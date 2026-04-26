const STORAGE_KEYS = {
  theme: "raven.dashboard.theme",
  autoplay: "raven.dashboard.autoplay",
};

const state = {
  busy: false,
  pollTimer: null,
  latestAudioRevisionPlayed: 0,
  playingAudio: false,
  audioUnlockNotified: false,
  autoplay: localStorage.getItem(STORAGE_KEYS.autoplay) !== "off",
  activeMemoryTab: "notes",
};

const els = {
  html: document.documentElement,
  themeToggle: document.getElementById("themeToggle"),
  autoplayToggle: document.getElementById("autoplayToggle"),
  ravenAvatar: document.getElementById("ravenAvatar"),
  waveform: document.getElementById("waveform"),
  botName: document.getElementById("botName"),
  statusLine: document.getElementById("statusLine"),
  listenTitle: document.getElementById("listenTitle"),
  currentStatus: document.getElementById("currentStatus"),
  statusDetail: document.getElementById("statusDetail"),
  composerState: document.getElementById("composerState"),
  signalScore: document.getElementById("signalScore"),
  modelName: document.getElementById("modelName"),
  voiceName: document.getElementById("voiceName"),
  audioMode: document.getElementById("audioMode"),
  metricsGrid: document.getElementById("metricsGrid"),
  currentUserText: document.getElementById("currentUserText"),
  currentAssistantText: document.getElementById("currentAssistantText"),
  errorText: document.getElementById("errorText"),
  memoryContent: document.getElementById("memoryContent"),
  notesTabButton: document.getElementById("notesTabButton"),
  remindersTabButton: document.getElementById("remindersTabButton"),
  chatForm: document.getElementById("chatForm"),
  promptInput: document.getElementById("promptInput"),
  sendButton: document.getElementById("sendButton"),
  copyButton: document.getElementById("copyButton"),
  promptChips: document.querySelectorAll("[data-prompt]"),
};

const AVATAR_STATE_CLASSES = ["state-idle", "state-warming", "state-thinking", "state-processing", "state-generating", "state-speaking", "state-error"];

const microcopy = {
  idle: ["Raven is poised and waiting.", "Quiet channel, clean signal.", "Ready for your next thought."],
  warming: ["Warming the voice chain.", "Preparing timing and tone.", "Priming response layers."],
  thinking: ["Analyzing intent.", "Mapping the response path.", "Shaping the first draft."],
  processing: ["Composing output.", "Sorting context.", "Refining answer structure."],
  generating: ["Generating response.", "Assembling final phrasing.", "Rendering voice-ready text."],
  speaking: ["Raven is speaking.", "Audio stream in progress.", "Delivering output now."],
  error: ["Voice orchestration ready.", "Awaiting your next prompt.", "System recovered and ready."],
  joke: ["Hunting the perfect comma.", "Pretending not to judge your typos.", "Consulting dramatic pause module."],
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function text(el, value) {
  if (el) {
    el.textContent = value;
  }
}

function html(el, value) {
  if (el) {
    el.innerHTML = value;
  }
}

function normalizeStatus(snapshot) {
  const rawStatus = String(snapshot.current_status || "idle").toLowerCase();
  const avatarState = String(snapshot.avatar_state || "").toLowerCase();
  if (rawStatus === "error") {
    return "error";
  }
  if (rawStatus === "warming" || snapshot.warmup_state === "warming") {
    return "warming";
  }
  if (avatarState === "speaking" || rawStatus === "speaking") {
    return "speaking";
  }
  if (snapshot.busy && snapshot.current_assistant_text) {
    return "generating";
  }
  if (snapshot.busy) {
    return rawStatus === "thinking" ? "thinking" : "processing";
  }
  return rawStatus || "idle";
}

function statusMessage(status, turnId = 0) {
  const bucket = microcopy[status] || microcopy.idle;
  if (status === "idle" && turnId > 0 && turnId % 3 === 0) {
    return microcopy.joke[(turnId / 3) % microcopy.joke.length];
  }
  return bucket[Math.abs(Number(turnId) || 0) % bucket.length];
}

function initializeTheme() {
  const stored = localStorage.getItem(STORAGE_KEYS.theme);
  const theme = stored === "dark" ? "dark" : "light";
  els.html.dataset.theme = theme;
  updateThemeButton(theme);
}

function updateThemeButton(theme) {
  text(els.themeToggle, theme === "dark" ? "Light" : "Dark");
  els.themeToggle?.classList.toggle("is-active", theme === "dark");
}

function updateAutoplayButton() {
  text(els.autoplayToggle, state.autoplay ? "Audio on" : "Audio off");
  els.autoplayToggle?.classList.toggle("is-active", state.autoplay);
}

function signalScore(status, snapshot) {
  if (status === "error") {
    return 37;
  }
  if (status === "speaking") {
    return 95;
  }
  if (status === "generating") {
    return 88;
  }
  if (status === "thinking" || status === "processing" || status === "warming") {
    return 74;
  }
  const total = snapshot.metrics?.total_turn_latency_ms || 0;
  return total > 0 ? Math.max(61, Math.min(99, 100 - Math.round(total / 900))) : 100;
}

function renderMetrics(snapshot) {
  const metrics = snapshot.metrics || {};
  const items = [
    ["Turn", `${Math.round(snapshot.turn_elapsed_ms || 0)}`],
    ["First token", `${Math.round(metrics.first_token_latency_ms || 0)}`],
    ["First audio", `${Math.round(metrics.first_audio_latency_ms || 0)}`],
    ["Total", `${Math.round(metrics.total_turn_latency_ms || 0)}`],
  ];
  html(els.metricsGrid, items.map(([label, value]) => `
    <article class="metric-chip">
      <span class="metric-label">${escapeHtml(label)}</span>
      <strong class="metric-value">${escapeHtml(value)}</strong>
    </article>
  `).join(""));
}

function formatTime(isoStamp) {
  const parsed = new Date(isoStamp);
  if (Number.isNaN(parsed.getTime())) {
    return "";
  }
  return parsed.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function renderMemory(snapshot) {
  const active = state.activeMemoryTab;
  const items = active === "reminders" ? (snapshot.reminders || []) : (snapshot.notes || []);

  els.notesTabButton.classList.toggle("is-active", active === "notes");
  els.remindersTabButton.classList.toggle("is-active", active === "reminders");
  els.notesTabButton.setAttribute("aria-selected", String(active === "notes"));
  els.remindersTabButton.setAttribute("aria-selected", String(active === "reminders"));

  if (!items.length) {
    const label = active === "notes" ? "notes" : "reminders";
    html(els.memoryContent, `<div class="empty-state">No ${label} yet. Ask Raven to save one.</div>`);
    return;
  }

  html(els.memoryContent, items.map((item) => `
    <article class="memory-item">
      <div class="memory-row">
        <span class="memory-pill">${escapeHtml(item.kind)}</span>
        <span class="memory-meta">${escapeHtml(formatTime(item.created_at))}</span>
      </div>
      <p class="memory-text">${escapeHtml(item.summary || "")}</p>
      ${item.due_hint ? `<span class="memory-meta">due: ${escapeHtml(item.due_hint)}</span>` : ""}
    </article>
  `).join(""));
}

function renderSnapshot(snapshot) {
  state.busy = !!snapshot.busy;
  state.activeMemoryTab = snapshot.active_memory_tab || state.activeMemoryTab;
  const status = normalizeStatus(snapshot);
  const message = statusMessage(status === "error" ? "idle" : status, snapshot.current_turn_id);
  const assistantText = snapshot.current_assistant_text || "";

  text(els.botName, snapshot.bot_name || "Raven");
  text(els.modelName, snapshot.model_name || "-");
  text(els.voiceName, snapshot.voice_name || "-");
  text(els.audioMode, snapshot.audio_mode || "-");
  text(els.currentStatus, status);
  text(els.statusLine, message);
  text(
    els.statusDetail,
    status === "error"
      ? "Last turn failed. You can send a new prompt any time."
      : state.busy
        ? "Live response in progress."
        : "System settled. Waiting for a prompt."
  );
  text(
    els.listenTitle,
    status === "speaking"
      ? "Raven is speaking"
      : status === "idle"
        ? "Raven is listening"
        : status === "warming"
          ? "Raven is warming up"
          : status === "error"
            ? "Raven hit an error"
            : "Raven is processing"
  );
  text(
    els.composerState,
    status === "error" ? "Error" : status === "warming" ? "Warming" : status === "speaking" ? "Speaking" : state.busy ? "Working" : "Ready"
  );
  text(els.signalScore, signalScore(status, snapshot));
  text(els.currentUserText, snapshot.current_user_text || "No active prompt.");
  text(els.currentAssistantText, assistantText || "Waiting for the next response.");

  els.currentStatus.className = `status-chip status-${status}`;
  els.ravenAvatar.classList.remove(...AVATAR_STATE_CLASSES);
  els.ravenAvatar.classList.add(`state-${status}`);
  els.waveform.classList.toggle("is-active", state.busy || status === "speaking" || status === "generating");
  els.sendButton.disabled = state.busy;
  els.promptInput.disabled = state.busy;

  if (snapshot.latest_error) {
    text(els.errorText, snapshot.latest_error);
    els.errorText.classList.remove("hidden");
  } else {
    text(els.errorText, "");
    els.errorText.classList.add("hidden");
  }

  renderMetrics(snapshot);
  renderMemory(snapshot);
  maybePlayLatestAudio(snapshot).catch(() => {
    // Browsers can block playback until the user interacts with the page.
  });
}

async function maybePlayLatestAudio(snapshot) {
  if (!state.autoplay) {
    return;
  }

  const revision = Number(snapshot.latest_audio_revision || 0);
  if (!snapshot.latest_audio_available || revision <= state.latestAudioRevisionPlayed || state.playingAudio) {
    return;
  }

  state.playingAudio = true;
  try {
    const response = await fetch(`/api/audio/latest?rev=${revision}`, { cache: "no-store" });
    if (!response.ok) {
      return;
    }
    const objectUrl = URL.createObjectURL(await response.blob());
    const audio = new Audio(objectUrl);
    try {
      await audio.play();
      await new Promise((resolve) => {
        audio.onended = resolve;
        audio.onerror = resolve;
      });
      state.latestAudioRevisionPlayed = revision;
    } catch {
      if (!state.audioUnlockNotified) {
        text(els.errorText, "Click once anywhere on the dashboard to enable audio playback.");
        els.errorText.classList.remove("hidden");
        state.audioUnlockNotified = true;
      }
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
  } finally {
    state.playingAudio = false;
  }
}

async function fetchState() {
  const response = await fetch("/api/state", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`State request failed with ${response.status}`);
  }
  renderSnapshot(await response.json());
}

async function sendPrompt(prompt) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: prompt }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: "Request failed." }));
    throw new Error(body.detail || "Request failed.");
  }
  renderSnapshot(await response.json());
}

async function setMemoryTab(tab) {
  const response = await fetch("/api/memory/tab", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tab }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: "Failed to change memory tab." }));
    throw new Error(body.detail || "Failed to change memory tab.");
  }
  const payload = await response.json();
  state.activeMemoryTab = payload.active_memory_tab || tab;
  await fetchState();
}

function startPolling() {
  if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
  }
  state.pollTimer = window.setInterval(() => {
    fetchState().catch((error) => {
      text(els.errorText, error.message);
      els.errorText.classList.remove("hidden");
    });
  }, 500);
}

els.themeToggle.addEventListener("click", () => {
  const nextTheme = els.html.dataset.theme === "dark" ? "light" : "dark";
  els.html.dataset.theme = nextTheme;
  localStorage.setItem(STORAGE_KEYS.theme, nextTheme);
  updateThemeButton(nextTheme);
});

els.autoplayToggle.addEventListener("click", () => {
  state.autoplay = !state.autoplay;
  localStorage.setItem(STORAGE_KEYS.autoplay, state.autoplay ? "on" : "off");
  updateAutoplayButton();
});

els.notesTabButton.addEventListener("click", async () => {
  if (state.activeMemoryTab === "notes") {
    return;
  }
  try {
    await setMemoryTab("notes");
  } catch (error) {
    text(els.errorText, error.message);
    els.errorText.classList.remove("hidden");
  }
});

els.remindersTabButton.addEventListener("click", async () => {
  if (state.activeMemoryTab === "reminders") {
    return;
  }
  try {
    await setMemoryTab("reminders");
  } catch (error) {
    text(els.errorText, error.message);
    els.errorText.classList.remove("hidden");
  }
});

els.copyButton.addEventListener("click", async () => {
  const value = (els.currentAssistantText.textContent || "").trim();
  if (!value || value === "Waiting for the next response.") {
    return;
  }
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
  }
  text(els.copyButton, "Copied");
  window.setTimeout(() => text(els.copyButton, "Copy"), 1200);
});

els.promptChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    els.promptInput.value = chip.dataset.prompt || "";
    els.promptInput.focus();
  });
});

els.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = els.promptInput.value.trim();
  if (!prompt || state.busy) {
    return;
  }
  els.errorText.classList.add("hidden");
  try {
    await sendPrompt(prompt);
    els.promptInput.value = "";
  } catch (error) {
    text(els.errorText, error.message);
    els.errorText.classList.remove("hidden");
  }
});

initializeTheme();
updateAutoplayButton();
fetchState().catch((error) => {
  text(els.errorText, error.message);
  els.errorText.classList.remove("hidden");
});
startPolling();
