"""The listening page `scripts/listen_ab.py` serves: WebAudio playback that switches at the same playhead.

Until 2026-10-12 every Play created a new `Audio(url)`, so each cut restarted from zero and the
listener compared A and B from memory across 10 s. The page now decodes the trial's cuts into
WebAudio buffers and plays them through one `Transport`:

- switching (keys 1, 2, 3 or the play buttons) keeps the playhead: the new buffer starts at the
  position the old one had reached, on a sample boundary, with an 8 ms linear crossfade
  (`FADE_S`), scheduled 30 ms ahead (`LEAD_S`) so both ramps start on the same sample. Linear,
  not equal-power: the stimuli are aligned and nearly identical, and two linear ramps sum to
  exactly 1, so switching between identical buffers plays the buffer unchanged (the browser
  check renders that in an `OfflineAudioContext`: the residual is zero);
- space plays and pauses; the slider seeks; a loop of 1, 2 or 3 s from the playhead (key L, or
  the selector) or over the quietest 2 s of the reference cut ("Loop the pause", where pause and
  hiss questions are answered) repeats while the listener switches;
- the threshold run's trials loop their one-second interval on their own (`autoloop`); key L,
  which would end that loop, does nothing there, and the help line leaves it out;
- practice trials show the answer afterwards; test trials never do, and catch trials look like
  any other trial.

The `Transport` takes any `BaseAudioContext`, so the same code runs in an `OfflineAudioContext`
for the sample-accuracy probe. No test runner for JavaScript exists in the repository; the
switching is checked in a browser on synthetic audio (docs/validation.md).
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listening test</title>
<style>
:root { color-scheme: light dark; --bg: #f7f7f5; --fg: #1b1b1b; --muted: #666; --accent: #2d5bd1; --card: #fff; --line: #d6d6d0; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #151515; --fg: #ececec; --muted: #a8a8a8; --accent: #86a8ff; --card: #222; --line: #383838; }
}
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }
main { max-width: 640px; margin: 0 auto; padding: 24px 16px; }
.row { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin: 16px 0; }
button, select { font: inherit; padding: 12px 20px; min-width: 80px; border-radius: 8px; cursor: pointer;
  border: 1px solid var(--line); background: var(--card); color: var(--fg); }
select { padding: 8px 12px; min-width: 0; }
button:disabled { opacity: 0.5; cursor: default; }
button.playing { box-shadow: inset 0 0 0 3px var(--accent); }
button.selected { border-color: var(--accent); }
button.choice { background: var(--accent); border-color: var(--accent); color: var(--bg); }
button.choice.aside { background: var(--card); color: var(--fg); border-color: var(--line); }
kbd { font: 12px ui-monospace, monospace; border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; margin-left: 8px; }
input[type=range] { flex: 1 1 200px; min-width: 0; }
#progress, #note, #help, #clock, #loopinfo { color: var(--muted); }
#feedback { font-weight: 600; min-height: 1.5em; }
</style>
</head>
<body>
<main>
<h1>Listening test</h1>
<p id="progress"></p>
<p id="feedback"></p>
<p id="prompt"></p>
<div class="row" id="play"></div>
<div class="row" id="transport">
<button id="toggle" type="button">Play<kbd>space</kbd></button>
<input id="seek" type="range" min="0" max="1" step="0.01" value="0" aria-label="Playhead">
<span id="clock">0.0 s</span>
</div>
<div class="row" id="loops">
<label>Loop <select id="looplen" aria-label="Loop length">
<option value="0">off</option><option value="1">1 s</option><option value="2">2 s</option><option value="3">3 s</option>
</select></label>
<button id="pauseloop" type="button" hidden>Loop the pause</button>
<span id="loopinfo"></span>
</div>
<div class="row" id="choices"></div>
<p id="note"></p>
<p id="help"></p>
</main>
<script>
"use strict";
const FADE_S = 0.008;
const LEAD_S = 0.03;
const LABELS = { cant_tell: "Can't tell" };
const HELP = {
  cut: "Keys: 1, 2 and 3 switch to that sound at the same moment of the cut; space plays and pauses; L loops 2 s from here.",
  loop: "Keys: 1, 2 and 3 switch to that interval at the same moment of its second, which loops on its own; space plays and pauses.",
};
const byId = (id) => document.getElementById(id);

class Transport {
  constructor(context, output) {
    this.ctx = context;
    this.out = output || context.destination;
    this.buffers = {};
    this.duration = 0;
    this.current = null;
    this.voice = null;
    this.anchor = { time: 0, offset: 0 };
    this.paused = 0;
    this.loop = null;
    this.onchange = () => {};
  }
  get playing() { return this.voice !== null; }
  get ready() { return this.duration > 0; }
  setBuffers(buffers) {
    this.stop();
    this.buffers = buffers;
    const durations = Object.values(buffers).map((buffer) => buffer.duration);
    this.duration = durations.length ? Math.min(...durations) : 0;
    this.current = null;
    this.loop = null;
    this.paused = 0;
    this.onchange();
  }
  snap(seconds) { return Math.round(seconds * this.ctx.sampleRate) / this.ctx.sampleRate; }
  later(seconds) { return Math.ceil(seconds * this.ctx.sampleRate) / this.ctx.sampleRate; }
  when() { return Math.max(this.later(this.ctx.currentTime + LEAD_S), this.later(this.anchor.time + FADE_S)); }
  wrap(position) {
    const loop = this.loop;
    if (loop && position >= loop[1]) return loop[0] + ((position - loop[0]) % (loop[1] - loop[0]));
    return position;
  }
  position(at) {
    if (!this.playing) return this.paused;
    return this.wrap(this.anchor.offset + (at - this.anchor.time));
  }
  inside(position) {
    if (this.loop) return position >= this.loop[0] && position < this.loop[1] ? position : this.loop[0];
    return position < this.duration ? position : 0;
  }
  voiceAt(name, at, offset) {
    const source = this.ctx.createBufferSource();
    const gain = this.ctx.createGain();
    source.buffer = this.buffers[name];
    if (this.loop) {
      source.loop = true;
      source.loopStart = this.loop[0];
      source.loopEnd = this.loop[1];
    }
    gain.gain.setValueAtTime(0, at);
    gain.gain.linearRampToValueAtTime(1, at + FADE_S);
    source.connect(gain);
    gain.connect(this.out);
    source.onended = () => { if (this.voice && this.voice.source === source) this.finish(); };
    source.start(at, offset);
    return { source, gain };
  }
  release(voice, at) {
    voice.source.onended = null;
    voice.gain.gain.cancelScheduledValues(at);
    voice.gain.gain.setValueAtTime(1, at);
    voice.gain.gain.linearRampToValueAtTime(0, at + FADE_S);
    voice.source.stop(at + FADE_S);
  }
  start(name, at, offset) {
    if (this.voice) this.release(this.voice, at);
    this.voice = this.voiceAt(name, at, offset);
    this.anchor = { time: at, offset: offset };
    this.current = name;
    this.onchange();
  }
  play(name, at) {
    const time = at === undefined ? this.when() : at;
    this.start(name, time, this.snap(this.inside(this.position(time))));
  }
  pause(at) {
    if (!this.voice) return;
    const time = at === undefined ? this.when() : at;
    this.paused = this.position(time);
    this.release(this.voice, time);
    this.voice = null;
    this.onchange();
  }
  seek(position) {
    const target = this.snap(Math.min(Math.max(position, 0), this.duration));
    if (!this.playing) {
      this.paused = target;
      this.onchange();
      return;
    }
    this.start(this.current, this.when(), this.inside(target));
  }
  setLoop(region) {
    const time = this.when();
    const position = this.position(time);
    this.loop = region ? [this.snap(region[0]), this.snap(Math.min(region[1], this.duration))] : null;
    if (this.playing) {
      this.start(this.current, time, this.snap(this.inside(position)));
      return;
    }
    this.paused = this.inside(position);
    this.onchange();
  }
  stop() {
    if (this.voice) {
      this.voice.source.onended = null;
      this.voice.source.stop();
    }
    this.voice = null;
    this.paused = 0;
    this.onchange();
  }
  finish() {
    this.voice = null;
    this.paused = 0;
    this.onchange();
  }
}

const audio = new AudioContext();
const transport = new Transport(audio);
let state = null;
let loadToken = 0;
let busy = false;

function wake() { if (audio.state === "suspended") audio.resume(); }

function button(text, cls, action, key) {
  const el = document.createElement("button");
  el.type = "button";
  el.textContent = text;
  if (key) {
    const hint = document.createElement("kbd");
    hint.textContent = key;
    el.appendChild(hint);
  }
  if (cls) el.className = cls;
  el.onclick = () => { el.blur(); action(el); };
  return el;
}

function paint() {
  document.querySelectorAll("#play button").forEach((el) => {
    const mine = el.dataset.name === transport.current;
    el.classList.toggle("playing", mine && transport.playing);
    el.classList.toggle("selected", mine && !transport.playing);
  });
  byId("toggle").firstChild.textContent = transport.playing ? "Pause" : "Play";
  const loop = transport.loop;
  byId("loopinfo").textContent = loop ? "looping " + loop[0].toFixed(2) + "-" + loop[1].toFixed(2) + " s" : "";
}
transport.onchange = paint;

function select(name) {
  wake();
  if (transport.ready) transport.play(name);
}

function toggle() {
  wake();
  if (!transport.ready) return;
  if (transport.playing) {
    transport.pause();
    return;
  }
  transport.play(transport.current || state.buttons[0].name);
}

function loopFromHere(seconds) {
  if (!transport.ready) return;
  if (!seconds) {
    transport.setLoop(null);
    return;
  }
  const length = Math.min(seconds, transport.duration);
  const start = Math.min(transport.position(audio.currentTime), transport.duration - length);
  transport.setLoop([start, start + length]);
}

function toggleLoop() {
  const picker = byId("looplen");
  picker.value = transport.loop ? "0" : "2";
  loopFromHere(Number(picker.value));
}

async function loadBuffers(next) {
  const token = ++loadToken;
  const decoded = {};
  byId("note").textContent = "Loading the sounds...";
  try {
    for (const b of next.buttons) {
      const response = await fetch(b.url, { cache: "no-store" });
      decoded[b.name] = await audio.decodeAudioData(await response.arrayBuffer());
    }
  } catch (error) {
    byId("note").textContent = "The sounds did not load: " + error;
    return;
  }
  if (token !== loadToken) return;
  transport.setBuffers(decoded);
  byId("seek").max = transport.duration.toFixed(3);
  byId("note").textContent = "";
  document.querySelectorAll("#play button").forEach((el) => { el.disabled = false; });
  if (next.autoloop) transport.setLoop(next.loops[next.autoloop]);
}

function render(next) {
  state = next;
  transport.stop();
  transport.setBuffers({});
  byId("play").textContent = "";
  byId("choices").textContent = "";
  const playing = !next.done;
  ["transport", "loops", "help"].forEach((id) => { byId(id).hidden = !playing; });
  if (next.done) {
    byId("progress").textContent = "Finished.";
    byId("prompt").textContent = next.summary;
    return;
  }
  const practice = next.phase === "practice";
  byId("progress").textContent = (practice ? "Practice " : "Trial ") + next.number + " of " + next.of +
    (practice ? ": the answer is shown afterwards." : "");
  byId("prompt").textContent = next.prompt;
  next.buttons.forEach((b, index) => {
    const el = button("Play " + b.name, "", () => select(b.name), String(index + 1));
    el.dataset.name = b.name;
    el.disabled = true;
    byId("play").appendChild(el);
  });
  next.choices.forEach((c) => {
    byId("choices").appendChild(button(LABELS[c] || c, c === "cant_tell" ? "choice aside" : "choice", () => answer(c)));
  });
  byId("loops").hidden = Boolean(next.autoloop);
  byId("help").textContent = next.autoloop ? HELP.loop : HELP.cut;
  byId("looplen").value = "0";
  byId("pauseloop").hidden = !(next.loops && next.loops.pause);
  loadBuffers(next);
}

async function load() {
  const response = await fetch("/api/state", { cache: "no-store" });
  render(await response.json());
}

async function answer(choice) {
  if (busy) return;
  busy = true;
  transport.pause();
  const response = await fetch("/api/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trial: state.index, choice: choice }),
  });
  busy = false;
  if (!response.ok) {
    byId("note").textContent = await response.text();
    return load();
  }
  const next = await response.json();
  byId("feedback").textContent = next.feedback || "";
  render(next);
}

function tick() {
  if (transport.ready) {
    const position = transport.position(audio.currentTime);
    byId("seek").value = position.toFixed(3);
    byId("clock").textContent = position.toFixed(1) + " / " + transport.duration.toFixed(1) + " s";
  }
  requestAnimationFrame(tick);
}

byId("toggle").onclick = (event) => { event.currentTarget.blur(); toggle(); };
byId("seek").oninput = (event) => { wake(); transport.seek(Number(event.target.value)); };
byId("looplen").onchange = (event) => { wake(); loopFromHere(Number(event.target.value)); };
byId("pauseloop").onclick = (event) => { event.currentTarget.blur(); wake(); transport.setLoop(state.loops.pause); };
document.addEventListener("keydown", (event) => {
  if (!state || state.done || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key === " ") {
    event.preventDefault();
    if (!event.repeat) toggle();
    return;
  }
  const slot = ["1", "2", "3"].indexOf(event.key);
  if (slot >= 0 && slot < state.buttons.length) {
    event.preventDefault();
    select(state.buttons[slot].name);
    return;
  }
  if ((event.key === "l" || event.key === "L") && !state.autoloop) {
    event.preventDefault();
    wake();
    toggleLoop();
  }
});
document.addEventListener("keyup", (event) => { if (event.key === " ") event.preventDefault(); });
load();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""
