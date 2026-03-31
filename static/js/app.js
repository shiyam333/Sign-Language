/* ================================================================
   SignBridge — app.js
   All translator logic + shared utilities (toast, etc.)
   ================================================================ */

"use strict";

/* ─── Shared Utilities ─── */

function showToast(msg, type) {
  type = type || 'info';
  var wrap = document.getElementById('toastWrap');
  if (!wrap) return;
  var el = document.createElement('div');
  var icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
  el.className = 'toast t-' + type;
  el.innerHTML = '<span>' + icon + '</span><span>' + msg + '</span>';
  wrap.appendChild(el);
  setTimeout(function () {
    el.style.transition = 'all 0.28s ease';
    el.style.opacity = '0';
    el.style.transform = 'translateX(110%)';
    setTimeout(function () { el.remove(); }, 300);
  }, 3000);
}

/* ─── Translator State ─── */
var camStream      = null;
var isRunning      = false;
var processing     = false;
var frameHistory   = [];
var lastSpoken     = null;
var sentence       = '';
var signHistory    = [];

var HIST_SIZE   = 5;
var STABILITY   = 0.70;
var FRAME_MS    = 33;
var CONF_THRESH = 0.70;

/* ─── Translator Boot ─── */
document.addEventListener('DOMContentLoaded', function () {
  if (!document.getElementById('startBtn')) return;  // not on translator page
  setupTranslator();
});

function setupTranslator() {
  document.getElementById('startBtn').addEventListener('click', startCamera);
  document.getElementById('stopBtn').addEventListener('click', stopCamera);
  document.getElementById('clearBtn').addEventListener('click', clearAll);
  document.getElementById('speakBtn').addEventListener('click', speakSentence);
  document.getElementById('copyBtn').addEventListener('click', copySentence);
  document.getElementById('downloadBtn').addEventListener('click', downloadText);
  document.getElementById('languageSelect').addEventListener('change', onLangChange);
}

async function startCamera() {
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false
    });
    var vid = document.getElementById('webcam');
    vid.srcObject = camStream;
    await new Promise(function (res) {
      vid.onloadedmetadata = function () { vid.play(); res(); };
    });
    isRunning = true;
    setStatus('live');
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled  = false;
    var ph = document.getElementById('placeholder');
    if (ph) ph.style.display = 'none';
    showToast('Camera started — show a sign! 🤟', 'success');
    frameLoop();
  } catch (e) {
    showToast('Camera error: ' + e.message, 'error');
    setStatus('error');
  }
}

function stopCamera() {
  isRunning = false;
  if (camStream) camStream.getTracks().forEach(function (t) { t.stop(); });
  camStream = null;
  setStatus('idle');
  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled  = true;
  var overlay = document.getElementById('camOverlay');
  if (overlay) overlay.style.display = 'none';
}

function setStatus(state) {
  var dot  = document.getElementById('statusDot');
  var text = document.getElementById('statusText');
  if (!dot || !text) return;
  dot.className = 'dot';
  if (state === 'live')  { dot.classList.add('live');  text.textContent = 'Live'; }
  else if (state === 'error') { dot.classList.add('error'); text.textContent = 'Error'; }
  else text.textContent = 'Idle';
}

async function frameLoop() {
  if (!isRunning) return;
  if (!processing) {
    processing = true;
    await captureFrame();
    processing = false;
  }
  setTimeout(frameLoop, FRAME_MS);
}

async function captureFrame() {
  var vid    = document.getElementById('webcam');
  var canvas = document.getElementById('canvas');
  if (!vid || !canvas) return;
  var ctx = canvas.getContext('2d', { alpha: false });
  canvas.width  = vid.videoWidth  || 640;
  canvas.height = vid.videoHeight || 480;
  ctx.drawImage(vid, 0, 0);
  var img = canvas.toDataURL('image/jpeg', 0.65);
  try {
    var resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: img })
    });
    if (resp.ok) {
      var data = await resp.json();
      if (data.success) handlePrediction(data.gesture, data.confidence, data.frame);
    }
  } catch (_) {}
}

function handlePrediction(gesture, conf, frameB64) {
  /* Update overlay on video */
  var overlay    = document.getElementById('camOverlay');
  var overlayW   = document.getElementById('overlayWord');
  var overlayC   = document.getElementById('overlayConf');
  if (overlay) {
    if (gesture !== 'None' && conf > CONF_THRESH) {
      overlay.style.display = 'flex';
      if (overlayW) overlayW.textContent = gesture;
      if (overlayC) overlayC.textContent = Math.round(conf * 100) + '% confidence';
    } else {
      overlay.style.display = 'none';
    }
  }

  /* Update gesture panel */
  var gEl  = document.getElementById('gesture');
  var cBar = document.getElementById('confBar');
  var cVal = document.getElementById('confPct');
  var chip = document.getElementById('gestureChip');
  if (gEl)  gEl.textContent  = gesture !== 'None' ? gesture : '—';
  if (cBar) cBar.style.width = Math.round(conf * 100) + '%';
  if (cVal) cVal.textContent = Math.round(conf * 100) + '%';
  if (chip) {
    chip.textContent = conf > CONF_THRESH ? 'Detected' : 'Waiting';
    chip.className   = 'chip ' + (conf > CONF_THRESH ? 'chip-teal' : 'chip-amber');
  }
  if (gEl && gesture !== 'None' && conf > CONF_THRESH) {
    gEl.classList.add('pop-anim');
    setTimeout(function () { gEl.classList.remove('pop-anim'); }, 300);
  }

  if (conf < CONF_THRESH || gesture === 'None') return;

  frameHistory.push(gesture);
  if (frameHistory.length > HIST_SIZE) frameHistory.shift();

  if (frameHistory.length === HIST_SIZE) {
    var most  = getMostCommon(frameHistory);
    var freq  = frameHistory.filter(function (g) { return g === most; }).length;
    if ((freq / HIST_SIZE) >= STABILITY && most !== lastSpoken) {
      addWord(most);
      lastSpoken = most;
      frameHistory = [];
    }
  }
}

function getMostCommon(arr) {
  var counts = {};
  var max = 0, best = arr[0];
  arr.forEach(function (v) {
    counts[v] = (counts[v] || 0) + 1;
    if (counts[v] > max) { max = counts[v]; best = v; }
  });
  return best;
}

function addWord(word) {
  var lang = getLang();
  if (word === 'space')     { sentence += ' '; }
  else if (word === 'del')  { sentence = sentence.slice(0, -1); }
  else if (word !== 'None') {
    sentence += word + ' ';
    translateAndSpeak(word, lang);
    pushHistory(word);
  }
  renderSentence();
}

function getLang() {
  var sel = document.getElementById('languageSelect');
  return sel ? sel.value : 'en';
}

function clearAll() {
  sentence = '';
  frameHistory = [];
  lastSpoken = null;
  renderSentence();
  showToast('Cleared', 'info');
}

function speakSentence() {
  var lang = getLang();
  var text = sentence.trim();
  if (!text) { showToast('Nothing to speak yet', 'info'); return; }
  var translatedEl = document.getElementById('translatedText');
  var toSpeak = (lang !== 'en' && translatedEl && translatedEl.textContent)
    ? translatedEl.textContent : text;
  fetch('/api/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: toSpeak, lang: lang })
  });
  showToast('Speaking…', 'success');
}

function copySentence() {
  if (!sentence.trim()) { showToast('Nothing to copy', 'info'); return; }
  navigator.clipboard.writeText(sentence.trim())
    .then(function () { showToast('Copied!', 'success'); })
    .catch(function () { showToast('Copy failed', 'error'); });
}

function downloadText() {
  if (!sentence.trim()) { showToast('Nothing to save', 'info'); return; }
  var blob = new Blob([sentence], { type: 'text/plain' });
  var url  = URL.createObjectURL(blob);
  var a    = document.createElement('a');
  a.href   = url;
  a.download = 'signbridge_' + new Date().toISOString().slice(0, 10) + '.txt';
  a.click();
  URL.revokeObjectURL(url);
  showToast('Saved!', 'success');
}

function onLangChange() {
  var lang = getLang();
  var sec  = document.getElementById('transSection');
  if (sec) sec.style.display = lang === 'en' ? 'none' : 'block';
  if (lang !== 'en' && sentence.trim()) doTranslation();
}

function renderSentence() {
  var el = document.getElementById('sentence');
  if (!el) return;
  el.textContent = sentence || '[Waiting for gestures…]';
  el.className   = sentence ? 'sentence-box' : 'sentence-box empty';
  el.scrollTop   = el.scrollHeight;
  var lang = getLang();
  if (lang !== 'en' && sentence.trim()) doTranslation();
}

async function doTranslation() {
  var lang = getLang();
  if (lang === 'en' || !sentence.trim()) return;
  try {
    var resp = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: sentence.trim(), lang: lang })
    });
    if (resp.ok) {
      var data = await resp.json();
      var el = document.getElementById('translatedText');
      if (el) el.textContent = data.translated || sentence;
    }
  } catch (_) {}
}

async function translateAndSpeak(word, lang) {
  if (lang === 'en') {
    doSpeak(word, lang);
    return;
  }
  try {
    var resp = await fetch('/api/translate-word', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word: word, lang: lang })
    });
    if (resp.ok) {
      var data = await resp.json();
      doSpeak(data.translated || word, lang);
    } else doSpeak(word, lang);
  } catch (_) { doSpeak(word, lang); }
}

function doSpeak(text, lang) {
  fetch('/api/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: text, lang: lang })
  }).catch(function () {});
}

function pushHistory(word) {
  var now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  signHistory.unshift({ word: word, time: now });
  if (signHistory.length > 30) signHistory.pop();
  renderHistory();
}

function renderHistory() {
  var el = document.getElementById('historyList');
  if (!el) return;
  if (!signHistory.length) {
    el.innerHTML = '<p class="hist-empty">No signs detected yet</p>';
    return;
  }
  el.innerHTML = signHistory.slice(0, 12).map(function (h) {
    return '<div class="hist-row">'
      + '<span class="hist-word">' + h.word + '</span>'
      + '<span class="hist-time">' + h.time + '</span>'
      + '</div>';
  }).join('');
}

function clearHistory() {
  signHistory = [];
  renderHistory();
}