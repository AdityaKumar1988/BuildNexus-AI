/* ============================================================
   SMARTBUILD AI — utils.js
   Shared helpers, formatters, validators & constants
   Loaded first — no dependencies on other files
   ============================================================ */

/* ============================================================
   CONSTANTS
   ============================================================ */
const SB = {
  API_BASE:    window.location.origin,  // Auto-detects: localhost:3000 OR https://smartbuild-1-2fm7.onrender.com
  ML_BASE:     'http://localhost:5000',
  WS_URL:      window.location.origin,  // Auto-detects production WebSocket URL
  VERSION:     '1.0.0',
  PROJECT_KEY: 'sb_active_project',
  TOKEN_KEY:   'sb_token',

  BUILD_TYPES: ['Residential', 'Commercial', 'Industrial', 'Infrastructure', 'Mixed Use'],
  QUALITY_GRADES: ['Standard', 'Premium', 'Luxury'],
  RISK_LEVELS: {
    LOW:      { max: 30,  label: 'LOW RISK',      color: '#22c55e' },
    MEDIUM:   { max: 60,  label: 'MEDIUM RISK',   color: '#f97316' },
    HIGH:     { max: 80,  label: 'HIGH RISK',      color: '#ef4444' },
    CRITICAL: { max: 100, label: 'CRITICAL RISK',  color: '#dc2626' },
  },

  CHART_DEFAULTS: {
    color_amber:    '#f59e0b',
    color_cyan:     '#06b6d4',
    color_danger:   '#ef4444',
    color_success:  '#22c55e',
    color_warning:  '#f97316',
    color_concrete: '#94a3b8',
    grid_color:     'rgba(42,53,80,0.5)',
    text_color:     '#94a3b8',
    font_mono:      "'DM Mono', monospace",
  },
};

/* ============================================================
   CHART COLORS ARRAY
   ============================================================ */
const CHART_COLORS = [
  SB.CHART_DEFAULTS.color_amber,
  SB.CHART_DEFAULTS.color_cyan,
  SB.CHART_DEFAULTS.color_danger,
  SB.CHART_DEFAULTS.color_success,
  SB.CHART_DEFAULTS.color_warning,
  SB.CHART_DEFAULTS.color_concrete,
];

/* ============================================================
   SENSOR CONFIG
   ============================================================ */
const SENSOR_CONFIG = [
  {
    id: 'S1', name: 'Temperature', icon: '🌡️', unit: '°C', max: 60,
    critical_high: 50, warning_high: 40, critical_low: -5, warning_low: 5,
  },
  {
    id: 'S2', name: 'Humidity', icon: '💧', unit: '%', max: 100,
    critical_high: 90, warning_high: 75, critical_low: 10, warning_low: 20,
  },
  {
    id: 'S3', name: 'Vibration', icon: '📳', unit: 'mm/s', max: 20,
    critical_high: 15, warning_high: 10, critical_low: 0, warning_low: 0,
  },
  {
    id: 'S4', name: 'Dust Level', icon: '🌫️', unit: 'µg/m³', max: 500,
    critical_high: 400, warning_high: 250, critical_low: 0, warning_low: 0,
  },
];

/* ============================================================
   DEFAULT SCHEDULE TASKS
   ============================================================ */
function getDefaultTasks() {
  return [
    { id: 't1', name: 'Site Survey & Clearance',   start: 0,  duration: 7,  critical_path: true,  dependencies: [] },
    { id: 't2', name: 'Foundation Excavation',      start: 7,  duration: 14, critical_path: true,  dependencies: ['t1'] },
    { id: 't3', name: 'Foundation Concrete Pour',   start: 21, duration: 10, critical_path: true,  dependencies: ['t2'] },
    { id: 't4', name: 'Ground Floor Slab',          start: 31, duration: 8,  critical_path: false, dependencies: ['t3'] },
    { id: 't5', name: 'Structural Steel Frame',     start: 31, duration: 20, critical_path: true,  dependencies: ['t3'] },
    { id: 't6', name: 'Masonry & Brickwork',        start: 51, duration: 15, critical_path: false, dependencies: ['t5'] },
    { id: 't7', name: 'Roofing & Waterproofing',    start: 51, duration: 12, critical_path: true,  dependencies: ['t5'] },
    { id: 't8', name: 'MEP Rough-In',               start: 63, duration: 18, critical_path: false, dependencies: ['t6'] },
    { id: 't9', name: 'Internal Finishes',          start: 63, duration: 20, critical_path: true,  dependencies: ['t7'] },
    { id: 't10', name: 'External Cladding',         start: 66, duration: 14, critical_path: false, dependencies: ['t7'] },
    { id: 't11', name: 'Fit-Out & Fixtures',        start: 83, duration: 12, critical_path: true,  dependencies: ['t9'] },
    { id: 't12', name: 'Testing & Commissioning',   start: 95, duration: 7,  critical_path: true,  dependencies: ['t11'] },
    { id: 't13', name: 'Snag List & Handover',      start: 102, duration: 5, critical_path: true,  dependencies: ['t12'] },
  ];
}

/* ============================================================
   FORMATTERS
   ============================================================ */
const Fmt = {
  currency(amount, compact = false) {
    if (amount == null || isNaN(amount)) return '₹0';
    const n = Number(amount);
    if (compact) {
      if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`;
      if (n >= 1e5) return `₹${(n / 1e5).toFixed(2)} L`;
      if (n >= 1e3) return `₹${(n / 1e3).toFixed(1)}K`;
      return `₹${n}`;
    }
    return '₹' + n.toLocaleString('en-IN');
  },

  percent(value, decimals = 1) {
    if (value == null || isNaN(value)) return '0%';
    return `${Number(value).toFixed(decimals)}%`;
  },

  days(n) {
    if (n == null) return '—';
    n = Math.round(n);
    if (n === 1) return '1 day';
    if (n < 30) return `${n} days`;
    if (n < 365) {
      const months = Math.floor(n / 30);
      const days   = n % 30;
      return days ? `${months}mo ${days}d` : `${months} months`;
    }
    const yrs = Math.floor(n / 365);
    const mos = Math.floor((n % 365) / 30);
    return mos ? `${yrs}yr ${mos}mo` : `${yrs} yr`;
  },

  area(sqm) {
    if (!sqm) return '—';
    return `${Number(sqm).toLocaleString('en-IN')} sq.m`;
  },

  time(date) {
    const d = date ? new Date(date) : new Date();
    return d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  },

  date(date) {
    const d = date ? new Date(date) : new Date();
    return d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
  },

  timeAgo(date) {
    const secs = Math.floor((Date.now() - new Date(date)) / 1000);
    if (secs < 60)    return `${secs}s ago`;
    if (secs < 3600)  return `${Math.floor(secs / 60)}m ago`;
    if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
    return `${Math.floor(secs / 86400)}d ago`;
  },

  num(n, decimals = 0) {
    if (n == null || isNaN(n)) return '—';
    return Number(n).toFixed(decimals);
  },
};

/* ============================================================
   VALIDATORS
   ============================================================ */
const Validate = {
  required(value, label = 'Field') {
    const v = String(value || '').trim();
    return v ? null : `${label} is required`;
  },

  range(value, min, max, label = 'Value') {
    const n = Number(value);
    if (isNaN(n)) return `${label} must be a number`;
    if (n < min)  return `${label} must be at least ${min}`;
    if (n > max)  return `${label} must be at most ${max}`;
    return null;
  },

  positiveInt(value, label = 'Value') {
    const n = parseInt(value);
    if (isNaN(n) || n <= 0) return `${label} must be a positive integer`;
    return null;
  },

  budget(value) {
    const n = Number(String(value).replace(/[,₹\s]/g, ''));
    if (isNaN(n) || n <= 0) return 'Enter a valid budget amount';
    if (n < 100000)          return 'Budget seems too low (min ₹1,00,000)';
    return null;
  },

  location(value) {
    const v = String(value || '').trim();
    if (!v)           return 'Location is required';
    if (v.length < 3) return 'Enter a valid location name';
    return null;
  },

  timeline(value) {
    const n = parseInt(value);
    if (isNaN(n) || n <= 0) return 'Enter valid timeline in days';
    if (n < 7)    return 'Minimum 7 days';
    if (n > 3650) return 'Maximum 10 years (3650 days)';
    return null;
  },

  first(...checks) {
    for (const err of checks) { if (err) return err; }
    return null;
  },

  all(rules) {
    const errors = {};
    for (const [field, fn] of Object.entries(rules)) {
      const err = fn();
      if (err) errors[field] = err;
    }
    return Object.keys(errors).length ? errors : null;
  },
};

/* ============================================================
   DOM HELPERS
   ============================================================ */
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

const DOM = {
  id: (id) => document.getElementById(id),

  el(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === 'class')     el.className = v;
      else if (k === 'html') el.innerHTML = v;
      else if (k === 'text') el.textContent = v;
      else el.setAttribute(k, v);
    }
    children.forEach(c => el.append(typeof c === 'string' ? document.createTextNode(c) : c));
    return el;
  },

  show(el) {
    if (typeof el === 'string') el = DOM.id(el);
    if (el) el.style.display = '';
  },

  hide(el) {
    if (typeof el === 'string') el = DOM.id(el);
    if (el) el.style.display = 'none';
  },

  toggle(el, cls, force) {
    if (typeof el === 'string') el = DOM.id(el);
    if (el) el.classList.toggle(cls, force);
  },

  text(id, value) {
    const el = typeof id === 'string' ? DOM.id(id) : id;
    if (el) el.textContent = value ?? '—';
  },

  html(id, value) {
    const el = typeof id === 'string' ? DOM.id(id) : id;
    if (el) el.innerHTML = value ?? '';
  },

  val(id, value) {
    const el = typeof id === 'string' ? DOM.id(id) : id;
    if (el) el.value = value ?? '';
  },

  fieldError(inputEl, msg) {
    if (typeof inputEl === 'string') inputEl = DOM.id(inputEl);
    if (!inputEl) return;
    inputEl.classList.add('input-error');
    let errEl = inputEl.nextElementSibling;
    if (!errEl || !errEl.classList.contains('error-msg')) {
      errEl = DOM.el('div', { class: 'error-msg' });
      inputEl.after(errEl);
    }
    errEl.textContent = msg;
  },

  fieldOk(inputEl) {
    if (typeof inputEl === 'string') inputEl = DOM.id(inputEl);
    if (!inputEl) return;
    inputEl.classList.remove('input-error');
    const errEl = inputEl.nextElementSibling;
    if (errEl?.classList.contains('error-msg')) errEl.remove();
  },

  clearErrors(container = document) {
    $$('.input-error', container).forEach(el => el.classList.remove('input-error'));
    $$('.error-msg',   container).forEach(el => el.remove());
  },

  animateCounter(el, target, duration = 1200, format = (n) => Math.round(n)) {
    if (typeof el === 'string') el = DOM.id(el);
    if (!el) return;
    const start = Date.now();
    const tick = () => {
      const elapsed  = Date.now() - start;
      const progress = Math.min(elapsed / duration, 1);
      const ease     = 1 - Math.pow(1 - progress, 3);
      el.textContent = format(ease * target);
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  },
};

/* ============================================================
   DEBOUNCE & THROTTLE
   ============================================================ */
function debounce(fn, wait = 300) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), wait);
  };
}

function throttle(fn, wait = 200) {
  let last = 0;
  return function (...args) {
    const now = Date.now();
    if (now - last >= wait) {
      last = now;
      fn.apply(this, args);
    }
  };
}

/* ============================================================
   LOCAL STORAGE HELPERS
   ============================================================ */
const Store = {
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); return true; }
    catch (e) { console.warn('Store.set error:', e); return false; }
  },
  get(key, fallback = null) {
    try {
      const raw = localStorage.getItem(key);
      return raw != null ? JSON.parse(raw) : fallback;
    } catch (e) { return fallback; }
  },
  remove(key) { try { localStorage.removeItem(key); } catch (_) {} },
  clear()     { try { localStorage.clear(); }         catch (_) {} },
};

/* ============================================================
   CHART.JS CONFIG DEFAULTS
   ============================================================ */
const ChartCfg = {
  base() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: SB.CHART_DEFAULTS.text_color,
            font:  { family: SB.CHART_DEFAULTS.font_mono, size: 11 },
            boxWidth: 12,
            padding: 16,
          }
        },
        tooltip: {
          backgroundColor: '#1a2035',
          borderColor:      '#2a3550',
          borderWidth: 1,
          titleColor: '#f1f5f9',
          bodyColor:  '#94a3b8',
          titleFont:  { family: SB.CHART_DEFAULTS.font_mono, size: 11 },
          bodyFont:   { family: SB.CHART_DEFAULTS.font_mono, size: 11 },
          padding: 10,
          cornerRadius: 6,
        }
      },
      scales: {
        x: {
          grid:  { color: SB.CHART_DEFAULTS.grid_color },
          ticks: { color: SB.CHART_DEFAULTS.text_color, font: { family: SB.CHART_DEFAULTS.font_mono, size: 10 } },
        },
        y: {
          grid:  { color: SB.CHART_DEFAULTS.grid_color },
          ticks: { color: SB.CHART_DEFAULTS.text_color, font: { family: SB.CHART_DEFAULTS.font_mono, size: 10 } },
        }
      }
    };
  },

  radar() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: SB.CHART_DEFAULTS.text_color,
            font:  { family: SB.CHART_DEFAULTS.font_mono, size: 11 },
          }
        },
        tooltip: {
          backgroundColor: '#1a2035',
          borderColor:      '#2a3550',
          borderWidth: 1,
          titleColor: '#f1f5f9',
          bodyColor:  '#94a3b8',
          titleFont:  { family: SB.CHART_DEFAULTS.font_mono, size: 11 },
          bodyFont:   { family: SB.CHART_DEFAULTS.font_mono, size: 11 },
          padding: 10,
          cornerRadius: 6,
        }
      },
      scales: {
        r: {
          grid:        { color: SB.CHART_DEFAULTS.grid_color },
          angleLines:  { color: SB.CHART_DEFAULTS.grid_color },
          pointLabels: { color: SB.CHART_DEFAULTS.text_color, font: { family: SB.CHART_DEFAULTS.font_mono, size: 10 } },
          ticks:       { color: SB.CHART_DEFAULTS.text_color, backdropColor: 'transparent', font: { size: 9 } },
          suggestedMin: 0,
          suggestedMax: 100,
        }
      }
    };
  },

  doughnut() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: {
          position: 'right',
          labels: {
            color: SB.CHART_DEFAULTS.text_color,
            font:  { family: SB.CHART_DEFAULTS.font_mono, size: 10 },
            boxWidth: 10, padding: 12,
          }
        },
        tooltip: {
          backgroundColor: '#1a2035',
          borderColor:      '#2a3550',
          borderWidth: 1,
          titleColor: '#f1f5f9',
          bodyColor:  '#94a3b8',
          titleFont:  { family: SB.CHART_DEFAULTS.font_mono, size: 11 },
          bodyFont:   { family: SB.CHART_DEFAULTS.font_mono, size: 11 },
          padding: 10, cornerRadius: 6,
        }
      }
    };
  },

  replace(canvasId, type, data, options = {}) {
    const canvas = DOM.id(canvasId);
    if (!canvas) return null;
    const existing = Chart.getChart(canvas);
    if (existing) existing.destroy();
    return new Chart(canvas, { type, data, options });
  },
};

/* ============================================================
   RISK LEVEL HELPERS
   ============================================================ */
function getRiskLevel(percent) {
  const p = Number(percent);
  if (p <= 30)  return SB.RISK_LEVELS.LOW;
  if (p <= 60)  return SB.RISK_LEVELS.MEDIUM;
  if (p <= 80)  return SB.RISK_LEVELS.HIGH;
  return SB.RISK_LEVELS.CRITICAL;
}

/* ============================================================
   DATE / TIMELINE UTILITIES
   ============================================================ */
const Timeline = {
  addDays(date, days) {
    const d = new Date(date);
    d.setDate(d.getDate() + days);
    return d;
  },

  diffDays(a, b) {
    const msPerDay = 1000 * 60 * 60 * 24;
    return Math.round((new Date(b) - new Date(a)) / msPerDay);
  },

  workingDays(startDate, endDate) {
    let count = 0;
    const cur = new Date(startDate);
    const end = new Date(endDate);
    while (cur <= end) {
      if (cur.getDay() !== 0) count++;
      cur.setDate(cur.getDate() + 1);
    }
    return count;
  },

  completionDate(startDate, durationDays) {
    return Fmt.date(Timeline.addDays(startDate, durationDays));
  },
};

/* ============================================================
   MISC UTILITIES
   ============================================================ */
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function randomBetween(min, max) { return Math.random() * (max - min) + min; }
function randomInt(min, max) { return Math.floor(randomBetween(min, max + 1)); }
function clamp(value, min, max) { return Math.min(Math.max(value, min), max); }
function generateId(prefix = 'sb') {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
}
function deepClone(obj) { return JSON.parse(JSON.stringify(obj)); }
function capitalize(str) { return str ? str[0].toUpperCase() + str.slice(1).toLowerCase() : ''; }
function truncate(str, max = 40) { return str && str.length > max ? str.slice(0, max) + '…' : str; }

/* ============================================================
   TOAST NOTIFICATION SYSTEM
   ============================================================ */
const Toast = (() => {
  let container;

  function getContainer() {
    if (!container) {
      container = document.querySelector('.toast-container');
      if (!container) {
        container = DOM.el('div', { class: 'toast-container' });
        document.body.appendChild(container);
      }
    }
    return container;
  }

  function show(message, type = 'amber', duration = 4000) {
    const icons = { amber: '⚡', cyan: '📡', danger: '🚨', success: '✅', warning: '⚠️', info: '📡', error: '🚨' };
    const typeMap = { info: 'cyan', error: 'danger', warning: 'amber', success: 'success' };
    const resolvedType = typeMap[type] || type;
    const toast = DOM.el('div', { class: `toast ${resolvedType}` });
    toast.innerHTML = `<span class="toast-icon">${icons[type] || '•'}</span><span class="toast-message">${message}</span><button class="toast-close" onclick="this.parentElement.remove()">✕</button>`;
    getContainer().appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, duration);
    return toast;
  }

  return {
    success: (msg, d) => show(msg, 'success', d),
    info:    (msg, d) => show(msg, 'info', d),
    warn:    (msg, d) => show(msg, 'warning', d),
    error:   (msg, d) => show(msg, 'error', d),
  };
})();

/* ============================================================
   GENERATE RECOMMENDATIONS
   ============================================================ */
function generateRecommendations(metrics) {
  const recs = [];

  if (metrics.risk_index > 60) {
    recs.push({
      icon: '🛡️',
      title: 'High Risk Index',
      desc: `Risk index at ${Math.round(metrics.risk_index)}%. Review mitigation strategies and update contingency plans immediately.`,
      priority: 'critical',
    });
  }

  if (metrics.budget_used > 85) {
    recs.push({
      icon: '💰',
      title: 'Budget Overrun Warning',
      desc: `${Math.round(metrics.budget_used)}% of budget consumed. Review procurement and explore cost optimisation measures.`,
      priority: 'warning',
    });
  }

  if (metrics.schedule_var > 5) {
    recs.push({
      icon: '📅',
      title: 'Schedule Delay Detected',
      desc: `Project is ${Math.round(metrics.schedule_var)} days behind plan. Increase resources on critical path tasks.`,
      priority: 'warning',
    });
  }

  if (metrics.productivity < 65) {
    recs.push({
      icon: '👷',
      title: 'Low Team Productivity',
      desc: `Productivity at ${Math.round(metrics.productivity)}%. Consider workflow optimisation, additional training, or equipment upgrades.`,
      priority: 'info',
    });
  }

  if (metrics.health_score >= 80) {
    recs.push({
      icon: '✅',
      title: 'Project On Track',
      desc: 'Health score is strong. Maintain current practices and document lessons learned for future projects.',
      priority: 'success',
    });
  }

  if (!recs.length) {
    recs.push({
      icon: '📊',
      title: 'Monitoring Active',
      desc: 'All metrics within normal ranges. Continue monitoring sensor feeds and daily reports.',
      priority: 'info',
    });
  }

  return recs;
}

/* ============================================================
   LOADING STATE HELPER
   ============================================================ */
function setLoading(btn, loading, label) {
  if (!btn) return;
  btn.disabled   = loading;
  btn.textContent = label;
  if (loading) {
    btn.style.opacity = '0.7';
    btn.style.cursor  = 'wait';
  } else {
    btn.style.opacity = '1';
    btn.style.cursor  = 'pointer';
  }
}

/* ============================================================
   UTILS — unified proxy object used by features.js
   Maps every Utils.X call to the correct function/constant
   ============================================================ */
const Utils = {
  /* Formatting */
  formatCurrency:     (v) => Fmt.currency(v, false),
  formatCurrencyFull: (v) => Fmt.currency(v, false),
  formatCurrencyCompact: (v) => Fmt.currency(v, true),
  formatPercent:      (v, d) => Fmt.percent(v, d),
  formatDays:         (v) => Fmt.days(v),
  formatTime:         (d) => Fmt.time(d),
  formatDate:         (d) => Fmt.date(d),
  formatTimeAgo:      (d) => Fmt.timeAgo(d),

  /* Date / Timeline */
  calcDaysBetween: (a, b) => Timeline.diffDays(a, b),
  addDays:         (d, n) => Timeline.addDays(d, n),

  /* DOM */
  showFieldError:  (el, msg) => DOM.fieldError(el, msg),
  clearFieldError: (el) => DOM.fieldOk(el),

  /* Toast */
  showToast: (msg, type = 'info') => {
    const map = { success: Toast.success, error: Toast.error, warning: Toast.warn, warn: Toast.warn, info: Toast.info, critical: Toast.error };
    const fn  = map[type] || Toast.info;
    fn(msg);
  },

  /* Loading state */
  setLoading: (btn, loading, label) => setLoading(btn, loading, label),

  /* Math */
  clamp:         (v, min, max) => clamp(v, min, max),
  randomBetween: (min, max)    => randomBetween(min, max),
  randomInt:     (min, max)    => randomInt(min, max),

  /* Chart defaults */
  chartDefaults: () => ChartCfg.base(),
  CHART_COLORS,

  /* Sensor config */
  SENSOR_CONFIG,

  /* Schedule tasks */
  getDefaultTasks: () => getDefaultTasks(),

  /* Dashboard recommendations */
  generateRecommendations: (metrics) => generateRecommendations(metrics),
};

/* ============================================================
   GLOBAL WINDOW EXPORTS
   ============================================================ */
if (typeof window !== 'undefined') {
  window.SB               = SB;
  window.Fmt              = Fmt;
  window.Validate         = Validate;
  window.DOM              = DOM;
  window.Store            = Store;
  window.ChartCfg         = ChartCfg;
  window.Timeline         = Timeline;
  window.Toast            = Toast;
  window.$                = $;
  window.$$               = $$;
  window.debounce         = debounce;
  window.throttle         = throttle;
  window.sleep            = sleep;
  window.randomBetween    = randomBetween;
  window.randomInt        = randomInt;
  window.clamp            = clamp;
  window.generateId       = generateId;
  window.deepClone        = deepClone;
  window.getRiskLevel     = getRiskLevel;
  window.setLoading       = setLoading;
  window.CHART_COLORS     = CHART_COLORS;
  window.SENSOR_CONFIG    = SENSOR_CONFIG;
  window.getDefaultTasks  = getDefaultTasks;
  window.generateRecommendations = generateRecommendations;
  window.Utils            = Utils;   /* ← THIS is what features.js needs */
}

console.info(`[SmartBuild] utils.js loaded — v${SB.VERSION}`);
