/* ══════════════════════════════════════════════════════════════════
   SMARTBUILD AI — api.js
   All server communication in one place
   Depends on: utils.js (loaded first)
   ════════════════════════════════════════════════════════════════════ */

/* ══════════════════════════════════════════════════════════════════
   BASE CONFIGURATION
   ─── FIX: Auto-detect environment — localhost in dev, Render URLs in prod
   ════════════════════════════════════════════════════════════════════ */
const _isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

const API = {
  BASE_URL: _isLocal ? 'http://localhost:8000' : 'https://buildnexus-ai-aditya.onrender.com',
  ML_URL:   _isLocal ? 'http://localhost:8000' : 'https://buildnexus-ai-aditya.onrender.com',
  socket: null,
  projectId: null,
};

/* ══════════════════════════════════════════════════════════════════
   BASE FETCH WRAPPER
   ════════════════════════════════════════════════════════════════════ */
async function apiFetch(url, options = {}) {
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  };

  const response = await fetch(url, { ...defaultOptions, ...options });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Request failed' }));
    throw new Error(error.message || `HTTP ${response.status}`);
  }

  return response.json();
}

/* ══════════════════════════════════════════════════════════════════
   WEBSOCKET / SOCKET.IO CONNECTION
   ════════════════════════════════════════════════════════════════════ */
function initWebSocket() {
  if (typeof io === 'undefined') {
    console.warn('[SmartBuild] Socket.IO not loaded — real-time features disabled');
    return;
  }

  API.socket = io(API.BASE_URL, {
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionAttempts: 5,
  });

  API.socket.on('connect', () => {
    console.log('[SmartBuild] ✓ WebSocket connected');
    updateConnectionStatus(true);
    Utils.showToast('Connected to server', 'success');
  });

  API.socket.on('disconnect', () => {
    console.log('[SmartBuild] ✗ WebSocket disconnected');
    updateConnectionStatus(false);
    Utils.showToast('Disconnected from server', 'warning');
  });

  API.socket.on('connect_error', (error) => {
    console.error('[SmartBuild] WebSocket error:', error);
    updateConnectionStatus(false);
  });

  API.socket.on('sensor-update', (data) => {
    if (window.Feature6) Feature6.updateSensor(data);
  });

  API.socket.on('alert-triggered', (alert) => {
    if (window.Feature7) Feature7.triggerAlert(alert);
  });

  API.socket.on('schedule-adjusted', (data) => {
    if (window.Feature8) {
      Feature8.tasks = data.tasks;
      Feature8.renderGantt();
      Utils.showToast('Schedule auto-adjusted by server', 'info');
    }
  });

  API.socket.on('design-generated', (designs) => {
    if (window.Feature2) Feature2.loadDesigns(designs);
  });
}

function updateConnectionStatus(isConnected) {
  const el = document.getElementById('connectionStatus');
  if (el) {
    el.textContent = isConnected ? 'LIVE' : 'OFFLINE';
    el.style.color = isConnected ? 'var(--cyan)' : 'var(--danger)';
  }
}

function subscribeToProject(projectId) {
  if (API.socket?.connected) {
    API.socket.emit('subscribe', { projectId });
    API.projectId = projectId;
  }
}

/* ══════════════════════════════════════════════════════════════════
   DESIGN API — used by Feature1, Feature2, Feature3
   ════════════════════════════════════════════════════════════════════ */
const DesignAPI = {
  async generateDesign(params) {
    const payload = {
      projectName: params.projectName || params.name || '',
      projectType: params.projectType || params.type || 'commercial',
      totalArea: parseFloat(params.totalArea || params.area || 0),
      location: params.location || '',
      budget: parseFloat(params.budget || 0),
      maxFloors: parseInt(params.maxFloors || 10),
      priority: params.priority || params.structPriority || 'balanced',
      startDate: params.startDate || '',
      endDate: params.endDate || '',
      qualityGrade: parseInt(params.qualityGrade || 2),
      siteAccess: parseInt(params.siteAccess || 2),
    };
    return apiFetch(`${API.BASE_URL}/generate-design`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async getDesign(designId) {
    return apiFetch(`${API.BASE_URL}/api/design/${designId}`);
  },

  async selectDesign(designId, projectData = {}) {
    return apiFetch(`${API.BASE_URL}/api/design/${designId}/select`, {
      method: 'PUT',
      body: JSON.stringify(projectData),
    });
  },

  async compareDesigns(designIds) {
    return apiFetch(`${API.BASE_URL}/api/design/compare`, {
      method: 'POST',
      body: JSON.stringify({ designIds }),
    });
  },

  async deleteDesign(designId) {
    return apiFetch(`${API.BASE_URL}/api/design/${designId}`, { method: 'DELETE' });
  },
};

/* ══════════════════════════════════════════════════════════════════
   PREDICTION API — used by Feature4, Feature5
   ─── FIX: predictRisk and estimateCost now call BASE_URL (Node backend)
            instead of ML_URL directly — avoids CORS issues on Render
            since the Node server proxies to Flask internally
   ════════════════════════════════════════════════════════════════════ */
const PredictionAPI = {
  async predictRisk(params) {
    const payload = {
      weather_condition:     parseInt(params.weather_condition     ?? params.weatherCondition ?? 1),
      material_availability: parseFloat(params.material_availability ?? params.materialAvail   ?? 80) / 100,
      labor_count:           parseInt(params.labor_count           ?? params.laborCount        ?? 50),
      equipment_status:      parseInt(params.equipment_status      ?? params.equipStatus       ?? 1),
      project_complexity:    parseInt(params.project_complexity    ?? params.complexity        ?? 2),
      site_accessibility:    parseInt(params.site_accessibility    ?? params.siteAccess        ?? 2),
    };
    return apiFetch(`${API.BASE_URL}/predict-risk`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async estimateCost(params) {
    const payload = {
      area:             parseFloat(params.area || params.totalArea || 0),
      building_type:    params.buildingType || params.projectType || 'commercial',
      location:         params.location || '',
      timeline_months:  parseInt(params.timeline || 12),
      quality_grade:    parseInt(params.qualityGrade || 2),
    };
    return apiFetch(`${API.BASE_URL}/estimate-cost`, {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /*
   * getWeather(locationString)  — city name or "lat,lon" string
   * getWeather(lat, lon)        — two separate numbers
   * Uses Open-Meteo — free, no API key required
   */
  async getWeather(latOrLocation, lon) {
    try {
      let lat, resolvedLon;

      if (typeof latOrLocation === 'string' && isNaN(parseFloat(latOrLocation))) {
        /* City name — geocode via Open-Meteo (no key required) */
        const geo = await fetch(
          `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(latOrLocation)}&count=1`
        ).then(r => r.json());
        if (!geo.results?.length) throw new Error('Location not found');
        lat = geo.results[0].latitude;
        resolvedLon = geo.results[0].longitude;
      } else if (typeof latOrLocation === 'string' && latOrLocation.includes(',')) {
        /* "lat,lon" string */
        const parts = latOrLocation.split(',');
        lat = parseFloat(parts[0]);
        resolvedLon = parseFloat(parts[1]);
      } else {
        /* Two numeric args */
        lat = parseFloat(latOrLocation);
        resolvedLon = parseFloat(lon);
      }

      const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${resolvedLon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode&timezone=auto&forecast_days=7`;
      const data = await fetch(url).then(r => r.json());
      return { forecast: this._processOpenMeteo(data) };
    } catch (err) {
      console.error('[PredictionAPI] getWeather error:', err);
      throw err;
    }
  },

  _processOpenMeteo(data) {
    const wmoCodes = {
      0: 'Clear', 1: 'Clear', 2: 'Partly Cloudy', 3: 'Cloudy',
      45: 'Fog', 48: 'Fog',
      51: 'Drizzle', 53: 'Drizzle', 55: 'Drizzle',
      61: 'Rain', 63: 'Rain', 65: 'Heavy Rain',
      71: 'Snow', 73: 'Snow', 75: 'Heavy Snow',
      80: 'Rain', 81: 'Rain', 82: 'Heavy Rain',
      95: 'Thunderstorm', 96: 'Thunderstorm', 99: 'Thunderstorm',
    };

    return data.daily.time.map((date, i) => ({
      date,
      condition: wmoCodes[data.daily.weathercode[i]] || 'Cloudy',
      rain_mm:   data.daily.precipitation_sum[i]    || 0,
      wind_kmh:  data.daily.windspeed_10m_max[i]    || 0,
      temp_max:  data.daily.temperature_2m_max[i]   || 30,
      temp_min:  data.daily.temperature_2m_min[i]   || 20,
    }));
  },
};

/* ══════════════════════════════════════════════════════════════════
   MONITOR API — used by Feature6, Feature7
   ════════════════════════════════════════════════════════════════════ */
const MonitorAPI = {
  async getSensors() {
    const res = await fetch(`${API.BASE_URL}/site-monitor`);
    const data = await res.json();

    return {
      sensors: [
        { sensorId: "S1", value: data.temperature },
        { sensorId: "S2", value: data.humidity },
        { sensorId: "S3", value: data.vibration },
        { sensorId: "S4", value: data.dust }
      ]
    };
  },

  async getSensor(sensorId) {
    return apiFetch(`${API.BASE_URL}/api/monitor/sensors/${sensorId}`);
  },

  async updateSensor(sensorId, value) {
    if (API.socket?.connected) {
      API.socket.emit('update-sensor', { sensorId, value });
    }
    return apiFetch(`${API.BASE_URL}/api/monitor/sensors/${sensorId}`, {
      method: 'PUT',
      body: JSON.stringify({ value }),
    });
  },

  async createAlert(alert) {
    return apiFetch(`${API.BASE_URL}/api/monitor/alert`, {
      method: 'POST',
      body: JSON.stringify(alert),
    });
  },

  async getAlerts(projectId) {
    return apiFetch(`${API.BASE_URL}/api/monitor/alerts?projectId=${projectId || 'demo'}`);
  },

  async dismissAlert(alertId) {
    return apiFetch(`${API.BASE_URL}/api/monitor/alerts/${alertId}`, { method: 'DELETE' });
  },

  async getStatus(projectId) {
    return apiFetch(`${API.BASE_URL}/api/monitor/status?projectId=${projectId || 'demo'}`);
  },
};

/* ══════════════════════════════════════════════════════════════════
   OPTIMIZE API — used by Feature8, Feature9
   ════════════════════════════════════════════════════════════════════ */
const OptimizeAPI = {
  async adjustSchedule(scheduleData) {
    return apiFetch(`${API.BASE_URL}/api/optimize/schedule`, {
      method: 'POST',
      body: JSON.stringify(scheduleData),
    });
  },

  async calculateCostImpact(costData) {
    return apiFetch(`${API.BASE_URL}/api/optimize/cost`, {
      method: 'POST',
      body: JSON.stringify(costData),
    });
  },

  async getRecommendations(projectId) {
    return apiFetch(`${API.BASE_URL}/api/optimize/recommendations?projectId=${projectId || 'demo'}`);
  },

  async optimizeResources(resourceData) {
    return apiFetch(`${API.BASE_URL}/api/optimize/resources`, {
      method: 'POST',
      body: JSON.stringify(resourceData),
    });
  },

  /** Client-side CPM — fallback when backend is offline */
  calculateCriticalPath(tasks) {
    const map = new Map(tasks.map(t => [t.id, { ...t, earlyStart: 0, earlyFinish: 0, lateStart: 0, lateFinish: 0, slack: 0 }]));

    /* Forward pass */
    for (const task of tasks) {
      const cur = map.get(task.id);
      if (task.dependencies?.length) {
        cur.earlyStart = Math.max(...task.dependencies.map(id => map.get(id)?.earlyFinish || 0));
      }
      cur.earlyFinish = cur.earlyStart + task.duration;
    }

    const projectDuration = Math.max(...[...map.values()].map(t => t.earlyFinish));

    /* Backward pass */
    for (const task of [...tasks].reverse()) {
      const cur = map.get(task.id);
      const dependents = tasks.filter(t => t.dependencies?.includes(task.id));
      cur.lateFinish = dependents.length
        ? Math.min(...dependents.map(d => map.get(d.id).lateStart))
        : projectDuration;
      cur.lateStart  = cur.lateFinish - task.duration;
      cur.slack      = cur.lateStart  - cur.earlyStart;
    }

    const critical = [...map.values()].filter(t => t.slack === 0);
    return { tasks: [...map.values()], criticalPath: critical.map(t => t.id), projectDuration };
  },
};

/* ══════════════════════════════════════════════════════════════════
   DASHBOARD API — used by Feature10
   ════════════════════════════════════════════════════════════════════ */
const DashboardAPI = {
  async getKPIs(projectId) {
    return apiFetch(`${API.BASE_URL}/api/dashboard/kpis?projectId=${projectId || 'demo'}`);
  },

  async getHealthMetrics(projectId) {
  return apiFetch(`${API.BASE_URL}/health`);
  },

  async getProductivity(projectId) {
    return apiFetch(`${API.BASE_URL}/api/dashboard/productivity?projectId=${projectId || 'demo'}`);
  },

  calculateHealthScore(metrics) {
    let score = 100;
    const bv = Math.abs((metrics.budgetUsed - metrics.budgetPlanned) / (metrics.budgetPlanned || 1));
    score -= Math.min(30, bv * 100);
    score -= Math.min(30, Math.abs(metrics.scheduleDays || 0) * 3);
    score -= Math.min(20, metrics.riskIndex || 0);
    score -= Math.min(20, (metrics.qualityIssues || 0) * 5);
    return Math.max(0, Math.round(score));
  },
};

/* ══════════════════════════════════════════════════════════════════
   INITIALIZATION
   ════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  //initWebSocket();
});

window.addEventListener('beforeunload', () => {
  if (API.socket) API.socket.disconnect();
});

/* ══════════════════════════════════════════════════════════════════
   GLOBAL EXPORTS
   ════════════════════════════════════════════════════════════════════ */
window.API         = API;
window.DesignAPI   = DesignAPI;
window.PredictionAPI = PredictionAPI;
window.MonitorAPI  = MonitorAPI;
window.OptimizeAPI = OptimizeAPI;
window.DashboardAPI = DashboardAPI;
window.apiFetch    = apiFetch;
window.subscribeToProject = subscribeToProject;


console.info('[SmartBuild] api.js loaded — ENV:', _isLocal ? 'LOCAL' : 'PRODUCTION');
