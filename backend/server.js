const express    = require('express');
const http       = require('http');
const { Server } = require('socket.io');
const cors       = require('cors');
const bodyParser = require('body-parser');
const axios      = require('axios');
const path       = require('path');       // ─── FIX: added path module

const app    = express();
const server = http.createServer(app);
const io     = new Server(server, { cors: { origin: '*' } });

app.use(cors());
app.use(bodyParser.json());
// ─── FIX: use absolute path so frontend loads on Render ──────────────
app.use(express.static(path.join(__dirname, '../frontend'), {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.css')) {
      res.setHeader('Content-Type', 'text/css');
    } else if (filePath.endsWith('.js')) {
      res.setHeader('Content-Type', 'application/javascript');
    } else if (filePath.endsWith('.html')) {
      res.setHeader('Content-Type', 'text/html');
    }
  }
}));

// ─── FIX: read ML_URL from environment variable ──────────────────────
// On Render → set ML_URL = https://smartbuild-ml.onrender.com
// Locally   → falls back to localhost:5000
const ML_URL = process.env.ML_URL || 'http://localhost:5000';

// ─── IN-MEMORY STORE ─────────────────────────────────────────────────
let projects  = {};
let sensors   = {};
let alerts    = [];
let schedules = {};

// ─── DESIGN ROUTES ───────────────────────────────────────────────────
app.post('/api/design/generate', async (req, res) => {
  try {
    const result = await axios.post(`${ML_URL}/generate-design`, req.body);
    const id = `d_${Date.now()}`;
    projects[id] = { id, ...result.data };
    res.json({ id, designs: result.data.designs });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/design/:id', (req, res) => {
  const project = projects[req.params.id];
  if (!project) return res.status(404).json({ error: 'Not found' });
  res.json(project);
});

app.put('/api/design/:id/select', (req, res) => {
  const projectId = req.params.id;
  if (projects[projectId]) {
    projects[projectId].selected = req.body.designIndex;
  }
  res.json({ success: true, projectId, designIndex: req.body.designIndex });
});

app.delete('/api/design/:id', (req, res) => {
  delete projects[req.params.id];
  res.json({ success: true });
});

// ─── PREDICTION ROUTES ───────────────────────────────────────────────
app.post('/api/predict/risk', async (req, res) => {
  try {
    const result = await axios.post(`${ML_URL}/predict-risk`, req.body);
    res.json(result.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/predict/cost', async (req, res) => {
  try {
    const result = await axios.post(`${ML_URL}/estimate-cost`, req.body);
    res.json(result.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/predict/weather', async (req, res) => {
  try {
    const { lat, lon } = req.query;
    const apiKey = process.env.WEATHER_API_KEY || 'demo';
    const url = `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric`;
    const result = await axios.get(url);
    res.json(result.data);
  } catch (err) {
    res.json({
      list: Array.from({ length: 7 }, (_, i) => ({
        dt: Date.now() / 1000 + i * 86400,
        main: { temp: 28 + Math.random() * 5, humidity: 60 + Math.random() * 20 },
        weather: [{ main: i % 3 === 0 ? 'Rain' : 'Clear', description: 'mock data' }],
        wind: { speed: 3 + Math.random() * 5 }
      }))
    });
  }
});

// ─── MONITORING ROUTES ───────────────────────────────────────────────
app.get('/api/monitor/sensors', (req, res) => {
  res.json({ sensors: Object.values(sensors) });
});

app.post('/api/monitor/alert', (req, res) => {
  const alert = { id: `a_${Date.now()}`, ...req.body, timestamp: new Date() };
  alerts.unshift(alert);
  io.emit('alert-triggered', alert);
  res.json(alert);
});

app.get('/api/monitor/status', (req, res) => {
  const sensorList = Object.values(sensors);
  res.json({
    health_score:  75 + Math.random() * 15,
    budget_used:   55 + Math.random() * 10,
    schedule_var:  Math.floor(Math.random() * 5),
    risk_index:    30 + Math.random() * 20,
    productivity:  70 + Math.random() * 20,
    totalProjects: Object.keys(projects).length,
    activeSensors: sensorList.length,
    alerts:        alerts.slice(0, 10)
  });
});

// ─── OPTIMIZATION ROUTES ─────────────────────────────────────────────
app.post('/api/optimize/schedule', (req, res) => {
  const { tasks } = req.body;
  if (!tasks || !Array.isArray(tasks)) {
    return res.status(400).json({ error: 'tasks array required' });
  }
  const adjusted = tasks.map(t => ({
    ...t,
    duration: Math.max(1, t.duration - Math.floor(Math.random() * 2)),
    adjusted: true
  }));
  const days_saved = tasks.reduce((sum, t, i) => sum + (t.duration - adjusted[i].duration), 0);
  io.emit('schedule-adjusted', { tasks: adjusted });
  res.json({ tasks: adjusted, days_saved });
});

app.post('/api/optimize/cost', (req, res) => {
  const {
    delay_days       = 0,
    daily_overhead   = 0,
    labor_idle_cost  = 0,
    rework_cost      = 0,
    penalty_per_day  = 0,
    opportunity_loss = 0,
  } = req.body;

  const direct      = delay_days * (daily_overhead + labor_idle_cost);
  const indirect    = delay_days * penalty_per_day + rework_cost;
  const opportunity = opportunity_loss;
  const risk        = (direct + indirect) * 0.15;
  const grand_total = direct + indirect + opportunity + risk;

  res.json({ direct, indirect, opportunity, risk, grand_total });
});

app.get('/api/optimize/recommendations', (req, res) => {
  res.json({
    recommendations: [
      { type: 'schedule', message: 'Pre-order materials 2 weeks early to avoid delay', impact: 'High' },
      { type: 'cost',     message: 'Bulk concrete procurement saves ~12%',              impact: 'Medium' },
      { type: 'risk',     message: 'Weather window optimal next week — accelerate foundation work', impact: 'High' }
    ]
  });
});

// ─── CATCH-ALL: serve index.html for any unknown route ───────────────
// Ensures the frontend loads correctly when accessing the site on Render
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

// ─── SOCKET.IO ───────────────────────────────────────────────────────
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  socket.on('subscribe', ({ projectId }) => {
    socket.join(projectId);
  });

  socket.on('update-sensor', ({ sensorId, value }) => {
    sensors[sensorId] = { sensorId, value, updatedAt: new Date() };
    io.emit('sensor-update', sensors[sensorId]);

    if (value?.temperature > 45) {
      const alert = {
        id:        `a_${Date.now()}`,
        type:      'critical',
        title:     `High Temperature — Sensor ${sensorId}`,
        message:   `Temperature ${value.temperature}°C exceeds safe limit.`,
        timestamp: new Date()
      };
      alerts.unshift(alert);
      io.emit('alert-triggered', alert);
    }
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// ─── SIMULATE LIVE SENSORS ───────────────────────────────────────────
setInterval(() => {
  ['S1', 'S2', 'S3', 'S4'].forEach(id => {
    sensors[id] = {
      sensorId: id,
      value: {
        temperature: +(28 + Math.random() * 10).toFixed(1),
        humidity:    +(55 + Math.random() * 30).toFixed(1),
        progress:    +(Math.random() * 100).toFixed(1),
        workers:     Math.floor(20 + Math.random() * 30),
        equipment:   Math.random() > 0.2 ? 'active' : 'idle'
      },
      updatedAt: new Date()
    };
    io.emit('sensor-update', sensors[id]);
  });
}, 3000);

// ─── FIX: read PORT from env (required by Render) ────────────────────
const PORT = process.env.PORT || 5000;
server.listen(PORT, () => console.log(`SmartBuild server running on port ${PORT}`));
