from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os, numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# ─── MODEL DIR: local to this folder, works on Render ────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── TRAIN FRESH MODELS (called if pkl missing or incompatible) ───────
def train_models():
    print("==> Training models from scratch...", flush=True)
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    np.random.seed(42)
    n = 2000

    # --- Delay predictor ---
    weather    = np.random.randint(0, 4, n)
    materials  = np.random.randint(30, 100, n)
    labor      = np.random.randint(10, 100, n)
    equipment  = np.random.randint(0, 3, n)
    complexity = np.random.randint(1, 4, n)
    access     = np.random.randint(1, 4, n)
    delay = (
        (weather >= 2).astype(int) * 30 +
        (materials < 60).astype(int) * 25 +
        (labor < 30).astype(int) * 20 +
        (equipment == 0).astype(int) * 15 +
        (complexity == 3).astype(int) * 10 +
        np.random.randint(0, 20, n)
    )
    delayed = (delay > 40).astype(int)
    X = np.column_stack([weather, materials, labor, equipment, complexity, access])
    X_train, X_test, y_train, y_test = train_test_split(X, delayed, test_size=0.2, random_state=42)
    d_model = RandomForestClassifier(n_estimators=100, random_state=42)
    d_model.fit(X_train, y_train)
    with open(os.path.join(MODEL_DIR, 'delay_predictor.pkl'), 'wb') as f:
        pickle.dump(d_model, f)
    print("==> Delay model trained ✓", flush=True)

    # --- Cost estimator ---
    area     = np.random.randint(500, 10000, n)
    btype    = np.random.randint(1, 5, n)
    loc      = np.round(np.random.uniform(0.8, 1.5, n), 2)
    timeline = np.random.randint(6, 36, n)
    quality  = np.random.randint(1, 4, n)
    cost = (area * 2000 * loc * (1 + (quality - 1) * 0.15) +
            btype * 50000 + timeline * 10000 +
            np.random.randint(-200000, 200000, n))
    X2 = np.column_stack([area, btype, loc, timeline, quality])
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, cost, test_size=0.2, random_state=42)
    c_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    c_model.fit(X2_train, y2_train)
    with open(os.path.join(MODEL_DIR, 'cost_estimator.pkl'), 'wb') as f:
        pickle.dump(c_model, f)
    print("==> Cost model trained ✓", flush=True)

    return d_model, c_model

# ─── LOAD MODEL SAFELY ───────────────────────────────────────────────
def load_model(name):
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"==> Could not load {name}: {e}", flush=True)
            return None
    return None

# ─── LOAD OR TRAIN ───────────────────────────────────────────────────
delay_model = load_model('delay_predictor.pkl')
cost_model  = load_model('cost_estimator.pkl')

if delay_model is None or cost_model is None:
    delay_model, cost_model = train_models()

print("==> All models ready ✓", flush=True)

# ─── LOAD OPTIONAL MODULES ───────────────────────────────────────────
try:
    from design_optimizer import run_genetic_algorithm
    HAS_GA = True
    print("==> design_optimizer loaded ✓", flush=True)
except Exception as e:
    HAS_GA = False
    print(f"==> design_optimizer not available: {e}", flush=True)

try:
    from risk_predictor import predict_delay_risk
    HAS_RISK = True
    print("==> risk_predictor loaded ✓", flush=True)
except Exception as e:
    HAS_RISK = False
    print(f"==> risk_predictor not available: {e}", flush=True)

# ─── ROUTES ──────────────────────────────────────────────────────────
@app.route('/generate-design', methods=['POST'])
def optimize_design():
    data = request.json
    if HAS_GA:
        designs = run_genetic_algorithm(data)
    else:
        # Fallback mock designs
        designs = [
            { 'id': 1, 'name': 'Design A', 'cost': 5200000, 'duration': 18, 'safety': 87, 'carbon': 420 },
            { 'id': 2, 'name': 'Design B', 'cost': 4800000, 'duration': 20, 'safety': 91, 'carbon': 390 },
            { 'id': 3, 'name': 'Design C', 'cost': 5600000, 'duration': 16, 'safety': 94, 'carbon': 460 },
        ]
    return jsonify({ 'designs': designs })

@app.route('/predict-risk', methods=['POST'])
def predict_risk():
    data = request.json
    if HAS_RISK:
        result = predict_delay_risk(data, delay_model)
    else:
        # Fallback mock risk
        features = [
            data.get('weather_condition', 1),
            data.get('material_availability', 70),
            data.get('labor_count', 50),
            data.get('equipment_status', 1),
            data.get('project_complexity', 2),
            data.get('site_accessibility', 2),
        ]
        prob = float(delay_model.predict_proba([features])[0][1])
        result = {
            'delay_probability': round(prob * 100, 1),
            'risk_category': 'High' if prob > 0.6 else 'Medium' if prob > 0.3 else 'Low',
            'confidence': 82,
            'top_factors': ['Weather conditions', 'Material availability', 'Labor count'],
            'mitigation': ['Monitor weather forecasts', 'Pre-order materials', 'Increase labor']
        }
    return jsonify(result)

@app.route('/estimate-cost', methods=['POST'])
def estimate_cost():
    data = request.json
    features = [
        data.get('area', 1000),
        data.get('building_type', 1),
        data.get('location_factor', 1.0),
        data.get('timeline_months', 12),
        data.get('quality_grade', 2)
    ]
    try:
        cost = float(cost_model.predict([features])[0])
    except Exception:
        cost = features[0] * 2500 * features[2] * (1 + (features[4] - 1) * 0.15)

    return jsonify({
        'estimated_cost': round(cost, 2),
        'breakdown': {
            'material':  round(cost * 0.45, 2),
            'labor':     round(cost * 0.30, 2),
            'equipment': round(cost * 0.15, 2),
            'overhead':  round(cost * 0.10, 2)
        },
        'confidence': 0.82
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models': {
            'cost':  cost_model  is not None,
            'delay': delay_model is not None
        }
    })

# ─── START ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"==> Starting on port {port}", flush=True)
    app.run(host='0.0.0.0', port=port, debug=False)
