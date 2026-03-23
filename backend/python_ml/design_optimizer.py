import random
import numpy as np
from deap import base, creator, tools, algorithms
import joblib
import os 

MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml_models")

cost_model = joblib.load(os.path.join(MODEL_DIR, "cost_estimator.pkl"))
delay_model = joblib.load(os.path.join(MODEL_DIR, "delay_predictor.pkl"))

BOUNDS = {
    'floors':         (1, 30),
    'area_per_floor': (500, 2000),
    'material_grade': (1, 3),
    'structural_type':(1, 3),
    'facade_type':    (1, 3)
}

def decode(ind):
    return {
        'floors':          max(1, int(ind[0])),
        'area_per_floor':  max(500, int(ind[1])),
        'material_grade':  max(1, min(3, int(round(ind[2])))),
        'structural_type': max(1, min(3, int(round(ind[3])))),
        'facade_type':     max(1, min(3, int(round(ind[4]))))
    }

def fitness(individual, constraints):
    p = decode(individual)
    floor_bonus = p['floors'] * 5000
    
    total_area = p['floors'] * p['area_per_floor']

    diversity_penalty = 0
    if p['floors'] < 3:
        diversity_penalty += 50000
    if p['area_per_floor'] > 2000:
        diversity_penalty += 30000
        
    min_area = constraints.get('min_area', 1000)

    area_penalty = 0
    if total_area < min_area:
        area_penalty = (min_area - total_area) * 1000
        
    # ─── COST PREDICTION USING ML MODEL ───
    cost_features = [[
        total_area,                 # area
        p['material_grade'],        # building_type
        1.0,                        # location_factor 
        12,                         # timeline guess
        p['material_grade']         # quality_grade
    ]]

    total_cost = cost_model.predict(cost_features)[0]

    # ─── DELAY PREDICTION USING ML MODEL ───
    delay_features = [[
        1,   # weather_condition
        80,  # material_availability
        50,  # labor_count
        1,   # equipment_status
        2,   # project_complexity
        2    # site_accessibility
    ]]

    delay_pred = delay_model.predict(delay_features)[0]

    # timeline adjustment if delay predicted
    timeline = 12 * (1.3 if delay_pred == 1 else 1)

    safety_score = 60 + p['structural_type'] * 12 + p['material_grade'] * 5
    carbon = total_area * (0.8 - p['material_grade'] * 0.1 + p['structural_type'] * 0.15)

    budget = constraints.get('budget', float('inf'))
    cost_penalty = 0
    if budget and budget < float('inf'):
        cost_penalty = max(0, total_cost - budget) / budget

    score = -(total_cost * 0.4 +
          timeline * 0.3 +
          carbon * 0.2 -
          safety_score * 0.1 +
          cost_penalty * 100000 +
          area_penalty +
          diversity_penalty) +floor_bonus

    return (score,)

def run_genetic_algorithm(data):
    constraints = {
        'budget':     data.get('budget', float('inf')),
        'max_floors': data.get('max_floors', 20),
        'min_area':   data.get('area', 1000)
    }
    budget = constraints['budget']

    if 'FitnessMax' not in creator.__dict__:
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    if 'Individual' not in creator.__dict__:
        creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('individual', tools.initIterate, creator.Individual,
                     lambda: [random.uniform(b[0], b[1]) for b in BOUNDS.values()])
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', lambda ind: fitness(ind, constraints))
    toolbox.register('mate', tools.cxBlend, alpha=0.3)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=2, indpb=0.3)
    toolbox.register('select', tools.selTournament, tournsize=3)

    pop = toolbox.population(n=40)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.1, ngen=25, verbose=False)

    top3 = tools.selRandom(pop, k=3)
    designs = []
    labels = ['Design A', 'Design B', 'Design C']
    strategies = [
    {"cost_bias": -0.1, "time_bias": 0.1, "safety_bias": -0.05},
    {"cost_bias": 0.0,  "time_bias": 0.0, "safety_bias": 0.0},
    {"cost_bias": 0.1,  "time_bias": -0.1,"safety_bias": 0.1}
    ]
    
    for i, ind in enumerate(top3):
        p = decode(ind)
        strategy = strategies[i]
        total_area  = p['floors'] * p['area_per_floor']
        variation = 1 + random.uniform(-0.15, 0.15)+(i * 0.05)
        cost_features = [[
            total_area,                # area
            p['material_grade'],       # building_type
            1.0,                       # location_factor (default)
            12,                        # timeline guess
            p['material_grade']        # quality_grade
        ]]
        base_cost = cost_model.predict(cost_features)[0]
        cost = base_cost * (1 + strategy["cost_bias"]) * variation
        timeline_base = (6 + p['floors'] * 1.5 + p['material_grade'] * 2)
        timeline_mo = round(timeline_base * (1 + strategy["time_bias"]) * variation, 1)     
        safety_base = (60 + p['structural_type'] * 12 + p['material_grade'] * 5)
        safety_raw = round(safety_base * (1 + strategy["safety_bias"]) * variation, 1)   # 60–111
        carbon_raw  = round(total_area * (0.8 - p['material_grade'] * 0.1), 2)

        if budget and budget < float('inf'):
            cost_eff = max(0.0, round(1.0 - min(cost / budget, 1.5), 3))
        else:
            cost_eff = round(max(0.1, 1.0 - cost / 5e7), 3)   # relative to 50M default

        speed_score      = round(max(0.0, 1.0 - min(timeline_mo / 60.0, 1.0)), 3)
        safety_score_norm= round(min((safety_raw - 60) / 51.0, 1.0), 3)        # 60–111 → 0–1
        carbon_norm      = round(min(carbon_raw / max(total_area * 0.8, 1), 1.0), 3)  # relative to worst case
        quality_score    = round((p['material_grade'] / 3.0) * 0.7 + (p['structural_type'] / 3.0) * 0.3, 3)
        resilience_score = round((p['structural_type'] / 3.0) * 0.6 + (p['material_grade'] / 3.0) * 0.4, 3)

        # ── Cost breakdown percentages for donut chart ──
        labor_pct    = round(min(max(25 + p['floors'] * 0.5, 20), 40), 1)
        material_pct = round(min(max(35 + p['material_grade'] * 3, 30), 50), 1)
        equipment_pct= round(min(max(20 - p['floors'] * 0.2, 10), 25), 1)
        overhead_pct = round(max(100 - labor_pct - material_pct - equipment_pct, 5), 1)

        designs.append({
            # ── Basic info ──
            'id':              i,
            'label':           labels[i],
            'floors':          p['floors'],
            'area_per_floor':  p['area_per_floor'],
            'total_area':      total_area,
            'material_grade':  ['Standard', 'Premium', 'Luxury'][p['material_grade'] - 1],
            'structural_type': ['RCC', 'Steel', 'Composite'][p['structural_type'] - 1],
            'facade_type':     ['Brick', 'Glass', 'Cladding'][p['facade_type'] - 1],

            # ── Cost / time (raw) ──
            'estimated_cost':  round(cost, 2),
            'timeline_months': timeline_mo,
            'duration_days':   round(timeline_mo * 30),      # ← used in design cards & line chart

            # ── Safety / carbon (raw) ──
            'safety_score':    round(safety_raw / 111.0, 3), # normalised 0–1 for card display
            'carbon_footprint':carbon_raw,

            # ── Fitness ──
            'fitness':         round(ind.fitness.values[0], 2),
            'fitness_score':   round(max(0.0, min((ind.fitness.values[0] + 2e6) / 4e6, 1.0)), 3),

            # ── Radar chart scores (all 0–1) ──
            'cost_efficiency':  cost_eff,
            'speed_score':      speed_score,
            'safety_score_norm':safety_score_norm,
            'carbon_norm':      carbon_norm,
            'quality_score':    quality_score,
            'resilience_score': resilience_score,

            # ── Donut chart percentages ──
            'labor_pct':    labor_pct,
            'material_pct': material_pct,
            'equipment_pct':equipment_pct,
            'overhead_pct': overhead_pct,
        })

    return designs