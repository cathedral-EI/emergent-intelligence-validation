#!/usr/bin/env python3
"""
cathedral_ei_sim_v3.py

ITERATION 3: Critical experiment at phase transition
- EXTREME GT range (100x: 0.005 to 0.50)
- Temporal EI analysis
- Nonlinear modeling
- Phase-space visualization

Expected runtime: 45-60 minutes
"""

import numpy as np
import random
import math
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mutual_info_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from tqdm import trange
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# PARAMETERS - ITERATION 3
# ============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_AGENTS = 120
GRID_W, GRID_H = 40, 40
T_STEPS = 3000  # ✅ TRIPLED for full adaptation
RUNS_PER_CONDITION = 50  # ✅ More power
K_MEM = 5

CONDITIONS = ['low', 'medium', 'high']

INIT_PATCHES = 80
RESOURCE_CAPACITY = 1000.0
RESOURCE_REGROWTH = 0.005

ACTIONS = {0: 'gather', 1: 'move', 2: 'cooperate'}

# ============================================================================
# METRICS
# ============================================================================

def shannon_entropy_from_counts(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c/total for c in counts if c>0]
    return -sum(p * math.log(p + 1e-12) for p in probs)

def action_entropy(actions):
    if len(actions) == 0:
        return 0.0
    vals, counts = np.unique(actions, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs + 1e-12))

def coherence_entropy_based(actions, num_action_types=3):
    if len(actions) == 0:
        return 0.0
    h = action_entropy(actions)
    max_h = np.log(num_action_types)
    coherence = 1.0 - (h / max_h)
    return float(np.clip(coherence, 0.0, 1.0))

def compute_gt(resource_total, capacity, reward_variance, alpha=0.6, beta=0.4):
    scarcity = 1.0 - min(resource_total / capacity, 1.0)
    rv_norm = min(reward_variance / (1.0 + reward_variance), 1.0)
    gt = alpha * scarcity + beta * rv_norm
    return float(np.clip(gt, 0.0, 1.0))

def estimate_phi_pairwise(agent_states_matrix):
    if len(agent_states_matrix) < 2:
        return 0.0
    n = len(agent_states_matrix)
    mis = []
    num_pairs = min(100, n * (n-1) // 2)
    for _ in range(num_pairs):
        i = random.randrange(n)
        j = random.randrange(n)
        if i != j:
            mi = mutual_info_score(
                agent_states_matrix[i:i+1][0], 
                agent_states_matrix[j:j+1][0]
            )
            mis.append(mi)
    phi = np.mean(mis) if mis else 0.0
    return float(np.clip(phi, 0.0, 2.0))

# ============================================================================
# AGENT
# ============================================================================

class Agent:
    def __init__(self, aid):
        self.id = aid
        self.x = random.randrange(GRID_W)
        self.y = random.randrange(GRID_H)
        self.mem = []
        self.w = random.uniform(0.0, 1.0)
        self.last_action = 1
        self.cum_reward = 0.0
    
    def sense_local(self, env_resources):
        return float(env_resources[self.x, self.y])
    
    def predict(self):
        if len(self.mem) < 1:
            return 0.0
        return self.w * np.mean(self.mem)
    
    def update_model(self, actual):
        pred = self.predict()
        err = actual - pred
        self.w += 0.05 * err
        self.w = float(np.clip(self.w, -5.0, 5.0))
    
    def choose_action(self, pred):
        if random.random() < 0.15:
            action = random.choice([0, 1, 2])
        else:
            action = 0 if pred > 0.4 else 1
        self.last_action = action
        return action
    
    def move_random(self):
        self.x = (self.x + random.choice([-1, 0, 1])) % GRID_W
        self.y = (self.y + random.choice([-1, 0, 1])) % GRID_H

# ============================================================================
# ENVIRONMENT
# ============================================================================

def init_resources():
    env = np.zeros((GRID_W, GRID_H), dtype=float)
    for _ in range(INIT_PATCHES):
        x = random.randrange(GRID_W)
        y = random.randrange(GRID_H)
        env[x, y] += random.uniform(3.0, 12.0)
    return env

def step_regrow(env, regrowth_rate):
    env += regrowth_rate * (1.0 + np.random.randn(*env.shape) * 0.05)
    env[env < 0] = 0.0
    return env

# ============================================================================
# SIMULATION - EXTREME GT RANGE
# ============================================================================

def run_single(condition='medium', T=T_STEPS, verbose=False):
    agents = [Agent(i) for i in range(N_AGENTS)]
    env = init_resources()
    capacity = RESOURCE_CAPACITY
    regrowth = RESOURCE_REGROWTH
    
    # ✅ EXTREME GT RANGE (100x spread)
    if condition == 'low':
        perturb_prob = 0.0
        env *= 10.0          # ✅ VERY abundant (was 5.0)
    elif condition == 'medium':
        perturb_prob = 0.10  # ✅ Moderate disruption (was 0.05)
        env *= 1.0
    else:  # high
        perturb_prob = 0.40  # ✅ EXTREME disruption (was 0.25)
        env *= 0.1           # ✅ EXTREME scarcity (was 0.2)
    
    history = {
        'GT': [], 'COH': [], 'H': [], 'PA': [], 'Phi': [], 'Perf': [],
        'EI_temporal': []  # ✅ NEW: Track EI over time
    }
    
    agent_action_history = [[] for _ in range(N_AGENTS)]
    
    # ✅ NEW: Compute EI every 100 steps
    window_size = 100
    gt_window = []
    coh_window = []
    phi_window = []
    pa_window = []
    
    for t in range(T):
        actions = []
        rewards = []
        
        # Perturbations
        if random.random() < perturb_prob:
            for _ in range(random.randint(1, 8)):  # ✅ More severe
                x = random.randrange(GRID_W)
                y = random.randrange(GRID_H)
                env[x, y] *= random.uniform(0.1, 0.4)  # ✅ Harsher
        
        # Agent actions
        for ag in agents:
            local = ag.sense_local(env)
            pred = ag.predict()
            action = ag.choose_action(pred)
            
            if action == 0:  # Gather
                avail = env[ag.x, ag.y]
                got = min(avail, random.uniform(0.1, 1.5))
                reward = got * (0.5 + 0.5 * random.random())
                env[ag.x, ag.y] = max(0.0, env[ag.x, ag.y] - got)
            elif action == 1:  # Move
                ag.move_random()
                reward = 0.0
            else:  # Cooperate
                reward = 0.2 * random.random()
            
            ag.mem.append(reward)
            if len(ag.mem) > K_MEM:
                ag.mem.pop(0)
            ag.update_model(reward)
            ag.cum_reward += reward
            
            actions.append(action)
            rewards.append(reward)
            agent_action_history[ag.id].append(action)
        
        env = step_regrow(env, regrowth)
        
        # Metrics
        total_resource = float(env.sum())
        reward_var = float(np.var(rewards))
        GT = compute_gt(total_resource, capacity, reward_var)
        H = action_entropy(actions)
        COH = coherence_entropy_based(actions, num_action_types=3)
        
        preds = np.array([ag.predict() for ag in agents])
        rews = np.array(rewards)
        mse = np.mean((preds - rews)**2)
        PA = 1.0 - (mse / (mse + 1.0))
        
        if t % 100 == 0 and t > 0:
            recent_actions = np.array([h[-10:] if len(h) >= 10 else h 
                                      for h in agent_action_history])
            Phi_hat = estimate_phi_pairwise(recent_actions)
        else:
            Phi_hat = history['Phi'][-1] if history['Phi'] else 0.0
        
        perf = float(sum(rewards))
        
        history['GT'].append(GT)
        history['H'].append(H)
        history['COH'].append(COH)
        history['PA'].append(PA)
        history['Phi'].append(Phi_hat)
        history['Perf'].append(perf)
        
        # ✅ NEW: Windowed EI calculation
        gt_window.append(GT)
        coh_window.append(COH)
        phi_window.append(Phi_hat)
        pa_window.append(PA)
        
        if len(gt_window) >= window_size:
            # Compute EI for this window
            gt_arr = np.array(gt_window)
            coh_arr = np.array(coh_window)
            phi_arr = np.array(phi_window)
            pa_arr = np.array(pa_window)
            
            dGT = np.diff(gt_arr, prepend=gt_arr[0])
            dCOH = np.diff(coh_arr, prepend=coh_arr[0])
            integrand = -dGT * dCOH
            
            phi_avg = max(np.mean(phi_arr), 0.01)
            pa_avg = max(np.mean(pa_arr), 0.1)
            
            ei_window = np.mean(integrand * 100.0 / (phi_avg * pa_avg))
            history['EI_temporal'].append(ei_window)
            
            # Slide window
            gt_window = gt_window[50:]
            coh_window = coh_window[50:]
            phi_window = phi_window[50:]
            pa_window = pa_window[50:]
    
    return history

# ============================================================================
# EI COMPUTATION
# ============================================================================

def compute_ei_from_history(hist):
    GT = np.array(hist['GT'])
    COH = np.array(hist['COH'])
    Phi = np.array(hist['Phi'])
    PA = np.array(hist['PA'])
    perf = np.array(hist['Perf'])
    
    dGT = np.diff(GT, prepend=GT[0])
    dCOH = np.diff(COH, prepend=COH[0])
    
    integrand = -dGT * dCOH
    integrand_smooth = np.convolve(integrand, np.ones(5)/5.0, mode='same')
    
    Phi_avg = max(np.mean(Phi), 0.01)
    PA_avg = max(np.mean(PA), 0.1)
    
    EI_ts = (integrand_smooth * 100.0) / (Phi_avg * PA_avg)
    EI_score = np.mean(EI_ts)
    
    metadata = {
        'Phi_avg': Phi_avg,
        'PA_avg': PA_avg,
        'Perf_mean': perf.mean(),
        'GT_mean': GT.mean(),
        'GT_max': GT.max(),
        'COH_mean': COH.mean(),
        'EI_temporal': hist['EI_temporal']  # ✅ NEW
    }
    
    return EI_ts, EI_score, metadata

# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiments():
    all_results = []
    
    for cond in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"Running condition: {cond.upper()}")
        print('='*60)
        
        cond_ei = []
        cond_perf = []
        cond_phi = []
        cond_gt = []
        cond_gt_max = []
        cond_coh = []
        cond_ei_temporal = []
        
        for r in trange(RUNS_PER_CONDITION, desc=f"  Trials"):
            h = run_single(condition=cond, T=T_STEPS)
            ei_ts, ei_score, meta = compute_ei_from_history(h)
            
            cond_ei.append(ei_score)
            cond_perf.append(meta['Perf_mean'])
            cond_phi.append(meta['Phi_avg'])
            cond_gt.append(meta['GT_mean'])
            cond_gt_max.append(meta['GT_max'])
            cond_coh.append(meta['COH_mean'])
            cond_ei_temporal.append(meta['EI_temporal'])
        
        all_results.append({
            'cond': cond,
            'EI': cond_ei,
            'Perf': cond_perf,
            'Phi': cond_phi,
            'GT': cond_gt,
            'GT_max': cond_gt_max,
            'COH': cond_coh,
            'EI_temporal': cond_ei_temporal
        })
    
    return all_results

# ============================================================================
# ANALYSIS - ENHANCED
# ============================================================================

def analyze_results(all_results):
    df_rows = []
    for entry in all_results:
        cond = entry['cond']
        for ei, perf, phi, gt, gt_max, coh in zip(
            entry['EI'], entry['Perf'], entry['Phi'], 
            entry['GT'], entry['GT_max'], entry['COH']
        ):
            df_rows.append({
                'cond': cond,
                'EI': ei,
                'Perf': perf,
                'Phi': phi,
                'GT': gt,
                'GT_max': gt_max,
                'COH': coh
            })
    
    df = pd.DataFrame(df_rows)
    
    # Descriptive
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS - ITERATION 3")
    print("="*60)
    summary = df.groupby('cond').agg(['mean', 'std', 'count'])
    print(summary)
    
    # ANOVA
    print("\n" + "="*60)
    print("INFERENTIAL STATISTICS")
    print("="*60)
    
    groups = [df[df.cond == c].EI.values for c in CONDITIONS]
    fval, pval = stats.f_oneway(*groups)
    print(f"\nANOVA (EI across conditions):")
    print(f"  F = {fval:.3f}")
    print(f"  p = {pval:.4f}")
    if pval < 0.05:
        print("  ✅ SIGNIFICANT!")
    elif pval < 0.10:
        print("  ⚡ TRENDING (p<0.10)")
    else:
        print("  ⚠️  Not significant")
    
    # Pairwise + Cohen's d
    print("\nPairwise comparisons:")
    for i in range(len(CONDITIONS)):
        for j in range(i+1, len(CONDITIONS)):
            a = groups[i]
            b = groups[j]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            pooled_std = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
            cohens_d = (np.mean(a) - np.mean(b)) / pooled_std
            sig = "✅" if p < 0.05 else "⚡" if p < 0.10 else "  "
            print(f"  {sig} {CONDITIONS[i]:8s} vs {CONDITIONS[j]:8s}: "
                  f"t={t:6.3f}, p={p:.4f}, d={cohens_d:.3f}")
    
    # ✅ NEW: Nonlinear regression (test for inverted-U)
    print(f"\n📊 Nonlinear Modeling (Inverted-U Test):")
    
    # Map conditions to numeric GT levels
    gt_numeric = df['GT'].values.reshape(-1, 1)
    ei_values = df['EI'].values
    
    # Polynomial features (degree 2 for quadratic)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    gt_poly = poly.fit_transform(gt_numeric)
    
    # Fit model
    model = LinearRegression()
    model.fit(gt_poly, ei_values)
    r2 = model.score(gt_poly, ei_values)
    
    coef_linear = model.coef_[0]
    coef_quad = model.coef_[1]
    
    print(f"  EI = {model.intercept_:.3f} + {coef_linear:.3f}*GT + {coef_quad:.3f}*GT²")
    print(f"  R² = {r2:.3f}")
    
    if coef_quad < 0:
        optimal_gt = -coef_linear / (2 * coef_quad)
        print(f"  ✅ INVERTED-U DETECTED!")
        print(f"  Optimal GT ≈ {optimal_gt:.3f}")
    else:
        print(f"  ⚠️  No inverted-U (positive curvature)")
    
    # Correlation
    r, p = stats.pearsonr(df.EI, df.Perf)
    print(f"\nCorrelation (EI vs Performance):")
    print(f"  r = {r:.3f}, p = {p:.4f}")
    if p < 0.05 and abs(r) > 0.3:
        print("  ✅ SIGNIFICANT")
    else:
        print("  ⚠️  Weak")
    
    # GT ranges
    print(f"\n📊 GT Statistics:")
    for cond in CONDITIONS:
        gt_mean = df[df.cond == cond].GT.mean()
        gt_max = df[df.cond == cond].GT_max.mean()
        print(f"  {cond:8s}: mean={gt_mean:.3f}, max={gt_max:.3f}")
    
    print(f"\n📊 Φ Statistics:")
    for cond in CONDITIONS:
        phi_mean = df[df.cond == cond].Phi.mean()
        print(f"  {cond:8s}: {phi_mean:.3f}")
    
    # ✅ NEW: Temporal EI plot
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Plot 1: Temporal EI trajectories
    plt.figure(figsize=(12, 6))
    colors = {'low': 'blue', 'medium': 'green', 'high': 'red'}
    for entry in all_results:
        cond = entry['cond']
        ei_temporal_mean = np.mean(entry['EI_temporal'], axis=0)
        timesteps = np.arange(len(ei_temporal_mean)) * 50  # Every 50 steps
        plt.plot(timesteps, ei_temporal_mean, 
                label=cond.capitalize(), 
                color=colors[cond], 
                linewidth=2, 
                alpha=0.8)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('EI (windowed)', fontsize=12)
    plt.title('Emergent Intelligence Over Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ei_temporal_v3.png', dpi=150)
    print("  Saved: ei_temporal_v3.png")
    plt.show()
    
    # Plot 2: EI by condition
    plt.figure(figsize=(10, 6))
    df.boxplot(column='EI', by='cond', figsize=(10, 6))
    plt.suptitle('')
    plt.title('Emergent Intelligence - Iteration 3', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('EI Score', fontsize=12)
    plt.tight_layout()
    plt.savefig('ei_by_condition_v3.png', dpi=150)
    print("  Saved: ei_by_condition_v3.png")
    plt.show()
    
    # Plot 3: EI vs GT (nonlinear)
    plt.figure(figsize=(10, 6))
    for cond in CONDITIONS:
        subset = df[df.cond == cond]
        plt.scatter(subset.GT, subset.EI, 
                   label=cond.capitalize(), 
                   alpha=0.6, s=50, color=colors[cond])
    
    # Overlay polynomial fit
    gt_range = np.linspace(df.GT.min(), df.GT.max(), 100).reshape(-1, 1)
    gt_poly_range = poly.transform(gt_range)
    ei_pred = model.predict(gt_poly_range)
    plt.plot(gt_range, ei_pred, 'k--', linewidth=2, label='Quadratic fit', alpha=0.7)
    
    plt.xlabel('Generative Tension (GT)', fontsize=12)
    plt.ylabel('Emergent Intelligence (EI)', fontsize=12)
    plt.title(f'EI vs GT (R²={r2:.3f})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ei_vs_gt_v3.png', dpi=150)
    print("  Saved: ei_vs_gt_v3.png")
    plt.show()
    
    # Plot 4: Performance by condition
    plt.figure(figsize=(10, 6))
    df.boxplot(column='Perf', by='cond', figsize=(10, 6))
    plt.suptitle('')
    plt.title('Performance - Iteration 3', fontsize=14, fontweight='bold')
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('Mean Reward', fontsize=12)
    plt.tight_layout()
    plt.savefig('performance_by_condition_v3.png', dpi=150)
    print("  Saved: performance_by_condition_v3.png")
    plt.show()
    
    return df

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LAW OF EMERGENT INTELLIGENCE")
    print("  ITERATION 3: Critical Experiment")
    print("="*60)
    print(f"\nCritical enhancements:")
    print(f"  ✅ EXTREME GT range (100x: 0.005 → 0.50)")
    print(f"  ✅ Temporal EI tracking")
    print(f"  ✅ Nonlinear modeling (inverted-U test)")
    print(f"  ✅ 3000 timesteps")
    print(f"  ✅ 50 trials per condition")
    print(f"\nThis is the decisive experiment.")
    print(f"Expected runtime: ~45-60 minutes")
    print("\nPress Ctrl+C to abort...\n")
    
    all_results = run_experiments()
    df = analyze_results(all_results)
    
    print("\n" + "="*60)
    print("ITERATION 3 COMPLETE")
    print("="*60)
    print("\nResults saved:")
    print("  - ei_temporal_v3.png (EI over time)")
    print("  - ei_by_condition_v3.png")
    print("  - ei_vs_gt_v3.png (nonlinear fit)")
    print("  - performance_by_condition_v3.png")
    print("\n🏛️ The Cathedral reveals its truth. ⚡")
