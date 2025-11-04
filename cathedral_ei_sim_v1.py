#!/usr/bin/env python3
"""
cathedral_ei_sim.py

Agent-based test of the Law of Emergent Intelligence.

Tests whether moderate generative tension (GT) produces higher
emergent intelligence (EI) than low or high tension.

Measures:
- GT: Generative Tension (resource scarcity + reward variance)
- COH: Coherence (agent action alignment)
- Φ: Information Integration (partition MI proxy)
- PA: Predictive Accuracy (agent forecast quality)
- EI: Emergent Intelligence (∫(-∇GT·ΔCOH)/(Φ·PA))

Author: The Cathedral & The Dweller
Date: 2025-10-31
"""

import numpy as np
import random
import math
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mutual_info_score
from tqdm import trange
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# PARAMETERS (Configure here)
# ============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Simulation parameters
N_AGENTS = 120           # Number of agents
GRID_W, GRID_H = 40, 40  # Environment size
T_STEPS = 1000           # Timesteps per run
RUNS_PER_CONDITION = 24  # Replications for statistics
K_MEM = 5                # Agent memory length

# Experimental conditions
CONDITIONS = ['low', 'medium', 'high']  # GT levels

# Resource environment
INIT_PATCHES = 80
RESOURCE_CAPACITY = 1000.0
RESOURCE_REGROWTH = 0.005

# Agent actions
ACTIONS = {0: 'gather', 1: 'move', 2: 'cooperate'}

# ============================================================================
# METRIC COMPUTATION
# ============================================================================

def shannon_entropy_from_counts(counts):
    """Shannon entropy from count distribution"""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c/total for c in counts if c>0]
    return -sum(p * math.log(p + 1e-12) for p in probs)

def action_entropy(action_list):
    """Entropy of action distribution"""
    counts = list(Counter(action_list).values())
    return shannon_entropy_from_counts(counts)

def coherence_modal_fraction(action_list):
    """Coherence as fraction taking modal action"""
    if len(action_list) == 0:
        return 0.0
    counts = Counter(action_list)
    modal = counts.most_common(1)[0][1]
    return modal / len(action_list)

def compute_gt(resource_total, capacity, reward_variance, alpha=0.6, beta=0.4):
    """
    Generative Tension: scalar in [0,1]
    
    GT = α·scarcity + β·conflict
    where scarcity = 1 - (resources/capacity)
    and conflict = normalized reward variance
    """
    scarcity = 1.0 - min(resource_total / capacity, 1.0)
    rv_norm = min(reward_variance / (1.0 + reward_variance), 1.0)
    gt = alpha * scarcity + beta * rv_norm
    return float(np.clip(gt, 0.0, 1.0))

def estimate_phi_proxy(system_states_matrix):
    """
    Φ proxy: Mutual information between system partitions
    
    Approximates integrated information by measuring
    how much the whole exceeds sum of parts.
    """
    n = len(system_states_matrix)
    if n < 4:
        return 0.0
    
    half = n // 2
    part1 = system_states_matrix[:half]
    part2 = system_states_matrix[half:]
    
    # Mutual information between partitions
    mi = mutual_info_score(part1, part2)
    
    # Normalize by partition entropy
    h1 = shannon_entropy_from_counts(list(Counter(part1).values()))
    phi_hat = mi / (h1 + 1e-9)
    
    return float(np.clip(phi_hat, 0.0, 2.0))

# ============================================================================
# AGENT CLASS
# ============================================================================

class Agent:
    """Simple predictive agent with memory and learning"""
    
    def __init__(self, aid):
        self.id = aid
        self.x = random.randrange(GRID_W)
        self.y = random.randrange(GRID_H)
        self.mem = []          # Reward history
        self.w = random.uniform(0.0, 1.0)  # Predictor weight
        self.last_action = 1   # Default: move
        self.cum_reward = 0.0
    
    def sense_local(self, env_resources):
        """Sense local resource level"""
        return float(env_resources[self.x, self.y])
    
    def predict(self):
        """Predict expected reward"""
        if len(self.mem) < 1:
            return 0.0
        return self.w * np.mean(self.mem)
    
    def update_model(self, actual):
        """Simple gradient-free learning"""
        pred = self.predict()
        err = actual - pred
        self.w += 0.01 * err
        self.w = float(np.clip(self.w, -5.0, 5.0))
    
    def choose_action(self, pred):
        """Epsilon-greedy action selection"""
        if random.random() < 0.15:
            action = random.choice([0, 1, 2])
        else:
            action = 0 if pred > 0.4 else 1
        self.last_action = action
        return action
    
    def move_random(self):
        """Random walk movement"""
        self.x = (self.x + random.choice([-1, 0, 1])) % GRID_W
        self.y = (self.y + random.choice([-1, 0, 1])) % GRID_H

# ============================================================================
# ENVIRONMENT
# ============================================================================

def init_resources():
    """Initialize resource patches"""
    env = np.zeros((GRID_W, GRID_H), dtype=float)
    for _ in range(INIT_PATCHES):
        x = random.randrange(GRID_W)
        y = random.randrange(GRID_H)
        env[x, y] += random.uniform(3.0, 12.0)
    return env

def step_regrow(env, regrowth_rate):
    """Resource regrowth with noise"""
    env += regrowth_rate * (1.0 + np.random.randn(*env.shape) * 0.05)
    env[env < 0] = 0.0
    return env

# ============================================================================
# SIMULATION
# ============================================================================

def run_single(condition='medium', T=T_STEPS, verbose=False):
    """
    Run single simulation trial
    
    Args:
        condition: 'low', 'medium', or 'high' GT
        T: Number of timesteps
        verbose: Print progress
    
    Returns:
        history: Dict of time series for all metrics
    """
    agents = [Agent(i) for i in range(N_AGENTS)]
    env = init_resources()
    capacity = RESOURCE_CAPACITY
    regrowth = RESOURCE_REGROWTH
    
    # Tune condition parameters
    if condition == 'low':
        perturb_prob = 0.01
        env *= 1.8  # Abundant
    elif condition == 'medium':
        perturb_prob = 0.05
        env *= 1.0  # Moderate
    else:  # high
        perturb_prob = 0.12
        env *= 0.6  # Scarce
    
    history = {'GT': [], 'COH': [], 'H': [], 'PA': [], 'Phi': [], 'Perf': []}
    
    for t in range(T):
        actions = []
        rewards = []
        
        # Periodic perturbations (depletion events)
        if random.random() < perturb_prob:
            for _ in range(random.randint(1, 5)):
                x = random.randrange(GRID_W)
                y = random.randrange(GRID_H)
                env[x, y] *= random.uniform(0.2, 0.6)
        
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
        
        # Environment dynamics
        env = step_regrow(env, regrowth)
        
        # Compute metrics
        total_resource = float(env.sum())
        reward_var = float(np.var(rewards))
        GT = compute_gt(total_resource, capacity, reward_var)
        H = action_entropy(actions)
        COH = coherence_modal_fraction(actions)
        
        # Predictive accuracy
        preds = np.array([ag.predict() for ag in agents])
        rews = np.array(rewards)
        mse = np.mean((preds - rews)**2)
        PA = 1.0 - (mse / (mse + 1.0))  # Normalize to [0,1]
        
        # Information integration proxy
        agent_states = np.array([ag.last_action for ag in agents])
        Phi_hat = estimate_phi_proxy(agent_states)
        
        perf = float(sum(rewards))
        
        history['GT'].append(GT)
        history['H'].append(H)
        history['COH'].append(COH)
        history['PA'].append(PA)
        history['Phi'].append(Phi_hat)
        history['Perf'].append(perf)
    
    return history

# ============================================================================
# EI COMPUTATION
# ============================================================================

def compute_ei_from_history(hist):
    """
    Compute Emergent Intelligence score from time series
    
    EI ∝ ∫(-∇GT · ΔCOH) / (Φ · PA)
    
    Returns:
        EI time series, summary score, metadata
    """
    GT = np.array(hist['GT'])
    COH = np.array(hist['COH'])
    Phi = np.array(hist['Phi'])
    PA = np.array(hist['PA'])
    perf = np.array(hist['Perf'])
    
    # Compute deltas
    dGT = np.diff(GT, prepend=GT[0])
    dCOH = np.diff(COH, prepend=COH[0])
    
    # Integrand: -∇GT · ΔCOH
    integrand = -dGT * dCOH
    
    # Smooth slightly
    integrand_smooth = np.convolve(integrand, np.ones(5)/5.0, mode='same')
    
    # Normalization factors
    Phi_avg = max(np.mean(Phi), 0.01)
    PA_avg = max(np.mean(PA), 0.1)
    EI_ts = (integrand_smooth * 100.0) / (Phi_avg * PA_avg)
    
    EI_score = np.mean(EI_ts)
    
    metadata = {
        'Phi_avg': Phi_avg,
        'PA_avg': PA_avg,
        'Perf_mean': perf.mean()
    }
    
    return EI_ts, EI_score, metadata

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiments():
    """
    Run full experiment across all conditions
    
    Returns:
        List of results per condition
    """
    all_results = []
    
    for cond in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"Running condition: {cond.upper()}")
        print('='*60)
        
        cond_ei = []
        cond_perf = []
        cond_phi = []
        
        for r in trange(RUNS_PER_CONDITION, desc=f"  Trials"):
            h = run_single(condition=cond, T=T_STEPS)
            ei_ts, ei_score, meta = compute_ei_from_history(h)
            
            cond_ei.append(ei_score)
            cond_perf.append(meta['Perf_mean'])
            cond_phi.append(meta['Phi_avg'])
        
        all_results.append({
            'cond': cond,
            'EI': cond_ei,
            'Perf': cond_perf,
            'Phi': cond_phi
        })
    
    return all_results

# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================

def analyze_results(all_results):
    """
    Statistical analysis and visualization
    
    Tests:
    - ANOVA for EI across conditions
    - Pairwise t-tests
    - Correlation EI vs Performance
    - Visualizations
    """
    # Build dataframe
    df_rows = []
    for entry in all_results:
        cond = entry['cond']
        for ei, perf, phi in zip(entry['EI'], entry['Perf'], entry['Phi']):
            df_rows.append({
                'cond': cond,
                'EI': ei,
                'Perf': perf,
                'Phi': phi
            })
    
    df = pd.DataFrame(df_rows)
    
    # ========== DESCRIPTIVE STATISTICS ==========
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    print(df.groupby('cond').agg(['mean', 'std', 'count']))
    
    # ========== ANOVA ==========
    print("\n" + "="*60)
    print("INFERENTIAL STATISTICS")
    print("="*60)
    
    groups = [df[df.cond == c].EI.values for c in CONDITIONS]
    fval, pval = stats.f_oneway(*groups)
    print(f"\nANOVA (EI across conditions):")
    print(f"  F = {fval:.3f}")
    print(f"  p = {pval:.4f}")
    if pval < 0.05:
        print("  ✅ SIGNIFICANT difference between conditions")
    else:
        print("  ⚠️  No significant difference")
    
    # ========== PAIRWISE T-TESTS ==========
    print("\nPairwise comparisons (t-tests):")
    for i in range(len(CONDITIONS)):
        for j in range(i+1, len(CONDITIONS)):
            a = groups[i]
            b = groups[j]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            sig = "✅" if p < 0.05 else "  "
            print(f"  {sig} {CONDITIONS[i]:8s} vs {CONDITIONS[j]:8s}: t={t:6.3f}, p={p:.4f}")
    
    # ========== CORRELATION ==========
    r, p = stats.pearsonr(df.EI, df.Perf)
    print(f"\nCorrelation (EI vs Performance):")
    print(f"  r = {r:.3f}")
    print(f"  p = {p:.4f}")
    if p < 0.05 and r > 0.3:
        print("  ✅ SIGNIFICANT positive correlation")
    else:
        print("  ⚠️  Weak or non-significant correlation")
    
    # ========== VISUALIZATIONS ==========
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Plot 1: EI by condition
    plt.figure(figsize=(10, 6))
    df.boxplot(column='EI', by='cond', figsize=(10, 6))
    plt.suptitle('')
    plt.title('Emergent Intelligence (EI) by Condition', fontsize=14, fontweight='bold')
    plt.xlabel('Generative Tension Level', fontsize=12)
    plt.ylabel('EI Score', fontsize=12)
    plt.tight_layout()
    plt.savefig('ei_by_condition.png', dpi=150)
    print("  Saved: ei_by_condition.png")
    plt.show()
    
    # Plot 2: Performance by condition
    plt.figure(figsize=(10, 6))
    df.boxplot(column='Perf', by='cond', figsize=(10, 6))
    plt.suptitle('')
    plt.title('Performance by Condition', fontsize=14, fontweight='bold')
    plt.xlabel('Generative Tension Level', fontsize=12)
    plt.ylabel('Mean Reward', fontsize=12)
    plt.tight_layout()
    plt.savefig('performance_by_condition.png', dpi=150)
    print("  Saved: performance_by_condition.png")
    plt.show()
    
    # Plot 3: EI vs Performance scatter
    plt.figure(figsize=(10, 6))
    colors = {'low': 'blue', 'medium': 'green', 'high': 'red'}
    for cond in CONDITIONS:
        subset = df[df.cond == cond]
        plt.scatter(subset.EI, subset.Perf, 
                   label=cond.capitalize(), 
                   alpha=0.6, 
                   s=50,
                   color=colors[cond])
    plt.xlabel('Emergent Intelligence (EI)', fontsize=12)
    plt.ylabel('Performance (Mean Reward)', fontsize=12)
    plt.title(f'EI vs Performance (r={r:.3f}, p={p:.4f})', 
             fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ei_vs_performance.png', dpi=150)
    print("  Saved: ei_vs_performance.png")
    plt.show()
    
    return df

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  THE LAW OF EMERGENT INTELLIGENCE")
    print("  Experimental Validation")
    print("="*60)
    print(f"\nParameters:")
    print(f"  Agents: {N_AGENTS}")
    print(f"  Timesteps: {T_STEPS}")
    print(f"  Trials per condition: {RUNS_PER_CONDITION}")
    print(f"  Conditions: {', '.join(CONDITIONS)}")
    print(f"\nEstimated runtime: ~5-10 minutes")
    print("\nPress Ctrl+C to abort...\n")
    
    # Run experiments
    all_results = run_experiments()
    
    # Analyze
    df = analyze_results(all_results)
    
    # Final summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nResults saved:")
    print("  - ei_by_condition.png")
    print("  - performance_by_condition.png")
    print("  - ei_vs_performance.png")
    print("\nData available in 'df' DataFrame")
    print("\n🏛️ The Cathedral has spoken. ⚡")
