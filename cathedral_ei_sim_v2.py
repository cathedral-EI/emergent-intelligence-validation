#!/usr/bin/env python3
"""
cathedral_ei_sim_v2.py

ITERATION 2: Enhanced experiment with:
- Wider GT range (5.0x vs 0.2x)
- Better Φ measurement (pairwise MI)
- Entropy-based coherence
- Faster learning (5x)
- Longer simulation (2000 steps)
- More trials (40 per condition)

Expected runtime: 20-30 minutes
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
# PARAMETERS - UPDATED FOR ITERATION 2
# ============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_AGENTS = 120
GRID_W, GRID_H = 40, 40
T_STEPS = 2000  # ✅ DOUBLED (was 1000)
RUNS_PER_CONDITION = 40  # ✅ INCREASED (was 24)
K_MEM = 5

CONDITIONS = ['low', 'medium', 'high']

INIT_PATCHES = 80
RESOURCE_CAPACITY = 1000.0
RESOURCE_REGROWTH = 0.005

ACTIONS = {0: 'gather', 1: 'move', 2: 'cooperate'}

# ============================================================================
# IMPROVED METRICS
# ============================================================================

def shannon_entropy_from_counts(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c/total for c in counts if c>0]
    return -sum(p * math.log(p + 1e-12) for p in probs)

def action_entropy(actions):
    """Entropy of action distribution"""
    if len(actions) == 0:
        return 0.0
    vals, counts = np.unique(actions, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs + 1e-12))

def coherence_entropy_based(actions, num_action_types=3):
    """
    ✅ NEW: Entropy-based coherence
    
    Coherence = 1 - (entropy / max_entropy)
    High coherence = organized (low entropy)
    Low coherence = random (high entropy)
    """
    if len(actions) == 0:
        return 0.0
    
    h = action_entropy(actions)
    max_h = np.log(num_action_types)
    coherence = 1.0 - (h / max_h)
    
    return float(np.clip(coherence, 0.0, 1.0))

def compute_gt(resource_total, capacity, reward_variance, alpha=0.6, beta=0.4):
    """GT = α·scarcity + β·conflict"""
    scarcity = 1.0 - min(resource_total / capacity, 1.0)
    rv_norm = min(reward_variance / (1.0 + reward_variance), 1.0)
    gt = alpha * scarcity + beta * rv_norm
    return float(np.clip(gt, 0.0, 1.0))

def estimate_phi_pairwise(agent_states_matrix):
    """
    ✅ NEW: Pairwise MI as Φ proxy
    
    More sensitive than partition-based MI
    Measures coordination between all agent pairs
    """
    if len(agent_states_matrix) < 2:
        return 0.0
    
    n = len(agent_states_matrix)
    mis = []
    
    # Sample pairs (all pairs for n=120 is 7140 - too many, sample 100)
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
# AGENT - FASTER LEARNING
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
        """✅ FASTER: 5x learning rate"""
        pred = self.predict()
        err = actual - pred
        self.w += 0.05 * err  # Was 0.01
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
# SIMULATION - WIDER GT RANGE
# ============================================================================

def run_single(condition='medium', T=T_STEPS, verbose=False):
    agents = [Agent(i) for i in range(N_AGENTS)]
    env = init_resources()
    capacity = RESOURCE_CAPACITY
    regrowth = RESOURCE_REGROWTH
    
    # ✅ MUCH WIDER GT RANGE
    if condition == 'low':
        perturb_prob = 0.0   # No perturbations
        env *= 5.0           # Very abundant (was 1.8)
    elif condition == 'medium':
        perturb_prob = 0.05
        env *= 1.0
    else:  # high
        perturb_prob = 0.25  # Frequent shocks (was 0.12)
        env *= 0.2           # Very scarce (was 0.6)
    
    history = {'GT': [], 'COH': [], 'H': [], 'PA': [], 'Phi': [], 'Perf': []}
    
    # Store agent action history for Φ calculation
    agent_action_history = [[] for _ in range(N_AGENTS)]
    
    for t in range(T):
        actions = []
        rewards = []
        
        # Perturbations
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
            agent_action_history[ag.id].append(action)
        
        # Environment dynamics
        env = step_regrow(env, regrowth)
        
        # Compute metrics
        total_resource = float(env.sum())
        reward_var = float(np.var(rewards))
        GT = compute_gt(total_resource, capacity, reward_var)
        H = action_entropy(actions)
        COH = coherence_entropy_based(actions, num_action_types=3)  # ✅ NEW
        
        # Predictive accuracy
        preds = np.array([ag.predict() for ag in agents])
        rews = np.array(rewards)
        mse = np.mean((preds - rews)**2)
        PA = 1.0 - (mse / (mse + 1.0))
        
        # ✅ NEW: Pairwise MI for Φ (compute every 100 steps to save time)
        if t % 100 == 0 and t > 0:
            # Take last 10 timesteps of agent actions
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
    
    return history

# ============================================================================
# EI COMPUTATION - IMPROVED NORMALIZATION
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
    
    # ✅ IMPROVED: Better guards and scaling
    Phi_avg = max(np.mean(Phi), 0.01)
    PA_avg = max(np.mean(PA), 0.1)
    
    EI_ts = (integrand_smooth * 100.0) / (Phi_avg * PA_avg)
    EI_score = np.mean(EI_ts)
    
    metadata = {
        'Phi_avg': Phi_avg,
        'PA_avg': PA_avg,
        'Perf_mean': perf.mean(),
        'GT_mean': GT.mean(),
        'COH_mean': COH.mean()
    }
    
    return EI_ts, EI_score, metadata

# ============================================================================
# EXPERIMENT RUNNER
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
        cond_coh = []
        
        for r in trange(RUNS_PER_CONDITION, desc=f"  Trials"):
            h = run_single(condition=cond, T=T_STEPS)
            ei_ts, ei_score, meta = compute_ei_from_history(h)
            
            cond_ei.append(ei_score)
            cond_perf.append(meta['Perf_mean'])
            cond_phi.append(meta['Phi_avg'])
            cond_gt.append(meta['GT_mean'])
            cond_coh.append(meta['COH_mean'])
        
        all_results.append({
            'cond': cond,
            'EI': cond_ei,
            'Perf': cond_perf,
            'Phi': cond_phi,
            'GT': cond_gt,
            'COH': cond_coh
        })
    
    return all_results

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_results(all_results):
    df_rows = []
    for entry in all_results:
        cond = entry['cond']
        for ei, perf, phi, gt, coh in zip(
            entry['EI'], entry['Perf'], entry['Phi'], 
            entry['GT'], entry['COH']
        ):
            df_rows.append({
                'cond': cond,
                'EI': ei,
                'Perf': perf,
                'Phi': phi,
                'GT': gt,
                'COH': coh
            })
    
    df = pd.DataFrame(df_rows)
    
    # Descriptive
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS - ITERATION 2")
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
        print("  ✅ SIGNIFICANT difference!")
    else:
        print("  ⚠️  No significant difference")
    
    # Pairwise + Cohen's d
    print("\nPairwise comparisons:")
    for i in range(len(CONDITIONS)):
        for j in range(i+1, len(CONDITIONS)):
            a = groups[i]
            b = groups[j]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            
            # Cohen's d
            pooled_std = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
            cohens_d = (np.mean(a) - np.mean(b)) / pooled_std
            
            sig = "✅" if p < 0.05 else "  "
            print(f"  {sig} {CONDITIONS[i]:8s} vs {CONDITIONS[j]:8s}: "
                  f"t={t:6.3f}, p={p:.4f}, d={cohens_d:.3f}")
    
    # Correlation
    r, p = stats.pearsonr(df.EI, df.Perf)
    print(f"\nCorrelation (EI vs Performance):")
    print(f"  r = {r:.3f}")
    print(f"  p = {p:.4f}")
    if p < 0.05 and abs(r) > 0.3:
        print("  ✅ SIGNIFICANT correlation")
    else:
        print("  ⚠️  Weak or non-significant")
    
    # Additional metrics
    print(f"\n📊 Mean GT by condition:")
    for cond in CONDITIONS:
        gt_mean = df[df.cond == cond].GT.mean()
        print(f"  {cond:8s}: {gt_mean:.3f}")
    
    print(f"\n📊 Mean Φ by condition:")
    for cond in CONDITIONS:
        phi_mean = df[df.cond == cond].Phi.mean()
        print(f"  {cond:8s}: {phi_mean:.3f}")
    
    # Plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    plt.figure(figsize=(10, 6))
    df.boxplot(column='EI', by='cond', figsize=(10, 6))
    plt.suptitle('')
    plt.title('Emergent Intelligence - Iteration 2', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('EI Score', fontsize=12)
    plt.tight_layout()
    plt.savefig('ei_by_condition_v2.png', dpi=150)
    print("  Saved: ei_by_condition_v2.png")
    plt.show()
    
    plt.figure(figsize=(10, 6))
    df.boxplot(column='Perf', by='cond', figsize=(10, 6))
    plt.suptitle('')
    plt.title('Performance - Iteration 2', fontsize=14, fontweight='bold')
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('Mean Reward', fontsize=12)
    plt.tight_layout()
    plt.savefig('performance_by_condition_v2.png', dpi=150)
    print("  Saved: performance_by_condition_v2.png")
    plt.show()
    
    plt.figure(figsize=(10, 6))
    colors = {'low': 'blue', 'medium': 'green', 'high': 'red'}
    for cond in CONDITIONS:
        subset = df[df.cond == cond]
        plt.scatter(subset.EI, subset.Perf, 
                   label=cond.capitalize(), 
                   alpha=0.6, s=50, color=colors[cond])
    plt.xlabel('Emergent Intelligence (EI)', fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    plt.title(f'EI vs Performance (r={r:.3f}, p={p:.4f})', 
             fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ei_vs_performance_v2.png', dpi=150)
    print("  Saved: ei_vs_performance_v2.png")
    plt.show()
    
    return df

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LAW OF EMERGENT INTELLIGENCE")
    print("  ITERATION 2: Enhanced Experiment")
    print("="*60)
    print(f"\nEnhancements:")
    print(f"  ✅ Wider GT range (5.0x vs 0.2x)")
    print(f"  ✅ Pairwise MI for Φ")
    print(f"  ✅ Entropy-based coherence")
    print(f"  ✅ 5x faster learning")
    print(f"  ✅ 2000 timesteps (doubled)")
    print(f"  ✅ 40 trials per condition")
    print(f"\nParameters:")
    print(f"  Agents: {N_AGENTS}")
    print(f"  Timesteps: {T_STEPS}")
    print(f"  Trials per condition: {RUNS_PER_CONDITION}")
    print(f"\nEstimated runtime: ~20-30 minutes")
    print("\nPress Ctrl+C to abort...\n")
    
    all_results = run_experiments()
    df = analyze_results(all_results)
    
    print("\n" + "="*60)
    print("ITERATION 2 COMPLETE")
    print("="*60)
    print("\nResults saved:")
    print("  - ei_by_condition_v2.png")
    print("  - performance_by_condition_v2.png")
    print("  - ei_vs_performance_v2.png")
    print("\n🏛️ The Cathedral learns through iteration. ⚡")
