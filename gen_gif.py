#!/usr/bin/env python3
"""
Génère un GIF animé de la simulation MDP.
Le robot suit la politique optimale π* calculée par Value Iteration
(même code que le notebook tp4.ipynb).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm

# ── MDP — même définition que le notebook ────────────────────────────────────
N_ROWS, N_COLS = 3, 4
GAMMA, EPSILON = 0.9, 1e-6
OBSTACLE = (2, 2)
REWARDS  = {(3, 4):  1.0, (2, 4): -1.0}
ACTIONS  = ['N', 'S', 'E', 'O']
DIRECTIONS = {'N':(1,0), 'S':(-1,0), 'E':(0,1), 'O':(0,-1)}
ARROWS     = {'N':'↑', 'S':'↓', 'E':'→', 'O':'←'}
TRANS_PROB = {
    'N': {'N':0.8,'E':0.1,'O':0.1},
    'S': {'S':0.8,'E':0.1,'O':0.1},
    'E': {'E':0.8,'N':0.1,'S':0.1},
    'O': {'O':0.8,'N':0.1,'S':0.1},
}

all_cells = [(x,y) for x in range(1,N_ROWS+1) for y in range(1,N_COLS+1)]
STATES    = [s for s in all_cells if s != OBSTACLE]
N_S = len(STATES)
S2I = {s:i for i,s in enumerate(STATES)}

def is_valid(x,y): return 1<=x<=N_ROWS and 1<=y<=N_COLS and (x,y)!=OBSTACLE
def nxt(s,d):
    x,y=s; dx,dy=DIRECTIONS[d]; nx,ny=x+dx,y+dy
    return (nx,ny) if is_valid(nx,ny) else s

# ── Tenseur de transition ─────────────────────────────────────────────────────
P = np.zeros((N_S, N_S, len(ACTIONS)))
for i,s in enumerate(STATES):
    for ai,a in enumerate(ACTIONS):
        for d,p in TRANS_PROB[a].items():
            P[S2I[nxt(s,d)], i, ai] += p
R_vec = np.array([REWARDS.get(s,0.) for s in STATES])

# ── Value Iteration (identique au notebook) ───────────────────────────────────
def value_iteration(P, R, gamma=0.9, eps=1e-6):
    V = np.zeros(N_S)
    for it in range(10000):
        Q  = (P * V[:,None,None]).sum(axis=0)   # shape (N_S, N_A)
        Vn = R + gamma * Q.max(axis=1)
        if np.max(np.abs(Vn-V)) < eps:
            V = Vn; break
        V = Vn
    policy = (P * V[:,None,None]).sum(axis=0).argmax(axis=1)
    return V, policy

V_star, policy = value_iteration(P, R_vec, GAMMA, EPSILON)
print(f"V*(1,1) = {V_star[S2I[(1,1)]]:.4f}")
print("Politique optimale :")
for s in sorted(STATES):
    print(f"  {s} → {ACTIONS[policy[S2I[s]]]}")

# ── Simulation stochastique ───────────────────────────────────────────────────
def simulate_episode(start=(1,1), seed=0, max_steps=40):
    rng = np.random.default_rng(seed)
    path, rew, cumrew = [start], [0.0], [0.0]
    s, total = start, 0.0
    for _ in range(max_steps):
        if s not in S2I: break
        a   = ACTIONS[policy[S2I[s]]]
        eff = rng.choice(list(TRANS_PROB[a].keys()),
                         p=list(TRANS_PROB[a].values()))
        ns  = nxt(s, eff)
        r   = REWARDS.get(ns, 0.0)
        total += r
        path.append(ns); rew.append(r); cumrew.append(total)
        s = ns
        if s == (3,4): break
    return path, rew, cumrew

# On génère 3 épisodes enchaînés dans le GIF
episodes = []
for seed in [42, 7, 123]:
    p, r, cr = simulate_episode(seed=seed)
    episodes.append((p, r, cr))
    print(f"Épisode seed={seed}: {len(p)} steps, récomp. cumulée={cr[-1]:+.2f}")

# ── Préparation du GIF ────────────────────────────────────────────────────────
vmin = V_star.min(); vmax = V_star.max()
vcenter = np.clip(0.0, vmin + 1e-6, vmax - 1e-6)
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

# Construire la liste de toutes les frames (épisodes séparés par une pause)
all_frames = []
for ep_idx, (path, rew, cumrew) in enumerate(episodes):
    for step_idx in range(len(path)):
        all_frames.append((ep_idx, step_idx, path, rew, cumrew))
    # Pause de 4 frames entre épisodes
    for _ in range(4):
        all_frames.append((ep_idx, len(path)-1, path, rew, cumrew))

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6), facecolor='#1a1a2e')
gs  = fig.add_gridspec(1, 2, width_ratios=[3,1], wspace=0.06)
ax_grid  = fig.add_subplot(gs[0])
ax_stats = fig.add_subplot(gs[1])

def draw_cell(ax, x, y, robot_here=False, is_visited=False):
    col, row = y-1, x-1
    if (x,y) == OBSTACLE:
        rect = patches.FancyBboxPatch((col+0.05,row+0.05),0.9,0.9,
               boxstyle="round,pad=0.03",
               facecolor='#2c2c54', edgecolor='#666', linewidth=2, zorder=1)
        ax.add_patch(rect)
        ax.text(col+0.5, row+0.5, 'MUR', ha='center', va='center',
                color='#777', fontsize=10, fontweight='bold')
        return

    s = (x, y); vi = S2I[s]
    c = plt.cm.RdYlGn(norm(V_star[vi]))
    alpha = 0.95 if not is_visited else 0.7

    goal  = s==(3,4); trap = s==(2,4)
    ec    = '#ffd700' if goal else ('#ff4040' if trap else '#555')
    lw    = 3 if (goal or trap) else 1.5

    rect = patches.FancyBboxPatch((col+0.05,row+0.05),0.9,0.9,
           boxstyle="round,pad=0.03",
           facecolor=(*c[:3], alpha), edgecolor=ec, linewidth=lw, zorder=1)
    ax.add_patch(rect)

    # V* value
    ax.text(col+0.5, row+0.74,f"V*={V_star[vi]:.2f}",
            ha='center',va='center',fontsize=8,color='white',fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2,foreground='black')], zorder=2)

    # Action arrow
    arr = ARROWS[ACTIONS[policy[vi]]]
    ax.text(col+0.5, row+0.42, arr, ha='center', va='center',
            fontsize=20, color='white', zorder=2,
            path_effects=[pe.withStroke(linewidth=3,foreground='black')])

    # Reward label
    if s in REWARDS:
        rc = '#ffd700' if REWARDS[s]>0 else '#ff6b6b'
        ax.text(col+0.5, row+0.15,f"R={REWARDS[s]:+.0f}",
                ha='center',va='center',fontsize=9,color=rc,fontweight='bold',zorder=2)

    # State coords
    ax.text(col+0.08, row+0.94,f"({x},{y})",
            ha='left',va='top',fontsize=6.5,color='#aaa',zorder=2)

    # Special labels
    if goal:
        ax.text(col+0.88, row+0.88,'🏆',ha='right',va='top',fontsize=12,zorder=3)
    if trap:
        ax.text(col+0.88, row+0.88,'⚠',ha='right',va='top',fontsize=12,zorder=3)

    # Robot
    if robot_here:
        circ = patches.Circle((col+0.5, row+0.5), 0.28,
                               facecolor='#00d4ff', edgecolor='white',
                               linewidth=2, zorder=4, alpha=0.9)
        ax.add_patch(circ)
        ax.text(col+0.5, row+0.5,'🤖',ha='center',va='center',fontsize=18,zorder=5)

def draw_frame(frame_idx):
    ep_idx, step_idx, path, rew, cumrew = all_frames[frame_idx]

    # ── Grille ────────────────────────────────────────────────────────────────
    ax_grid.clear()
    ax_grid.set_facecolor('#16213e')
    ax_grid.set_xlim(-0.05, N_COLS+0.05)
    ax_grid.set_ylim(-0.05, N_ROWS+0.05)
    ax_grid.set_aspect('equal')
    ax_grid.set_xticks([]); ax_grid.set_yticks([])
    ax_grid.set_title(
        f"  MDP Grid World — Épisode {ep_idx+1}  |  Étape {step_idx}",
        color='white', fontsize=13, fontweight='bold', pad=8,
        loc='left'
    )

    # Trajet parcouru (trait cyan)
    visited = set(path[:step_idx])
    for k in range(min(step_idx, len(path)-1)):
        x1,y1 = path[k]; x2,y2 = path[k+1]
        alpha = max(0.15, 0.6 * (k+1)/max(step_idx,1))
        ax_grid.plot([y1-0.5, y2-0.5],[x1-0.5, x2-0.5],
                     color='#00d4ff', alpha=alpha, linewidth=2.5, zorder=3)

    robot_pos = path[step_idx]
    for x in range(1,N_ROWS+1):
        for y in range(1,N_COLS+1):
            draw_cell(ax_grid, x, y,
                      robot_here=((x,y)==robot_pos),
                      is_visited=(x,y) in visited)

    # Lignes de grille
    for v in range(N_COLS+1):
        ax_grid.axvline(v, color='#333355', linewidth=0.8, zorder=0)
    for h in range(N_ROWS+1):
        ax_grid.axhline(h, color='#333355', linewidth=0.8, zorder=0)

    # ── Stats ─────────────────────────────────────────────────────────────────
    ax_stats.clear()
    ax_stats.set_facecolor('#16213e')
    ax_stats.axis('off')

    s = robot_pos
    vi = S2I.get(s, S2I[(3,4)])
    cr = cumrew[step_idx]
    r  = rew[step_idx]

    def txt(x, y, label, val, cval='#00d4ff', size=10):
        ax_stats.text(0.05, y, label, color='#888888', fontsize=size-1,
                      transform=ax_stats.transAxes, va='top')
        ax_stats.text(0.96, y, val, color=cval, fontsize=size,
                      fontweight='bold', transform=ax_stats.transAxes,
                      va='top', ha='right')

    ax_stats.text(0.5, 0.97,'📊 STATS', ha='center', va='top',
                  color='white', fontsize=12, fontweight='bold',
                  transform=ax_stats.transAxes)
    ax_stats.axhline(0, color='#444', linewidth=1)

    y0 = 0.88
    dy = 0.09
    txt(0,y0-0*dy,'Épisode',   f"{ep_idx+1} / {len(episodes)}")
    txt(0,y0-1*dy,'Étape',     f"{step_idx}")
    txt(0,y0-2*dy,'État',      f"{s}")
    txt(0,y0-3*dy,'V*(état)',  f"{V_star[vi]:.4f}", '#00ff9f')
    txt(0,y0-4*dy,'Action π*', ARROWS[ACTIONS[policy[vi]]], '#ffdd57')

    ax_stats.plot([0.05, 0.95], [y0-5*dy-0.005]*2, color='#444', linewidth=0.8,
                  transform=ax_stats.transAxes)

    rc = '#ff6b6b' if r<0 else ('#00ff9f' if r>0 else '#aaa')
    txt(0,y0-5.3*dy,'Récompense', f"{r:+.1f}", rc)
    txt(0,y0-6.3*dy,'Σ Récomp.',  f"{cr:+.2f}", '#ffd700', size=11)

    ax_stats.plot([0.05, 0.95], [y0-7.3*dy-0.005]*2, color='#444', linewidth=0.8,
                  transform=ax_stats.transAxes)

    txt(0,y0-7.6*dy,'γ',        f"{GAMMA}", '#aaa', size=9)
    txt(0,y0-8.4*dy,'ε',        f"{EPSILON:.0e}", '#aaa', size=9)
    txt(0,y0-9.2*dy,'V*(1,1)',   f"{V_star[S2I[(1,1)]]:.4f}", '#aaa', size=9)
    txt(0,y0-10*dy, '|S|',      f"{N_S} états", '#aaa', size=9)

    if s == (3,4):
        ax_stats.text(0.5, 0.06,'🏆 OBJECTIF !', ha='center', va='bottom',
                      color='#ffd700', fontsize=13, fontweight='bold',
                      transform=ax_stats.transAxes)

    fig.patch.set_facecolor('#1a1a2e')

ani = FuncAnimation(fig, draw_frame, frames=len(all_frames),
                    interval=600, repeat=True)

out_path = 'mdp_simulation.gif'
print(f"\nSauvegarde du GIF ({len(all_frames)} frames)...")
ani.save(out_path, writer=PillowWriter(fps=2), dpi=110)
plt.close()
print(f"✅  GIF enregistré : {out_path}")
