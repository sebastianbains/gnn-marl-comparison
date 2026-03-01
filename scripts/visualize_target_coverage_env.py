#!/usr/bin/env python3
# visualize target coverage environment
# creates a diagram showing the grid, agents, targets, and communication ranges
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# environment parameters
GRID_SIZE = 14
NUM_AGENTS = 7
NUM_TARGETS = 15
COMM_RANGE = 3.5

# set random seed for reproducibility
np.random.seed(42)

# generate agent positions (random but spread out)
agent_positions = []
while len(agent_positions) < NUM_AGENTS:
    pos = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
    # ensure agents aren't too close to each other
    if all(np.sqrt((pos[0]-p[0])**2 + (pos[1]-p[1])**2) > 2 for p in agent_positions):
        agent_positions.append(pos)

# generate target positions (random)
target_positions = []
while len(target_positions) < NUM_TARGETS:
    pos = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
    # ensure targets aren't on agent positions
    if pos not in agent_positions and pos not in target_positions:
        target_positions.append(pos)

# create figure
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_xlim(-0.5, GRID_SIZE - 0.5)
ax.set_ylim(-0.5, GRID_SIZE - 0.5)
ax.set_aspect('equal')

# draw grid
ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
ax.set_xticks(range(GRID_SIZE))
ax.set_yticks(range(GRID_SIZE))

# color palette for agents (different colors)
agent_colors = plt.cm.Set3(np.linspace(0, 1, NUM_AGENTS))

# draw communication range for one example agent (agent 0)
example_agent_idx = 3  # Show comm range for agent 4
example_agent_pos = agent_positions[example_agent_idx]
comm_circle = plt.Circle(example_agent_pos, COMM_RANGE, 
                         color='lightblue', alpha=0.15, zorder=1)
ax.add_patch(comm_circle)
comm_circle_border = plt.Circle(example_agent_pos, COMM_RANGE, 
                                fill=False, color='lightblue', 
                                linestyle='--', linewidth=1.5, alpha=0.4, zorder=1)
ax.add_patch(comm_circle_border)

# add arrow and label for communication range
arrow_end_x = example_agent_pos[0] + COMM_RANGE * 0.7
arrow_end_y = example_agent_pos[1] + COMM_RANGE * 0.7
ax.annotate('', xy=(arrow_end_x, arrow_end_y), 
            xytext=(example_agent_pos[0], example_agent_pos[1]),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(example_agent_pos[0] + 0.3, example_agent_pos[1] + COMM_RANGE - 1.5,
        f'Communication\nRange ({COMM_RANGE})',
        fontsize=10, color='blue', ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='blue'))

# draw targets (yellow stars)
for target_pos in target_positions:
    ax.plot(target_pos[0], target_pos[1], marker='*', markersize=25, 
            color='gold', markeredgecolor='orange', markeredgewidth=1.5, zorder=3)

# draw agents (colored circles with numbers)
for i, (agent_pos, color) in enumerate(zip(agent_positions, agent_colors)):
    # agent circle
    circle = plt.Circle(agent_pos, 0.35, color=color, 
                       edgecolor='navy', linewidth=2.5, zorder=4)
    ax.add_patch(circle)
    
    # agent number
    ax.text(agent_pos[0], agent_pos[1], str(i+1), 
            fontsize=14, fontweight='bold', ha='center', va='center', 
            color='black', zorder=5)

# add legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markeredgecolor='navy', markersize=12, markeredgewidth=2, label='Agent'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
               markeredgecolor='orange', markersize=18, markeredgewidth=1.5, label='Target'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
               markeredgecolor='lightblue', markersize=15, 
               linestyle='--', markeredgewidth=1.5, label='Comm. Range')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
          framealpha=0.95, edgecolor='gray')

# add environment specifications box
specs_text = (
    f"Environment Specifications:\n"
    f"• Grid Size: {GRID_SIZE} × {GRID_SIZE}\n"
    f"• Number of Agents: {NUM_AGENTS}\n"
    f"• Number of Targets: {NUM_TARGETS}\n"
    f"• Communication Range: {COMM_RANGE} units\n"
    f"• Observation: Local\n"
    f"• Reward: Global (shared)"
)
ax.text(0.02, 0.15, specs_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'),
        family='monospace')

# labels and title
ax.set_xlabel('X Coordinate', fontsize=13, fontweight='bold')
ax.set_ylabel('Y Coordinate', fontsize=13, fontweight='bold')
ax.set_title(f'Target Coverage Environment: {GRID_SIZE}×{GRID_SIZE} Grid with {NUM_AGENTS} Agents',
             fontsize=15, fontweight='bold', pad=15)

# save figure
output_dir = Path('results/observability_sweep/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'target_coverage_environment_7agents.png'

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved visualization to: {output_path}")

# also save to results root for easy access
output_path_root = Path('results/target_coverage_environment_7agents.png')
plt.savefig(output_path_root, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved visualization to: {output_path_root}")

plt.close()
