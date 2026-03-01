# Multi-Agent Coordination with Graph Neural Networks

Research comparing graph-based and independent learning approaches for multi-agent tasks.

## Overview

This project investigates how communication range affects multi-agent coordination performance. Two algorithms are compared:
- Graph Neural Networks (GNN) with message passing
- Independent Q-Learning (IQL) baseline

## Key Finding

GNN performance depends on communication range, with optimal results at moderate ranges rather than maximum connectivity.

## Installation

```bash
git clone https://github.com/yourusername/gnn-marl-comparison.git
cd gnn-marl-comparison
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the observability sweep experiment:
```bash
python observability_sweep_experiment.py
```

Analyze results:
```bash
python scripts/analyze_observability_sweep.py results/observability_sweep/observability_sweep_results_*.json
python scripts/statistical_analysis.py
```

## Results

The observability sweep tested 6 communication ranges with 20 runs each. GNN showed peak performance at range 2.5, with 32% improvement over IQL (p=0.0018, d=1.016).

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- NumPy, SciPy
