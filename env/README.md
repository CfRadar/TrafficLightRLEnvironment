---
title: Traffic Environment Server
emoji: 🚦
colorFrom: red
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - traffic-control
---

# Adaptive Traffic Signal Control - OpenEnv

This directory contains the production-ready OpenEnv environment for the Adaptive Traffic Signal Control project (`TrafficEnvV4`). It provides an OpenAI Gym-like interface integrated perfectly with Hugging Face via the OpenEnv standard.

## 🚀 Tasks & Difficulties

The environment is configured with three distinct multi-task difficulty levels, accessible dynamically via the OpenEnv task loader:

- **Easy**: Low traffic inflow (averaging 2 cars/step). Ideal for baseline testing.
- **Medium**: Moderate traffic inflow (averaging 6 cars/step). A balanced test requiring solid signal coordination.
- **Hard**: High and imbalanced traffic inflow (averaging 8 cars/step, heavily biased towards North/South). Requires strict lane prioritization to avoid catastrophic queues.

All tasks are processed against deterministic graders located in `graders.py` which dynamically score the agent's performance strictly out of `1.00`.

## ⚙️ Environment Mechanics

- **Intersection Type**: 4-way cross-intersection.
- **Maximum Outflow**: `10` cars can be safely cleared per step in the green direction.
- **Reward Function**: Operates gracefully to penalize massive queue build-ups and lane starvation while applying temporal smoothing to prevent erratic policy behavior.

## 🏃 Running Inference

A local language model loop via the Hugging Face Inference API is built into `inference.py`. 
You can run the full suite directly:

```bash
python inference.py
```

This will automatically instantiate the tasks (`easy`, `medium`, `hard`) from `openenv.yaml` sequentially, executing 100 simulation steps each. Upon completion of each task, the grader mathematically evaluates the queue backlog efficiency and assigns a deterministic final score.

## 📦 File Overview

- `openenv.yaml` - Core configuration, port mapping, and task definitions.
- `my_env_v4.py` - Core logical structure, queue mechanics, flow logic, and reward algorithms.
- `graders.py` - Fully isolated metric extractors validating the evaluation outcomes.
- `inference.py` - Custom loop integrating OpenEnv task loading securely with your LLM.
- `server/app.py` - FastAPI application serving the environment over JSON/HTTP and WebSockets.
