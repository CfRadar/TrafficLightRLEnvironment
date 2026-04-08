# 🚦 Adaptive Traffic Signal Control using Reinforcement Learning

Welcome to the **Adaptive Traffic Signal Control** project! This repository contains an intelligent traffic light system built using Reinforcement Learning (RL) that dynamically optimizes signal timings. Designed to reduce congestion, improve traffic flow, and ensure fairness across all lanes, this project is built for production and rapid experimentation using **OpenEnv** and **Hugging Face Spaces**.

---

## 🧠 Problem Statement

Traditional traffic signals operate on fixed timers, which leads to a host of real-world frustrations:
- 🚗 **Long waiting times** during non-peak hours.
- 🚦 **Inefficient traffic flow** that ignores real-time congestion.
- 😤 **Lane starvation**, where minor roads rarely get green signals, punishing drivers simply for their route choice.

This project solves these issues by treating traffic control as a dynamic optimization problem, using an RL-agent to actively balance loads.

---

## 🎯 Project Objectives

- **Minimize total vehicle queue length** across the intersection.
- **Maximize traffic flow efficiency** (throughput).
- **Ensure fairness across all directions**, preventing indefinitely red lights.
- **Avoid systemic gridlock and deadlocks** mathematically.

---

## ⚙️ Environment Overview

- **Architecture**: Custom RL environment built using **OpenEnv** standards.
- **Deployment**: Readily hosted via **Docker + Hugging Face Spaces**.
- **Simulation**: 4-way cross-intersection traffic simulation.

### 🧩 State Space
The agent observes an `Observation` containing real-time intersection telemetry:
- `north_queue`, `south_queue`, `east_queue`, `west_queue` (Queue lengths per lane)
- `current_signal_phase`
- `time_elapsed_in_phase`
- `cars_passed_last_step`

### 🎮 Action Space
Discrete actions controlling the active traffic light direction. Only one direction can securely be green at any given time to prevent collisions:
- **`0`**: Green light for North lane
- **`1`**: Green light for South lane
- **`2`**: Green light for East lane
- **`3`**: Green light for West lane

---

## 🔧 Setup & Installation

Follow these instructions to run the environment locally. These commands are tailored for **Windows PowerShell**.

### 1. Clone the Repository
```powershell
git clone https://github.com/CfRadar/TrafficLightRLEnvironment.git
cd TrafficLightRLEnvironment
```

### 2. Create and Activate a Virtual Environment
We recommend using standard `venv` or the ultra-fast `uv` package manager:
```powershell
# Using built-in venv:
python -m venv .venv

# Using uv (much faster!):
uv venv .venv
```

Activate the environment (Windows PowerShell):
```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
# OR using uv:
uv pip install -r requirements.txt
```

### 4. Setup Environment Variables
To connect your Local Agent to Hugging Face or an API provider, set up the required environment variables:
```powershell
$env:HF_TOKEN="your_huggingface_access_token_here"
$env:API_BASE_URL="your_environment_url_here" 
$env:MODEL_NAME="your_model_identifier"
```

Verify the setup by echoing the values to the terminal:
```powershell
echo "Token: $env:HF_TOKEN | API URL: $env:API_BASE_URL"
```

---

## ▶️ How to Run the Project

Once the environment is running via Docker or Hugging Face Spaces, you can trigger the agent loop using `inference.py`. 

```powershell
python env/inference.py
```

### What Happens Under the Hood?
1. **START Log**: The script initializes a connection, logs the chosen model, and resets the simulation to step 0. 
2. **STEP Logs**: The agent observes the intersection, queries the RL policy (via the API client), and submits an action (0-3). Terminal logs will output live step data natively explaining what happened.
3. **END Log**: Once the maximum steps are reached, a final normalized score `0.0` to `1.0` is logged.

### Expected Output
```json
STARTING EPISODE...
STEP 1: Agent activated signal 0 (North). Queues: [N:5, S:6, E:8, W:9] -> Reward: 0.81
STEP 2: Agent activated signal 2 (East). Queues: [N:1, S:8, E:2, W:11] -> Reward: 0.83
...
END OF EPISODE. Final Score: 0.88, Success: True
```

---

## 🌐 How to Use the Environment API

The environment exposes RESTful API endpoints natively structured for reinforcement learning loops. **External users can interact directly with the environment without an `HF_TOKEN`**, as token authentication is primarily used by the agent client making heavy LLM-inference calls. 

### Core Endpoints

#### 1. `POST /reset`
Initializes a new episode and returns the starting state.

#### 2. `POST /step`
Submits an action to the intersection and advances time by one tick.
**Request:**
```json
{
  "action": {
    "signal": 2
  }
}
```
**Response:**
```json
{
  "observation": {
    "north_queue": 2,
    "south_queue": 5,
    "east_queue": 0,
    "west_queue": 8,
    "current_signal_phase": 2,
    "time_elapsed_in_phase": 1,
    "cars_passed_last_step": 3
  },
  "reward": 0.87,
  "done": false,
  "info": {}
}
```

#### 3. `GET /state`
Retrieve the current intersection state at any point without advancing simulation time.

---

## ⚖️ Fairness Enforcement

### What is Lane Starvation?
Lane starvation occurs when an algorithm optimizes heavily for total throughput by keeping the primary/busy lanes permanently green, forcing minority/side traffic to wait forever. **In the real world, this leads to immense driver frustration and traffic violations.**

### How We Avoid It
Our algorithm deliberately introduces a **Soft Starvation Constraint**:
- **Reward Penalties for Wait Times:** As a lane's wait duration grows, a proportional ratiometric penalty is smoothly subtracted from the agent's reward. 
- **Balanced Action Selection:** The agent is mathematically pushed to service waiting lanes eventually, as the increasing starvation penalty will heavily outweigh simple throughput metrics.
- **Dynamic Adaptation:** Fixed cycles force unnecessary waits; our system responds organically based on queue accumulation instead of static times.

### The Outcome
**No lane is ignored.** Signals rotate intelligently to prioritize heavy traffic lanes but *guarantee* side streets get a green light. The system achieves maximum global throughput while remaining inherently fair regardless of the unevenness of traffic distribution.

---

## 🚦 Efficiency & Adaptation

- **Minimizing Congestion**: The continuous base reward is directly tied to an inverse ratio of the total intersection queue length. If total cars drop, the system earns high baseline rewards.
- **Real-Time Responsiveness**: The environment tracks structural changes in queue sizes sequentially. If an agent's action causes the queue to shrink, it gets a scaled `tanh` improvement bonus. If it worsens, it gets penalized.
- **Performance Phases**:
  - *Early High Rewards*: Usually means traffic starts low and the agent succeeds effortlessly.
  - *Mid-Phase Adaptation*: True challenge phase. The system learns how to tackle congestion buildups. Short dips in reward here represent real-world traffic waves.
  - *Stability*: A well-trained agent will flatline into stable high-scoring rewards, meaning it has discovered the optimal policy for the intersection.

---

## 📊 Result Interpretation

A score returned by `inference.py`, such as:
`success=true, score=0.75`

Means the following:
- **`score` ranges strictly from `0.0` to `1.0`**. 
- **0.85 - 1.00**: Incredible efficiency. Minimal waiting, near-perfect flow.
- **0.65 - 0.85**: Stable but leaving slight optimizations on the table.
- **Below 0.50**: The agent is struggling with congestion or hitting starvation penalties. 
- **Stability vs Randomness**: A consistent `0.80` across 50 steps is far superior to an agent that bounces wildly between `0.4` and `0.9`, as driver predictability is key in traffic control.

---

## 🧠 Design Decisions & Engineering Tradeoffs

- **Why Temporal Reward Smoothing?**
  RL agents struggle heavily with "noisy" rewards. A single car spawning randomly can instantly dump the queue score. We integrated an Exponential Moving Average (EMA) into the reward output to filter out single-step noise, establishing a steady gradient for the optimizer.
- **Why Avoid Large Penalties?**
  Instantly dropping a reward from `0.8` to `0.2` completely destabilizes learning algorithms and leads to policy collapse. All penalties are mathematically capped to small relative fractions (±0.15 limit).
- **Why is Fairness a Soft Constraint Instead of a Hard Rule?**
  Hard thresholds (e.g. "Lane MUST go green after 2 minutes") force agents into unnatural suboptimal behaviours. By mathematically blending fairness strictly into the reward matrix linearly, the agent learns to prioritize it naturally without breaking global throughput optimizations.

---
*Built with ❤️ for AI enthusiasts and Smart City planning.*
