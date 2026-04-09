import math
import random
from dataclasses import dataclass
from typing import Tuple

@dataclass
class MyEnvV4Action:
    signal: int

@dataclass
class Observation:
    north_queue: int
    south_queue: int
    east_queue: int
    west_queue: int
    current_signal_phase: int
    time_elapsed_in_phase: int
    cars_passed_last_step: int

class MyEnvV4Env:
    def __init__(self, max_steps=50, task="medium"):
        self.north = 0
        self.south = 0
        self.east = 0
        self.west = 0
        self.wait_times = [0, 0, 0, 0]
        self.current_signal = 0
        self.time_elapsed = 0
        self.step_count = 0
        self._last_heuristic_action = None
        self.MAX_QUEUE = 40
        self.MAX_THROUGHPUT = 10
        self.MAX_STEPS = max_steps
        
        # New state variables for advanced reward structure
        self.action_history = []
        self.prev_total_queue = 0
        self.previous_reward = 0.0
        self.task = task.lower() if task else "medium"

    async def reset(self) -> Observation:
        self.north = 0
        self.south = 0
        self.east = 0
        self.west = 0
        self.wait_times = [0, 0, 0, 0]
        self.current_signal = 0
        self.time_elapsed = 0
        self.step_count = 0
        self._last_heuristic_action = None
        
        self.action_history = []
        self.prev_total_queue = 0
        self.previous_reward = 0.0
        
        return self._get_obs(0)

    def _get_obs(self, cars_cleared: int) -> Observation:
        return Observation(
            north_queue=self.north,
            south_queue=self.south,
            east_queue=self.east,
            west_queue=self.west,
            current_signal_phase=self.current_signal,
            time_elapsed_in_phase=self.time_elapsed,
            cars_passed_last_step=cars_cleared
        )

    async def step(self, action: MyEnvV4Action) -> Tuple[Observation, float, bool, dict]:
        # Update action history
        self.action_history.append(action.signal)
        if len(self.action_history) > 5:
            self.action_history.pop(0)

        if action.signal != self.current_signal:
            self.current_signal = action.signal
            self.time_elapsed = 0
        else:
            self.time_elapsed += 1

        if self.task == "easy":
            self.north += random.randint(0, 1)
            self.south += random.randint(0, 1)
            self.east += random.randint(0, 1)
            self.west += random.randint(0, 1)
        elif self.task == "hard":
            # Imbalanced and high traffic
            self.north += random.randint(1, 5)
            self.south += random.randint(1, 5)
            self.east += random.randint(0, 2)
            self.west += random.randint(0, 2)
        else:
            # medium
            self.north += random.randint(0, 3)
            self.south += random.randint(0, 3)
            self.east += random.randint(0, 3)
            self.west += random.randint(0, 3)

        cars_cleared = 0
        if self.current_signal == 0:
            n_clear = min(self.north, self.MAX_THROUGHPUT)
            self.north -= n_clear
            cars_cleared = n_clear
        elif self.current_signal == 1:
            s_clear = min(self.south, self.MAX_THROUGHPUT)
            self.south -= s_clear
            cars_cleared = s_clear
        elif self.current_signal == 2:
            e_clear = min(self.east, self.MAX_THROUGHPUT)
            self.east -= e_clear
            cars_cleared = e_clear
        else:
            w_clear = min(self.west, self.MAX_THROUGHPUT)
            self.west -= w_clear
            cars_cleared = w_clear

        current_queues = [self.north, self.south, self.east, self.west]
        total_queue = sum(current_queues)

        for i in range(4):
            if i == self.current_signal:
                self.wait_times[i] = 0
            else:
                self.wait_times[i] += 1

        # 2. BASE REWARD (PRIMARY SIGNAL)
        max_possible_queue = self.MAX_QUEUE * 4
        total_queue_ratio = min(1.0, total_queue / max_possible_queue)
        base_reward = 1.0 - total_queue_ratio

        # 3. SMOOTH IMPROVEMENT BONUS
        scale_factor = 10.0
        delta = self.prev_total_queue - total_queue
        improvement_bonus = math.tanh(delta / scale_factor) * 0.1
        self.prev_total_queue = total_queue

        # 4. SOFT PENALTIES
        repetition_count = self.action_history.count(action.signal)
        repetition_penalty = min(0.1, repetition_count * 0.02)

        MAX_ALLOWED_WAIT = 10.0
        max_wait_time = max(self.wait_times)
        starvation_penalty = min(0.15, (max_wait_time / MAX_ALLOWED_WAIT) * 0.1)

        max_lane_queue = max(current_queues)
        queue_penalty = min(0.15, (max_lane_queue / self.MAX_QUEUE) * 0.15)

        # 6. FINAL REWARD
        raw_reward = base_reward + improvement_bonus - repetition_penalty - starvation_penalty - queue_penalty
        
        # Clamp to strictly [0.0, 1.0] before smoothing
        current_reward = max(0.0, min(1.0, raw_reward))

        # 5. TEMPORAL SMOOTHING
        smoothed_reward = 0.7 * current_reward + 0.3 * self.previous_reward
        self.previous_reward = smoothed_reward
        
        reward = smoothed_reward

        self.step_count += 1
        done = self.step_count >= self.MAX_STEPS

        obs = self._get_obs(cars_cleared)
        return obs, reward, done, {}

    async def close(self):
        pass
