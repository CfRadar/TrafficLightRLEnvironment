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
    def __init__(self, max_steps=50):
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
        self.MAX_THROUGHPUT = 6
        self.MAX_STEPS = max_steps

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
        switch_penalty = 0.0
        consistency_bonus = 0.0
        
        if action.signal != self.current_signal:
            switch_penalty = 0.05
            self.current_signal = action.signal
            self.time_elapsed = 0
        else:
            consistency_bonus = 0.02
            self.time_elapsed += 1

        self.north += random.randint(0, 3)
        self.south += random.randint(0, 3)
        self.east += random.randint(0, 3)
        self.west += random.randint(0, 3)

        cars_cleared = 0
        if self.current_signal == 0:
            n_clear = min(self.north, 6)
            self.north -= n_clear
            cars_cleared = n_clear
        elif self.current_signal == 1:
            s_clear = min(self.south, 6)
            self.south -= s_clear
            cars_cleared = s_clear
        elif self.current_signal == 2:
            e_clear = min(self.east, 6)
            self.east -= e_clear
            cars_cleared = e_clear
        else:
            w_clear = min(self.west, 6)
            self.west -= w_clear
            cars_cleared = w_clear

        for i in range(4):
            if i == self.current_signal:
                self.wait_times[i] = 0
            else:
                self.wait_times[i] += 1

        total_queue = self.north + self.south + self.east + self.west
        congestion_penalty = min(total_queue / self.MAX_QUEUE, 1.0)
        
        throughput_score = cars_cleared / self.MAX_THROUGHPUT
        
        MAX_WAIT = 10.0
        fairness_penalty = min(max(self.wait_times) / MAX_WAIT, 1.0)
        
        reward = 0.7 * (1.0 - congestion_penalty) + 0.2 * throughput_score + 0.1 * (1.0 - fairness_penalty)
        reward = reward - switch_penalty + consistency_bonus
        reward = max(0.0, min(1.0, reward))

        self.step_count += 1
        done = self.step_count >= self.MAX_STEPS

        obs = self._get_obs(cars_cleared)
        return obs, reward, done, {}

    async def close(self):
        pass
