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
    def __init__(self):
        self.north = 0
        self.south = 0
        self.east = 0
        self.west = 0
        self.current_signal = 0
        self.time_elapsed = 0
        self.step_count = 0
        self.MAX_QUEUE = 40
        self.MAX_THROUGHPUT = 6

    async def reset(self) -> Observation:
        self.north = 0
        self.south = 0
        self.east = 0
        self.west = 0
        self.current_signal = 0
        self.time_elapsed = 0
        self.step_count = 0
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
        if action.signal != self.current_signal:
            switch_penalty = 0.1
            self.current_signal = action.signal
            self.time_elapsed = 0
        else:
            self.time_elapsed += 1

        self.north += random.randint(0, 3)
        self.south += random.randint(0, 3)
        self.east += random.randint(0, 3)
        self.west += random.randint(0, 3)

        cars_cleared = 0
        if self.current_signal == 0:
            n_clear = min(self.north, 3)
            s_clear = min(self.south, 3)
            self.north -= n_clear
            self.south -= s_clear
            cars_cleared = n_clear + s_clear
        else:
            e_clear = min(self.east, 3)
            w_clear = min(self.west, 3)
            self.east -= e_clear
            self.west -= w_clear
            cars_cleared = e_clear + w_clear

        total_queue = self.north + self.south + self.east + self.west
        congestion_penalty = min(total_queue / self.MAX_QUEUE, 1.0)
        
        throughput_score = cars_cleared / self.MAX_THROUGHPUT
        
        reward = (0.6 * (1.0 - congestion_penalty)) + (0.4 * throughput_score) - switch_penalty
        reward = max(0.0, min(1.0, reward))

        self.step_count += 1
        done = self.step_count >= 50

        obs = self._get_obs(cars_cleared)
        return obs, reward, done, {}

    async def close(self):
        pass
