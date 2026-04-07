import uuid
import uvicorn
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

from my_env_v4 import MyEnvV4Env, MyEnvV4Action

class TrafficAction(Action):
    signal: int = Field(default=0, description="0 for NS, 1 for EW")

class TrafficObservation(Observation):
    north_queue: int = 0
    south_queue: int = 0
    east_queue: int = 0
    west_queue: int = 0
    current_signal_phase: int = 0
    time_elapsed_in_phase: int = 0
    cars_passed_last_step: int = 0

class TrafficEnvironmentAdapter(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    def __init__(self):
        self.env = MyEnvV4Env()
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

    def reset(self, seed=None, episode_id=None, **kwargs) -> TrafficObservation:
        import asyncio
        obs = asyncio.run(self.env.reset())
        self._state.step_count = 0
        return TrafficObservation(
            north_queue=obs.north_queue,
            south_queue=obs.south_queue,
            east_queue=obs.east_queue,
            west_queue=obs.west_queue,
            current_signal_phase=obs.current_signal_phase,
            time_elapsed_in_phase=obs.time_elapsed_in_phase,
            cars_passed_last_step=obs.cars_passed_last_step,
            reward=0.0,
            done=False
        )

    def step(self, action: TrafficAction, timeout_s=None, **kwargs) -> TrafficObservation:
        import asyncio
        v4_action = MyEnvV4Action(signal=action.signal)
        obs, reward, done, _ = asyncio.run(self.env.step(v4_action))
        self._state.step_count += 1
        return TrafficObservation(
            north_queue=obs.north_queue,
            south_queue=obs.south_queue,
            east_queue=obs.east_queue,
            west_queue=obs.west_queue,
            current_signal_phase=obs.current_signal_phase,
            time_elapsed_in_phase=obs.time_elapsed_in_phase,
            cars_passed_last_step=obs.cars_passed_last_step,
            reward=float(reward),
            done=bool(done)
        )

    @property
    def state(self) -> State:
        return self._state

# Create the app with web interface and README integration
app = create_app(
    TrafficEnvironmentAdapter,
    TrafficAction,
    TrafficObservation,
    env_name="traffic-env",
    max_concurrent_envs=4
)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
