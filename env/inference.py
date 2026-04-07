import os
import asyncio
from openai import OpenAI
from my_env_v4 import MyEnvV4Env, MyEnvV4Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
MAX_STEPS = int(os.getenv("MAX_STEPS", "150"))
import random

async def run_inference():
    task_name = "Adaptive Traffic Signal Control"
    env_name = "TrafficEnvV4"

    print(f"[START] task={task_name} env={env_name} model={MODEL_NAME}")

    env = MyEnvV4Env(max_steps=MAX_STEPS)
    rewards = []
    error_msg = "null"
    done = False
    step_count = 0

    try:
        if not API_KEY:
            raise ValueError("HF_TOKEN environment variable is not set.")

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        obs = await env.reset()

        while not done and step_count < MAX_STEPS:
            step_count += 1
            
            prompt = (
                f"Current state:\n"
                f"North queue: {obs.north_queue}\n"
                f"South queue: {obs.south_queue}\n"
                f"East queue: {obs.east_queue}\n"
                f"West queue: {obs.west_queue}\n"
                f"Current signal phase: {obs.current_signal_phase} (0=North, 1=South, 2=East, 3=West)\n"
                f"Time elapsed in phase: {obs.time_elapsed_in_phase}\n"
                f"Cars passed last step: {obs.cars_passed_last_step}\n"
                f"Goal:\n"
                f"Output exactly '0' to switch to/maintain North green.\n"
                f"Output exactly '1' to switch to/maintain South green.\n"
                f"Output exactly '2' to switch to/maintain East green.\n"
                f"Output exactly '3' to switch to/maintain West green.\n"
                f"Only output the integer 0, 1, 2, or 3."
            )

            # 1. Compute priority score for fairness with load-aware modes
            queues = [
                obs.north_queue,
                obs.south_queue,
                obs.east_queue,
                obs.west_queue
            ]
            
            total_queue = sum(queues)
            HIGH_THRESHOLD = 20
            
            mode = "congested" if total_queue >= HIGH_THRESHOLD else "normal"
            
            alpha = 0.5
            MAX_WAIT = 10.0
            
            priorities = []
            for i in range(4):
                base_priority = float(queues[i])
                
                if mode == "normal":
                    norm_wait = env.wait_times[i] / MAX_WAIT
                    score = base_priority + (alpha * norm_wait)
                    if env.wait_times[i] > MAX_WAIT:
                        score += 5.0
                else:
                    # Congested mode: pure efficiency, ignore fairness
                    score = base_priority
                    
                priorities.append(score)
                
            # Optional boost for congested mode
            if mode == "congested":
                max_index = queues.index(max(queues))
                priorities[max_index] += 10.0
                
            max_priority = max(priorities)
            
            # Smooth switching logic
            if env._last_heuristic_action is not None:
                last_act = env._last_heuristic_action
                if max_priority - priorities[last_act] <= 2.0:
                    heuristic_action = last_act
                else:
                    heuristic_action = priorities.index(max_priority)
            else:
                heuristic_action = priorities.index(max_priority)
                
            env._last_heuristic_action = heuristic_action

            # 2. Parse model output safely
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0
                )
                model_action_str = response.choices[0].message.content.strip()
                
                model_action = None
                if model_action_str in ["0", "1", "2", "3"]:
                    model_action = int(model_action_str)
                else:
                    for option in ["0", "1", "2", "3"]:
                        if option in model_action_str:
                            model_action = int(option)
                            break
            except Exception:
                model_action = None

            # 3. Final decision logic
            if model_action not in [0, 1, 2, 3]:
                action_val = heuristic_action
            else:
                if random.random() < 0.7:
                    action_val = heuristic_action
                else:
                    action_val = model_action
            
            # CRITICAL FIX: always define action_str
            action_str = str(action_val)
            
            action = MyEnvV4Action(signal=action_val)
            
            obs, reward, done, _ = await env.step(action)
            rewards.append(reward)

            print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('=', '')
        if step_count == 0:
            step_count = 1
        print(f"[STEP] step={step_count} action=null reward=0.00 done=false error={error_msg}")

    finally:
        await env.close()
        
        score = sum(rewards) / float(MAX_STEPS)
        score = max(0.0, min(1.0, score))
        success = score >= 0.6
        
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        
        print(f"[END] success={str(success).lower()} steps={step_count} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    asyncio.run(run_inference())
