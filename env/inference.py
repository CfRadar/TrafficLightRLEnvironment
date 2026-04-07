import os
import asyncio
from openai import OpenAI
from my_env_v4 import MyEnvV4Env, MyEnvV4Action

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

async def run_inference():
    task_name = "Adaptive Traffic Signal Control"
    env_name = "TrafficEnvV4"

    print(f"[START] task={task_name} env={env_name} model={MODEL_NAME}")

    env = MyEnvV4Env()
    rewards = []
    error_msg = "null"
    done = False
    step_count = 0

    try:
        if not API_KEY:
            raise ValueError("HF_TOKEN environment variable is not set.")

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        obs = await env.reset()

        while not done and step_count < 50:
            step_count += 1
            
            prompt = (
                f"Current state:\n"
                f"North queue: {obs.north_queue}\n"
                f"South queue: {obs.south_queue}\n"
                f"East queue: {obs.east_queue}\n"
                f"West queue: {obs.west_queue}\n"
                f"Current signal phase: {obs.current_signal_phase} (0=NS, 1=EW)\n"
                f"Time elapsed in phase: {obs.time_elapsed_in_phase}\n"
                f"Cars passed last step: {obs.cars_passed_last_step}\n"
                f"Goal:\n"
                f"Output exactly '0' to switch to/maintain North-South green.\n"
                f"Output exactly '1' to switch to/maintain East-West green.\n"
                f"Only output the integer 0 or 1."
            )

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )

            action_str = response.choices[0].message.content.strip()
            
            if action_str == "0":
                action_val = 0
            elif action_str == "1":
                action_val = 1
            else:
                if "0" in action_str:
                    action_val = 0
                elif "1" in action_str:
                    action_val = 1
                else:
                    action_val = obs.current_signal_phase
            
            action = MyEnvV4Action(signal=action_val)
            
            obs, reward, done, _ = await env.step(action)
            rewards.append(reward)

            action_log = str(action_val)
            print(f"[STEP] step={step_count} action={action_log} reward={reward:.2f} done={str(done).lower()} error=null")

    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('=', '')
        if step_count == 0:
            step_count = 1
        print(f"[STEP] step={step_count} action=null reward=0.00 done=false error={error_msg}")

    finally:
        await env.close()
        
        score = sum(rewards) / 50.0  # MAX_STEPS = 50
        score = max(0.0, min(1.0, score))
        success = score >= 0.6
        
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        
        print(f"[END] success={str(success).lower()} steps={step_count} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    asyncio.run(run_inference())
