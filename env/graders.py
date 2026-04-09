def __base_grader(trajectory) -> float:
    if not trajectory:
        return 0.0
    
    try:
        last_step = trajectory[-1]
        
        # Determine if it's an object or dict
        if hasattr(last_step, "observation"):
            obs = last_step.observation
        elif isinstance(last_step, dict) and "observation" in last_step:
            obs = last_step["observation"]
        else:
            return 0.5
            
        if hasattr(obs, "north_queue"):
            total_queue = obs.north_queue + obs.south_queue + obs.east_queue + obs.west_queue
        elif isinstance(obs, dict) and "north_queue" in obs:
            total_queue = obs["north_queue"] + obs["south_queue"] + obs["east_queue"] + obs["west_queue"]
        else:
            return 0.5
            
        MAX_TOTAL_QUEUE = 40 * 4  # From my_env_v4.py max_queue is 40 per lane
        score = max(0.0, min(1.0, 1.0 - (total_queue / MAX_TOTAL_QUEUE)))
        return float(score)
    except Exception:
        return 0.0

def easy_grader(trajectory) -> float:
    return __base_grader(trajectory)

def medium_grader(trajectory) -> float:
    return __base_grader(trajectory)

def hard_grader(trajectory) -> float:
    return __base_grader(trajectory)
