import importlib.util
import sys
import os
from osim.env import L2M2019Env


env = L2M2019Env(visualize=False,difficulty=2)

# number 159
# initializing agent
agent_path = "111000104_hw4_test.py"
module_name = agent_path.replace('/', '.').replace('.py', '')
spec = importlib.util.spec_from_file_location(module_name, agent_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
Agent = getattr(module, 'Agent')

# evaluating
import time
from tqdm import tqdm

total_reward = 0
total_time = 0
agent = Agent()
time_limit = 120
max_timesteps = env.spec.timestep_limit - 1 

for episode in tqdm(range(10), desc="Evaluating"):
    obs = env.reset()
    start_time = time.time()
    episode_reward = 0
    timestep = 0
    
    while True:
        action = agent.act(obs) 

        obs, reward, done, info = env.step(action)
        episode_reward += reward
        timestep += 1

        if timestep >= max_timesteps:
            print(f"Max timestep reached for episode {episode}, reward: {episode_reward}")
            break

        if time.time() - start_time > time_limit:
            print(f"Time limit reached for episode {episode}")
            break
        

        if done:
            break

    end_time = time.time()
    total_reward += episode_reward
    total_time += (end_time - start_time)


score = total_reward / 10
print(f"Final Score: {score}")
