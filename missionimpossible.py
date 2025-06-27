

from time import sleep
import torch
from agent import AgentNN 
from environment import RubiksCubeEnv

print("MISSION IMPOSSIBLE")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

state_size = 54
action_size = 12

# Load the trained model
agent = AgentNN(state_size, action_size).to(device)
agent.load_state_dict(torch.load("agent.pth", map_location=device))
agent.eval()  # Set the model to evaluation mode

env = RubiksCubeEnv()
state = env.reset()  # Get the initial scrambled state
done = False

while not done:
    # Convert the state to a tensor
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    
    # Get Q-values or action probabilities from the model
    q_values = agent(state_tensor)
    
    # Choose the best action
    action = torch.argmax(q_values).item()
    
    # Take the action in the environment
    state, reward, done, info = env.step(action)
    
    # Optionally, print or visualize the state
    print(f"Action: {action}, Reward: {reward}")
    sleep(1)  # Slow down for visualization
