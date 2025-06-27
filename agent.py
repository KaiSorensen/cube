import torch
import gym  # If you use OpenAI Gym-like design
from environment import RubiksCubeEnv  # Your environment

# Define the environment
env = RubiksCubeEnv()

# Define the neural network
class AgentNN(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(AgentNN, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 200)
        self.fc2 = torch.nn.Linear(200, 100)
        self.fc3 = torch.nn.Linear(100, 64)
        self.fc4 = torch.nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)  # Outputs Q-values or action probabilities

def train_agent():
    state_size = 54  # Depends on your state representation
    action_size = env.action_space.n       # Number of possible actions

    print("State size:", state_size)
    print("Action size:", action_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    agent = AgentNN(state_size, action_size).to(device)

    # Train the agent using an RL algorithm (e.g., DQN)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    print("Training the agent...")

    for episode in range(3000):  # Number of training episodes
        if episode % 100 == 0:
            print("Episode:", episode)
        state = env.reset()  # Initialize cube
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = agent(state_tensor)
            action = torch.argmax(q_values).item()
            next_state, reward, done, _ = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
            
            # compute loss and update the model
            target = reward + 0.99 * torch.max(agent(next_state_tensor))
            loss = loss_fn(q_values[action], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state  # Move to the next state


    print ("loss", loss)
    torch.save(agent.state_dict(), "agent.pth")  # Save the trained model


if __name__ == "__main__":
    train_agent()