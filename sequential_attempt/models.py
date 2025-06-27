from bad.csvdata import loadData

import torch
import torch.nn as nn
import torch.optim as optim

modelType = "seq" # Change this to 'seq' for sequential model or 'cnn' for convolutional model
if modelType != "seq" and modelType != "cnn":
    raise ValueError("model type must be 'seq' (sequential) or 'cnn' (convolutional) dumbass")

# the sequential model
class RubiksCubeSequential(nn.Module):
    def __init__(self):
        super(RubiksCubeSequential, self).__init__()
        self.fc1 = nn.Linear(54, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 13)  # 13 outputs (12 moves + solved state)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.view(-1, 54)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# the convolutional model
class RubiksCubeCNN(nn.Module):
    def __init__(self):
        super(RubiksCubeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 13)  # 13 outputs (12 moves + solved state)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 4 * 3)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# Create the model
if modelType == "seq":
    model = RubiksCubeSequential()
elif modelType == 'cnn':
    model = RubiksCubeCNN()


# Define the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

labels, states = loadData()
print("labels[234]: ", labels[234])
print("states[234]: ", states[234])
# Convert `states` (list of lists) to a tensor
labels = torch.tensor(labels, dtype=torch.long).to('cuda')  # Move labels to CUDA
states_tensor = torch.tensor(states, dtype=torch.float32).to('cuda')  # Common preprocessing step
if modelType == "seq":
    inputs = states_tensor  # No additional reshaping for sequential model
elif modelType == 'cnn':
    inputs = states_tensor.view(-1, 1, 6, 9).to('cuda')  # Reshape for CNN


# Move model to GPU
model = model.to('cuda')

# # Move inputs and labels to GPU
# inputs = inputs.to('cuda')
# labels = labels.to('cuda')

# Training loop
for epoch in range(1):  # Number of epochs
    optimizer.zero_grad()          # Clear gradients
    outputs = model(inputs)        # Forward pass
    loss = criterion(outputs, labels)  # Compute loss
    loss.backward()                # Backward pass
    optimizer.step()               # Update weights

    # Calculate accuracy
    with torch.no_grad():  # No gradients needed for accuracy calculation
        predictions = torch.argmax(outputs, dim=1)  # Get predicted class for each sample
        correct = (predictions == labels).sum().item()  # Count correct predictions
        total = labels.size(0)  # Total number of samples
        accuracy = correct / total * 100  # Calculate accuracy percentage
    
    # Print loss and accuracy
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")



# save model
if modelType == "seq":
    torch.save(model.state_dict(), 'seq.pth')
elif modelType == 'cnn':
    torch.save(model.state_dict(), 'cnn.pth')
