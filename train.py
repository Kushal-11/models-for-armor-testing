import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from code.model_def import Net  # Import the model definition


# --- 1. Load and Preprocess CIFAR-10 Dataset ---

# Define transforms for training (including augmentation) and testing
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


# Download and create datasets
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=test_transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                           shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
                                          shuffle=False, num_workers=4)

# --- 2. Define a Simple Convolutional Neural Network (CNN) ---

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Three convolutional layers with increasing depth
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers for classification
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Block 1: Conv -> ReLU -> Pool
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # Block 2: Conv -> ReLU -> Pool
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Block 3: Conv -> ReLU -> Pool
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model and move it to GPU if available
net = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# --- 3. Define Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# --- 4. Train the Network ---

num_epochs = 10  # Adjust epochs as needed

for epoch in range(num_epochs):
    net.train()  # Set model to training mode
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move inputs and labels to GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()        # Reset gradients for this batch
        outputs = net(inputs)        # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()              # Backpropagation
        optimizer.step()             # Update weights
        
        running_loss += loss.item()
        # Print status every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss: {running_loss/100:.3f}")
            running_loss = 0.0

    # Evaluate on test set after each epoch
    net.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1} Test Accuracy: {accuracy:.2f}%")

print("Finished Training")

# --- 5. Save the Trained Model ---
torch.save(net.state_dict(), "model.pth")
print("Model saved to model.pth")