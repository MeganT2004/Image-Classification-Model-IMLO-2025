from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

device = torch.device('cpu') #making sure the model isn't being trained on GPU

#Transforms
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load data
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

train_size = int(len(train_data)*0.9) #90/10 split of training and validation images
val_size = int(len(train_data)*0.1)

train_set, val_set = torch.utils.data.random_split( #splitting the data randomly between training and validation sets
    train_data,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_set.dataset.transform = val_transform

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)

# Model
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(6*6*64, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 6*6*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = TestNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss() #since we're using multiple categories

losses = []
start_time = time.time()

#Validation accuracy checker
def validate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    correct   = 0
    total     = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device) #ensure we're running on cpu
            outputs = model(data)

            val_loss += F.cross_entropy(outputs, target, reduction='sum').item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = val_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

best_val_loss = float('inf')
patience = 3
wait = 0

for epoch in range(10):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        losses.append(loss.item())

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if time.time() - start_time > 60 * 60 * 4: #staying within the 4 hour training time limit
            print("Reached 4-hour training limit. Stopping.")
            break

    #Checking accuracy and loss for each epoch on average    
    avg_loss = running_loss / len(train_loader)
    
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"\nEpoch {epoch+1} Training Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}")

    #Checking validation set to counter overfitting early
    val_loss, val_acc = validate(model, val_loader, device)
    print(f"Epoch {epoch+1}: Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}\n")

    if val_loss < (best_val_loss-0.25): #Since the jump in loss can be miniscule, I want to make sure there's a significant jump to not trigger early stop
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early Stopping - Validation Loss not improving.")
            break

#Saving the model to a different file so that all weights are properly accounted for.
torch.save(model.state_dict(), 'model.pth')

