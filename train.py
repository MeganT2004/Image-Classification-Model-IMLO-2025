from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.beta import Beta
import time

device = torch.device('cpu') #making sure the model isn't being trained on GPU

#Transforms
train_transform1 = transforms.Compose([ #No Augmentation, for tests
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_transform2 = transforms.Compose([ #Big Transformations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomCrop(32, padding=4),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02,0.2)),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)) #standard for CIFAR10 
)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
])

#Load data
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform2)

train_size = int(len(train_data)*0.9) #90/10 split of training and validation images
val_size = int(len(train_data)*0.1)

train_set, val_set = torch.utils.data.random_split( #splitting the data randomly between training and validation sets
    train_data,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_set.dataset.transform = val_transform

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)

#Model
class TestNet(nn.Module):
    def __init__(self, pFC):        
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32) #Batch Normalisation
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(4*4*256, 512) #Fully Connected layer
        self.fc_drop  = nn.Dropout(p=pFC)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1) #Dynamic flattening, in case I want to add another layer
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        x = self.fc2(x)
        return x

epochs = 200

model = TestNet(pFC = 0.6).to(device)
#optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) #since we're using multiple categories

scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,
        total_steps=epochs * len(train_loader),
        pct_start=0.10,
        div_factor=10,
        anneal_strategy='cos'
    )

losses = []
startTime = time.time()

def cutmixData(x, y, alpha=1.0):
    if alpha > 0:
        lam = Beta(alpha, alpha).sample()
    else:
        lam = torch.tensor(1.0, device=x.device)

    B, C, H, W = x.size()
    index = torch.randperm(B, device=x.device)
    y_a, y_b = y, y[index]

    cutRat = torch.sqrt(1.0 - lam)
    cutW = (W * cutRat).to(torch.int).item()
    cutH = (H * cutRat).to(torch.int).item()

    cx = torch.randint(0, W, (1,), device=x.device).item()
    cy = torch.randint(0, H, (1,), device=x.device).item()

    bbx1 = max(cx - cutW // 2, 0)
    bbx2 = min(cx + cutW // 2, W)
    bby1 = max(cy - cutH // 2, 0)
    bby2 = min(cy + cutH // 2, H)

    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / float(W * H))
    return x, y_a, y_b, lam

#Validation accuracy checker
def validate(model, val_loader):
    model.eval()
    valLoss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device) #ensure we're running on cpu
            outputs = model(data)

            batch_loss = F.cross_entropy(outputs, target).item()   # default is mean
            valLoss  += batch_loss
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avgValLoss = valLoss / len(val_loader)
    accuracy = 100.0 * correct / total
    return avgValLoss, accuracy

bestValLoss = float('inf')
patience = 10
wait = 0

for epoch in range(epochs):
    model.train()
    runningLoss = 0.0

    if epoch <= 30: #dropout decreases over time, more neurons are deactivated randomly at first
        pFC = 0.6 - (0.3 * (epoch/30))

    model.fc_drop.p = pFC

    print(f"Epoch {epoch+1}, pFC: {pFC:.2f}")

    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        if torch.rand(1) < 0.5: #50/50 split of cutmix and normally augmented data
            data, targetsA, targetsB, lam = cutmixData(data, targets, alpha=1.0)
            outputs = model(data)
            loss = lam*criterion(outputs, targetsA) + (1-lam)*criterion(outputs, targetsB)
        else:
            outputs = model(data)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        runningLoss += loss.item()
        losses.append(loss.item())

        #if batch_idx % 100 == 0:
            #print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if time.time() - startTime > 60 * 60 * 4: #staying within the 4 hour training time limit
            print("Reached 4-hour training limit. Stopping.")
            break

    #Checking accuracy and loss for each epoch on average    
    avgLoss = runningLoss / len(train_loader)
    
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
    print(f"\nEpoch {epoch+1} Training Accuracy: {accuracy:.2f}%, Average Loss: {avgLoss:.4f}")

    #Checking validation set to counter overfitting early
    avgValLoss, valAcc = validate(model, val_loader)
    print(f"Epoch {epoch+1}: Val Acc: {valAcc:.2f}%, Val Loss: {avgValLoss:.4f}\n")

    if avgValLoss < (bestValLoss): #Since the jump in loss can be miniscule, I want to make sure there's a significant jump to not trigger early stop
        bestValLoss = avgValLoss
        torch.save(model.state_dict(), 'model.pth') #Updating model weights in case of crash
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early Stopping - Validation Loss not improving.")
            break

#Saving the model to a different file so that all weights are properly accounted for.
torch.save(model.state_dict(), 'model.pth')

