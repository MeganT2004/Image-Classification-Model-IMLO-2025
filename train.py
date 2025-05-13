from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

device = torch.device('cpu') #making sure the model isn't being trained on GPU

#Transforms
train_transform1 = transforms.Compose([ #No Augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_transform2 = transforms.Compose([ #Flip and Colour Adjustment
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_transform3 = transforms.Compose([ #Vertical Flip and cropping
    transforms.RandomVerticalFlip(p=0.2), #Orientation matters in cifar10, so not a ton of vertical flips
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

])

train_transform4 = transforms.Compose([ #Auto Augmentation
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02,0.2)),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

#Load data
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform1)

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

#Model
class TestNet(nn.Module):
    def __init__(self, pFC):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32) #Batch Normalisation
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.bn3   = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(4*4*128, 128) #Fully Connected layer
        self.fc_drop  = nn.Dropout(p=pFC)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1) #Dynamic flattening, in case I want to add another layer
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        x = self.fc2(x)
        return x

epochs = 100

model = TestNet(pFC = 0.5).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss('''label_smoothing=0.1''') #since we're using multiple categories

scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.03,
        total_steps=epochs * len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

losses = []
startTime = time.time()
stage2 = False
stage3 = False
stage4 = False

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

            valLoss += F.cross_entropy(outputs, target, reduction='sum').item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avgLoss = valLoss / total
    accuracy = 100.0 * correct / total
    return avgLoss, accuracy

bestValLoss = float('inf')
patience = 7
wait = 0

for epoch in range(epochs):
    model.train()
    runningLoss = 0.0

    if epoch <= 30: #dropout decreases over time, more neurons are deactivated randomly at first
        pFC = 0.5 - (0.3 * (epoch/30))
    elif epoch == 50:
        pFC = 0.5
        wait = 0
    elif epoch <= 80 and epoch >= 50:
        pFC = 0.5 - (0.3 * ((epoch-50)/30))

    model.fc_drop.p = pFC

    print(f"Epoch {epoch+1}, pFC: {pFC:.2f}")

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
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
    valLoss, valAcc = validate(model, val_loader)
    print(f"Epoch {epoch+1}: Val Acc: {valAcc:.2f}%, Val Loss: {valLoss:.4f}\n")

    if valLoss < (bestValLoss - 0.03): #Since the jump in loss can be miniscule, I want to make sure there's a significant jump to not trigger early stop
        bestValLoss = valLoss
        torch.save(model.state_dict(), 'model.pth') #Updating model weights in case of crash
        wait = 0
        '''if wait > 0:
            wait -= 1'''
    else:
        wait += 1
        if wait >= patience: #Switching to augmented datasets when validation accuracy doesn't improve
            '''if stage2 == False:
                stage2 = True
                train_set.dataset.transform = train_transform2
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
                wait = 0
                best_val_loss = float('inf')
                print("Stage 2 Training Set in Use\n")
                continue
            elif stage2 == True and stage3 == False:
                stage3 = True
                train_set.dataset.transform = train_transform3
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
                wait = 0
                best_val_loss = float('inf')
                print("Stage 3 Training Set in Use\n")
                continue
            elif stage3 == True and stage4 == False:
                stage4 = True
                train_set.dataset.transform = train_transform4
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
                wait = 0
                best_val_loss = float('inf')
                print("Stage 4 Training Set in Use\n")
                continue'''
            print("Early Stopping - Validation Loss not improving.")
            break

#Saving the model to a different file so that all weights are properly accounted for.
torch.save(model.state_dict(), 'model.pth')

