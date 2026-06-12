import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(
    root = "data",
    transform = transforms
)
loader = DataLoader(
    dataset,
    batch_size = 3,
    shuffle=True
)
class SimpleCNN(nn.Module):
    def __init__(self):
        super((SimpleCNN),self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32 , kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*32,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )
    def forward(self,x):
            x = self.conv(x)
            x = self.fc(x)
            return x
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
epochs = 10
for epoch in range(epochs):
    loss_items = 0
    for image , label in loader:
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        loss_items += loss.item()
print(f"Loss:{loss_items/len(loader):.4f}")
if not os.path.exists("model"):
     os.makedirs("model")
torch.save(model.state_dict(),"model/simple_cnn.pth")
print("Successfully trained!")
