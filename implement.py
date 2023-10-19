import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

class CDVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_classes):
        super(CDVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
       
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # Ekleme: aktivasyon katmanı
            nn.Sigmoid() # veya nn.Softmax(dim=-1)
        )

    def encode(self, x, y):
      
        xy = torch.cat([x, y], dim=1)
       
        h = self.encoder(xy)
        
        return self.mean(h), self.logvar(h)

    def reparameterize(self, mean, logvar):
       
        epsilon = torch.randn_like(mean)
        
        return mean + torch.exp(0.5 * logvar) * epsilon

    def decode(self, z, y):
        
        zy = torch.cat([z, y], dim=1)
        
        return self.decoder(zy)

    def forward(self, x, y):
        
        mean, logvar = self.encode(x, y)
        
        z = self.reparameterize(mean, logvar)
        
        x_hat = self.decode(z, y)
        
        return x_hat, mean, logvar


input_dim = 28 * 28 # Dimension of MNIST images
latent_dim = 10 # Dimension of latent variable z
hidden_dim = 256 # Dimension of hidden layers
num_classes = 10 # Number of classes in MNIST
num_epochs = 100 # Number of training epochs
batch_size = 128 # Batch size for training
learning_rate = 0.001 # Learning rate for optimizer


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False,
                   transform=transforms.ToTensor()),
    batch_size=batch_size)

model = CDVAE(input_dim, latent_dim, hidden_dim, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def loss_function(x, x_hat, mean, logvar):
    
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    return BCE + KLD


for epoch in range(num_epochs):
    
    model.train()
   
    train_loss = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        
        x = x.view(-1, input_dim)
        y = F.one_hot(y, num_classes=num_classes).float()
        
        optimizer.zero_grad()
        
        x_hat, mean, logvar = model(x, y)
        
        loss = loss_function(x, x_hat, mean, logvar)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader.dataset)
    
    print(f'Epoch {epoch}, Train loss: {train_loss:.4f}')


model.eval()

test_loss = 0
test_acc = 0

with torch.no_grad():
    for x, y in test_loader:
        
        x = x.view(-1, input_dim)
        y = F.one_hot(y, num_classes=num_classes).float()
      
        x_hat, mean, logvar = model(x, y)
        
        test_loss += loss_function(x, x_hat, mean, logvar).item()
        test_acc += (x_hat.round() == x).float().mean().item()

test_loss /= len(test_loader.dataset)
test_acc /= len(test_loader)

print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')



index = np.random.randint(len(datasets))
x, y = datasets[index]
x = x.view(-1, input_dim)
y = F.one_hot(torch.tensor(y), num_classes=num_classes).float()


x_hat, mean, logvar = model(x, y)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title(f"Gerçek rakam: {y.argmax()}")
plt.imshow(x.view(28, 28), cmap="gray")
plt.subplot(1, 2, 2)
plt.title(f"Tahmin edilen rakam: {x_hat.round().argmax()}")
plt.imshow(x_hat.view(28, 28), cmap="gray")
plt.savefig("result.png")
