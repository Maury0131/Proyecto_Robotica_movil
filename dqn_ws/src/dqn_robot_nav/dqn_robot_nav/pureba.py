import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

# 1. Configuración de Hardware
device = torch.device("cuda")
print(f"Probando potencia en: {torch.cuda.get_device_name(0)}")

# 2. Preparar Datos (MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=1024, # Batch grande para saturar la VRAM de la 5060
    shuffle=True
)

# 3. Definir una CNN (Red Neuronal Convolucional)
class BlackwellNet(nn.Module):
    def __init__(self):
        super(BlackwellNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = BlackwellNet().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)
criterion = nn.CrossEntropyLoss()

# 4. Bucle de Entrenamiento (Benchmarking)
print("\nIniciando entrenamiento a máxima velocidad...")
start_time = time.time()

model.train()
for epoch in range(1, 6): # 5 épocas para ver consistencia
    epoch_start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # Envío a GPU
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    epoch_duration = time.time() - epoch_start
    print(f"Época {epoch} completada en: {epoch_duration:.4f} segundos")

total_time = time.time() - start_time
print(f"\n--- RESULTADOS ---")
print(f"Tiempo total para 5 épocas: {total_time:.2f}s")
print(f"Velocidad: {60000 * 5 / total_time:.0f} imágenes por segundo")